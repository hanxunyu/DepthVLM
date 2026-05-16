"""Extract RGB images, sparse LiDAR depth maps, and camera intrinsics from Argoverse 2.

Output structure:
    out_folder/{train,val,test}/rgb/{log_id}/{timestamp_ns}_{cam_name}.jpg
    out_folder/{train,val,test}/depth/{log_id}/{timestamp_ns}_{cam_name}.png  (uint16, depth_m * 256)
    out_folder/{train,val,test}/intrinsics/{log_id}/{timestamp_ns}_{cam_name}.json  ([fx, fy, cx, cy, W, H])
    out_folder/{train,val,test}/index.jsonl  (per-frame metadata)

Usage:
    python extract_rgb_depth_argoverse2.py \\
        --root_folder /path/to/argoverse_raw \\
        --out_folder /path/to/argoverse \\
        --num_workers 64

    python extract_rgb_depth_argoverse2.py \\
        --root_folder /path/to/argoverse_raw \\
        --out_folder /path/to/argoverse \\
        --num_workers 32 \\
        --intrinsics_only
"""

import argparse
import json
import logging
import os
from functools import partial
from multiprocessing import Pool, Value, Lock
from pathlib import Path
from typing import Final

import av2.utils.io as io_utils
import numpy as np
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.utils.typing import NDArrayInt
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

RING_CAMERA_FPS: Final[int] = 20
# Depth save scale: depth_m × 256 → uint16 PNG
# Max representable depth: 65535 / 256 ≈ 255.9 m (covers VLP-32C 200 m range)
# Precision: 1/256 ≈ 3.9 mm
# Read back: depth_m = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
DEPTH_SAVE_SCALE: Final[float] = 256.0

CAMERAS_USED = [
    "ring_front_left",
    "ring_front_right",
    "ring_rear_left",
    "ring_rear_right",
    "ring_side_left",
    "ring_side_right",
    "ring_front_center",
    # "stereo_front_left",    # AV2 SDK's get_log_pinhole_camera does not support stereo cameras
    # "stereo_front_right",   # AV2 SDK's get_log_pinhole_camera does not support stereo cameras
]


def get_immediate_subfolders(folder_path: str) -> list:
    """Return a list of immediate subfolders in the given folder path."""
    return [f.name for f in Path(folder_path).iterdir() if f.is_dir()]


# ---- Per-worker global: each worker initialises its own loader ----
_worker_loader = None
_worker_split_dir = None


def _worker_init(split_dir: str):
    """Initializer for each pool worker: create a private AV2SensorDataLoader."""
    global _worker_loader, _worker_split_dir
    _worker_split_dir = split_dir
    _worker_loader = AV2SensorDataLoader(
        data_dir=Path(split_dir), labels_dir=Path(split_dir)
    )


def process_one_log(log_id: str, split: str, out_folder: str,
                    frame_sample_interval: int, intrinsics_only: bool = False) -> dict:
    """Process all cameras for a single log.

    Returns a dict with:
        count: number of frames saved
        index_entries: list of per-timestamp index dicts for index.jsonl
    """
    loader = _worker_loader
    split_dir = _worker_split_dir
    count = 0
    index_entries = []

    # Collect all unique timestamps across cameras to build per-timestamp index
    timestamp_cameras: dict = {}  # {timestamp_ns: set(cam_names)}

    for cam_name in CAMERAS_USED:
        cam_im_fpaths = loader.get_ordered_log_cam_fpaths(log_id, cam_name)
        if len(cam_im_fpaths) == 0:
            continue
        sampled_cam_im_fpaths = cam_im_fpaths[::frame_sample_interval]

        for frame_idx, im_fpath in enumerate(sampled_cam_im_fpaths):
            try:
                cam_timestamp_ns = int(im_fpath.stem)

                # --- Construct 2-level relative path: {log_id}/{timestamp_ns}_{cam_name} ---
                relative_stem = f"{log_id}/{cam_timestamp_ns}_{cam_name}"

                # --- Get camera intrinsics for image size ---
                pinhole_camera = loader.get_log_pinhole_camera(
                    log_id=log_id, cam_name=cam_name
                )
                img_h = pinhole_camera.intrinsics.height_px
                img_w = pinhole_camera.intrinsics.width_px

                # --- Save intrinsics (original, no rescaling) ---
                K = pinhole_camera.intrinsics.K
                intrinsics_data = [
                    float(K[0, 0]),  # fx
                    float(K[1, 1]),  # fy
                    float(K[0, 2]),  # cx
                    float(K[1, 2]),  # cy
                    img_w,           # width
                    img_h,           # height
                ]
                intrinsics_out_path = os.path.join(out_folder, split, "intrinsics", relative_stem + ".json")
                os.makedirs(os.path.dirname(intrinsics_out_path), exist_ok=True)
                with open(intrinsics_out_path, "w") as f:
                    json.dump(intrinsics_data, f)

                if intrinsics_only:
                    count += 1
                    timestamp_cameras.setdefault(cam_timestamp_ns, set()).add(cam_name)
                    continue

                # --- Check ego pose ---
                city_SE3_ego = loader.get_city_SE3_ego(log_id, cam_timestamp_ns)
                if city_SE3_ego is None:
                    continue

                # --- Find closest LiDAR sweep ---
                lidar_fpath = loader.get_closest_lidar_fpath(
                    log_id, cam_timestamp_ns
                )
                if lidar_fpath is None:
                    continue

                lidar_timestamp_ns = int(lidar_fpath.stem)

                # --- Read LiDAR points ---
                lidar_points_ego = io_utils.read_lidar_sweep(
                    lidar_fpath, attrib_spec="xyz"
                )

                # --- Project LiDAR to image ---
                (
                    uv,
                    points_cam,
                    is_valid_points,
                ) = loader.project_ego_to_img_motion_compensated(
                    points_lidar_time=lidar_points_ego,
                    cam_name=cam_name,
                    cam_timestamp_ns=cam_timestamp_ns,
                    lidar_timestamp_ns=lidar_timestamp_ns,
                    log_id=log_id,
                )

                if is_valid_points is None or uv is None or points_cam is None:
                    continue
                if is_valid_points.sum() == 0:
                    continue

                uv_int: NDArrayInt = np.round(uv[is_valid_points]).astype(np.int32)
                points_cam_valid = points_cam[is_valid_points]

                # --- Build sparse depth map ---
                euclidean_depth = np.sqrt(
                    (points_cam_valid ** 2).sum(axis=1)
                ).astype(np.float32)

                depth_map = np.zeros((img_h, img_w), dtype=np.float32)
                us = np.clip(uv_int[:, 0], 0, img_w - 1)
                vs = np.clip(uv_int[:, 1], 0, img_h - 1)
                depth_map[vs, us] = euclidean_depth

                # --- Save RGB image (original, no rescaling) ---
                rgb_out_path = os.path.join(out_folder, split, "rgb", relative_stem + ".jpg")
                os.makedirs(os.path.dirname(rgb_out_path), exist_ok=True)
                rgb_image = Image.open(im_fpath)
                rgb_image.save(rgb_out_path)

                # --- Save depth map as uint16 PNG (depth_m × DEPTH_SAVE_SCALE) ---
                depth_out_path = os.path.join(out_folder, split, "depth", relative_stem + ".png")
                os.makedirs(os.path.dirname(depth_out_path), exist_ok=True)
                depth_uint16 = (depth_map * DEPTH_SAVE_SCALE).clip(0, 65535).astype(np.uint16)
                Image.fromarray(depth_uint16).save(depth_out_path)

                count += 1
                timestamp_cameras.setdefault(cam_timestamp_ns, set()).add(cam_name)

            except Exception as e:
                print(f"Error processing {im_fpath}: {e}")
                continue

    # Build index entries (one per unique timestamp)
    for ts_ns in sorted(timestamp_cameras.keys()):
        index_entries.append({
            "scene_name": log_id,
            "timestamp": ts_ns,
            "split": split,
            "cameras": sorted(timestamp_cameras[ts_ns]),
            "source_dataset": "argoverse2",
        })

    return {"count": count, "index_entries": index_entries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract RGB images and sparse LiDAR depth maps from Argoverse 2."
    )
    parser.add_argument("--root_folder", type=str, required=True,
                        help="Argoverse 2 sensor dataset root folder")
    parser.add_argument("--out_folder", type=str, required=True,
                        help="Output root folder ({split}/rgb/ and {split}/depth/ will be created inside)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--intrinsics_only", action="store_true",
                        help="Only save camera intrinsics (skip RGB and depth extraction)")
    args = parser.parse_args()

    root_folder = args.root_folder
    out_folder = args.out_folder
    num_workers = args.num_workers

    frame_sample_interval = 1

    splits = sorted(get_immediate_subfolders(root_folder))
    print(f"Found splits: {splits}")

    total_count = 0
    for split in splits:
        split_dir = os.path.join(root_folder, split)
        folders = sorted(get_immediate_subfolders(split_dir))
        print(f"Split [{split}]: {len(folders)} logs, using {num_workers} workers")

        # Each worker initializes its own AV2SensorDataLoader in _worker_init
        worker_fn = partial(
            process_one_log,
            split=split,
            out_folder=out_folder,
            frame_sample_interval=frame_sample_interval,
            intrinsics_only=args.intrinsics_only,
        )

        with Pool(processes=num_workers, initializer=_worker_init,
                  initargs=(split_dir,)) as pool:
            results = list(tqdm(
                pool.imap_unordered(worker_fn, folders),
                total=len(folders),
                desc=f"Logs ({split})",
            ))

        split_count = sum(r["count"] for r in results)
        total_count += split_count
        print(f"Split [{split}] done. Frames processed: {split_count}")

        # Write index.jsonl for this split
        all_index_entries = []
        for r in results:
            all_index_entries.extend(r["index_entries"])
        all_index_entries.sort(key=lambda e: (e["scene_name"], e["timestamp"]))

        index_path = os.path.join(out_folder, split, "index.jsonl")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        with open(index_path, "w") as f:
            for entry in all_index_entries:
                json.dump(entry, f)
                f.write("\n")
        print(f"Wrote {len(all_index_entries)} entries to {index_path}")

    print(f"Done. Total frames processed: {total_count}")
