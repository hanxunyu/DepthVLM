# Extract RGB images, sparse LiDAR depth maps, and camera intrinsics from Waymo Open Dataset (v2 parquet format).
# Reference: utils/curate_waymo.py
# Usage: python extract_rgb_depth_waymo.py --dataset_dir <...> --out_folder <...> [--num_workers 8]
#
# Output structure:
#   out_folder/{split}/rgb/segment_context_name/timestamp_camera.jpg
#   out_folder/{split}/depth/segment_context_name/timestamp_camera.png  (uint16, depth_m × 256)
#   out_folder/{split}/intrinsics/segment_context_name/timestamp_camera.json  ([fx, fy, cx, cy, W, H])
#   out_folder/{split}/index.jsonl  (per-frame metadata for traceability)
"""
Example usage:

# Process a single split:
python extract_rgb_depth_waymo.py \
    --dataset_dir /path/to/waymo_raw \
    --out_folder /path/to/waymo_out \
    --num_workers 8 \
    --splits training

# Process all splits (default):
python extract_rgb_depth_waymo.py \
    --dataset_dir /path/to/waymo_raw \
    --out_folder /path/to/waymo_out \
    --num_workers 32
"""

import argparse
import io
import json
import logging
import os
import warnings
from functools import partial
from multiprocessing import Pool
from typing import Final

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)

from waymo_open_dataset import v2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.v2.perception.utils import lidar_utils

logger = logging.getLogger(__name__)

# Depth save scale: depth_m × 256 → uint16 PNG
# Max representable depth: 65535 / 256 ≈ 255.9 m (covers Waymo LiDAR range)
# Precision: 1/256 ≈ 3.9 mm
# Read back: depth_m = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
DEPTH_SAVE_SCALE: Final[float] = 256.0


def read_parquet(dataset_dir: str, tag: str, context_name: str):
    """Read a single parquet file directly with pandas (no Dask, no glob)."""
    import pandas as pd
    path = os.path.join(dataset_dir, tag, f"{context_name}.parquet")
    return pd.read_parquet(path)


def undistort_image(pil_image, intrinsic):
    """Undistort an image using the camera intrinsic parameters.

    Returns:
        undistorted_pil_image: undistorted PIL image
        new_K: new 3x3 camera intrinsic matrix
        map_x, map_y: undistortion maps (for remapping pixel coordinates)
    """
    cv_image = np.array(pil_image)
    fx = intrinsic.f_u
    fy = intrinsic.f_v
    cx = intrinsic.c_u
    cy = intrinsic.c_v
    k1 = intrinsic.k1
    k2 = intrinsic.k2
    p1 = intrinsic.p1
    p2 = intrinsic.p2
    k3 = intrinsic.k3

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist_coeffs = np.array([k1, k2, p1, p2, k3])

    h, w = cv_image.shape[:2]
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), 1, (w, h))
    new_K[0, 0] = fx
    new_K[1, 1] = fy
    new_K[0, 2] = w / 2
    new_K[1, 2] = h / 2

    map_x, map_y = cv2.initUndistortRectifyMap(K, dist_coeffs, None, new_K, (w, h), 5)
    undistorted_image = cv2.remap(cv_image, map_x, map_y, cv2.INTER_LINEAR)
    undistorted_pil_image = Image.fromarray(undistorted_image)

    return undistorted_pil_image, new_K, map_x, map_y


def process_one_file(filename: str, dataset_dir: str, out_folder: str, intrinsics_only: bool = False) -> dict:
    """Process all rows (camera images) for a single parquet file.
    Returns a dict with count and index_entries.
    """
    # Check .done flag — skip if already completed
    done_dir = os.path.join(out_folder, ".done")
    done_flag = os.path.join(done_dir, filename + ".done")
    if os.path.exists(done_flag):
        print(f"[Worker {os.getpid()}] Skip (already done): {filename}", flush=True)
        return {"count": 0, "index_entries": [], "skipped": True}

    count = 0
    index_entries = []
    timestamp_cameras: dict = {}  # {timestamp_micros: set(camera_name)}
    print(f"[Worker {os.getpid()}] Start processing: {filename}", flush=True)

    try:
        lidar = read_parquet(dataset_dir, "lidar", filename)
        lidar_calib = read_parquet(dataset_dir, "lidar_calibration", filename)
        camera_calib = read_parquet(dataset_dir, "camera_calibration", filename)
        lidar_pose = read_parquet(dataset_dir, "lidar_pose", filename)
        vehicle_pose = read_parquet(dataset_dir, "vehicle_pose", filename)
        cam_img = read_parquet(dataset_dir, "camera_image", filename)
        lidar_camera_projection = read_parquet(dataset_dir, "lidar_camera_projection", filename)

        df = v2.merge(lidar_calib, lidar)
        df = v2.merge(df, lidar_camera_projection)
        df = v2.merge(df, lidar_pose)
        df = v2.merge(df, vehicle_pose)
        df = v2.merge(df, camera_calib)
        df = v2.merge(df, cam_img)
    except Exception as e:
        print(f"Error reading parquet for {filename}: {e}")
        return {"count": 0, "index_entries": []}

    for _, row in df.iterrows():
        try:
            # Create all component objects
            lidar_comp = v2.LiDARComponent.from_dict(row)
            lidar_calib_comp = v2.LiDARCalibrationComponent.from_dict(row)
            camera_calib_comp = v2.CameraCalibrationComponent.from_dict(row)
            lidar_pose_comp = v2.LiDARPoseComponent.from_dict(row)
            vehicle_pose_comp = v2.VehiclePoseComponent.from_dict(row)
            camera_image_comp = v2.CameraImageComponent.from_dict(row)
            lidar_cam_proj_comp = v2.LiDARCameraProjectionComponent.from_dict(row)

            # --- Decode and undistort RGB image ---
            pil_image = Image.open(io.BytesIO(camera_image_comp.image))
            undistorted_pil_image, new_K, map_x, map_y = undistort_image(
                pil_image, camera_calib_comp.intrinsic
            )

            # --- Construct output paths ---
            relative_name = (
                f"{camera_image_comp.key.segment_context_name}/"
                f"{camera_image_comp.key.frame_timestamp_micros}"
                f"_{camera_image_comp.key.camera_name}.jpg"
            )
            relative_stem = os.path.splitext(relative_name)[0]

            # --- Save intrinsics (undistorted, original resolution) ---
            img_w, img_h = undistorted_pil_image.size
            intrinsics_data = [
                float(new_K[0, 0]),  # fx
                float(new_K[1, 1]),  # fy
                float(new_K[0, 2]),  # cx
                float(new_K[1, 2]),  # cy
                img_w,               # width
                img_h,               # height
            ]
            intrinsics_out_path = os.path.join(out_folder, "intrinsics", relative_stem + ".json")
            os.makedirs(os.path.dirname(intrinsics_out_path), exist_ok=True)
            with open(intrinsics_out_path, "w") as f:
                json.dump(intrinsics_data, f)

            if not intrinsics_only:
                # --- Build depth image from range image ---
                range_image_cartesian = lidar_utils.convert_range_image_to_cartesian(
                    range_image=lidar_comp.range_image_return1,
                    calibration=lidar_calib_comp,
                    pixel_pose=lidar_pose_comp.range_image_return1,
                    frame_pose=vehicle_pose_comp,
                )
                extrinsic = np.reshape(
                    camera_calib_comp.extrinsic.transform, [1, 4, 4]
                ).astype(np.float32)
                camera_image_size = (camera_calib_comp.height, camera_calib_comp.width)

                ric_shape = range_image_cartesian.shape
                ric = np.reshape(
                    range_image_cartesian,
                    [1, ric_shape[0], ric_shape[1], ric_shape[2]],
                )

                cp = lidar_cam_proj_comp.range_image_return1
                cp_tensor = tf.reshape(tf.convert_to_tensor(value=cp.values), cp.shape)
                cp_tensor = tf.cast(cp_tensor, tf.int32)
                cp_shape = cp_tensor.shape
                cp_tensor = np.reshape(
                    cp_tensor, [1, cp_shape[0], cp_shape[1], cp_shape[2]]
                )

                depth_image = range_image_utils.build_camera_depth_image(
                    ric,
                    extrinsic,
                    cp_tensor,
                    list(camera_image_size),
                    camera_image_comp.key.camera_name,
                )
                depth_image_np = depth_image.numpy().squeeze(axis=0)  # (H, W)

                # --- Undistort depth map using the same maps ---
                depth_undistorted = cv2.remap(
                    depth_image_np.astype(np.float32),
                    map_x, map_y,
                    cv2.INTER_NEAREST,
                    borderValue=0,
                )

                # --- Save RGB image ---
                rgb_out_path = os.path.join(out_folder, "rgb", relative_name)
                os.makedirs(os.path.dirname(rgb_out_path), exist_ok=True)
                undistorted_pil_image.save(rgb_out_path)

                # --- Save depth map as uint16 PNG (depth_m × DEPTH_SAVE_SCALE) ---
                depth_out_path = os.path.join(out_folder, "depth", relative_stem + ".png")
                os.makedirs(os.path.dirname(depth_out_path), exist_ok=True)
                depth_uint16 = (depth_undistorted * DEPTH_SAVE_SCALE).clip(0, 65535).astype(np.uint16)
                Image.fromarray(depth_uint16).save(depth_out_path)

            count += 1
            ts = camera_image_comp.key.frame_timestamp_micros
            cam = camera_image_comp.key.camera_name
            timestamp_cameras.setdefault(ts, set()).add(cam)

        except Exception as e:
            print(f"Error processing row in {filename}: {e}")
            continue

    print(f"[Worker {os.getpid()}] Finished {filename}: {count} frames", flush=True)

    # Build index entries (one per unique timestamp)
    for ts in sorted(timestamp_cameras.keys()):
        index_entries.append({
            "scene_name": filename,
            "timestamp": ts,
            "cameras": sorted(timestamp_cameras[ts]),
            "source_dataset": "waymo",
        })

    # Write .done flag to mark this segment as fully processed
    os.makedirs(done_dir, exist_ok=True)
    with open(done_flag, "w") as f:
        f.write(f"{count}\n")

    return {"count": count, "index_entries": index_entries, "skipped": False}


ALL_SPLITS = ["training", "validation", "testing", "testing_location"]


def process_split(
    split: str,
    dataset_dir: str,
    out_folder: str,
    num_workers: int,
    start_file: int,
    end_file: int,
    intrinsics_only: bool = False,
):
    """Process a single split (training / validation / testing / testing_location)."""
    split_dataset_dir = os.path.join(dataset_dir, split)
    split_out_folder = os.path.join(out_folder, split)

    camera_image_dir = os.path.join(split_dataset_dir, "camera_image")
    if not os.path.isdir(camera_image_dir):
        print(f"[Skip] {camera_image_dir} does not exist, skipping split '{split}'.")
        return 0

    filenames = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(camera_image_dir)
        if f.endswith(".parquet")
    ])

    ef = end_file if end_file > 0 else len(filenames)
    filenames = filenames[start_file:ef]

    total_files = len(filenames)

    # Filter out already-completed segments (check .done flags)
    done_dir = os.path.join(split_out_folder, ".done")
    already_done = set()
    if os.path.isdir(done_dir):
        already_done = {
            os.path.splitext(f)[0]
            for f in os.listdir(done_dir)
            if f.endswith(".done")
        }
    filenames_todo = [f for f in filenames if f not in already_done]

    print(f"\n{'='*60}")
    print(f"Split: {split}")
    print(f"Found {total_files} parquet files in {camera_image_dir}")
    print(f"Already done: {len(already_done)}, remaining: {len(filenames_todo)}")
    if len(filenames_todo) > 0:
        print(f"Sample: {filenames_todo[:3]}")
    print(f"{'='*60}")

    if len(filenames_todo) == 0:
        print(f"[{split}] All files already processed, nothing to do.")
        return 0

    worker_fn = partial(
        process_one_file,
        dataset_dir=split_dataset_dir,
        out_folder=split_out_folder,
        intrinsics_only=intrinsics_only,
    )

    with Pool(processes=num_workers, maxtasksperchild=1) as pool:
        results = list(tqdm(
            pool.imap_unordered(worker_fn, filenames_todo),
            total=len(filenames_todo),
            desc=f"[{split}] Parquet files ({len(already_done)} skipped)",
        ))

    split_count = sum(r["count"] for r in results)

    # Write index.jsonl for this split
    all_index_entries = []
    for r in results:
        all_index_entries.extend(r["index_entries"])
    all_index_entries.sort(key=lambda e: (e["scene_name"], e["timestamp"]))

    index_path = os.path.join(split_out_folder, "index.jsonl")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    with open(index_path, "w") as f:
        for entry in all_index_entries:
            entry["split"] = split
            json.dump(entry, f)
            f.write("\n")
    print(f"Wrote {len(all_index_entries)} entries to {index_path}")

    print(f"[{split}] Done. Frames processed: {split_count}")
    return split_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract RGB images and sparse LiDAR depth maps from Waymo Open Dataset."
    )
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Waymo raw dataset root (contains training/, validation/, etc.)")
    parser.add_argument("--out_folder", type=str, required=True,
                        help="Output root folder (split dirs with rgb/ and depth/ will be created inside)")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--splits", type=str, nargs="+", default=ALL_SPLITS,
                        help="Splits to process (default: training validation testing testing_location)")
    parser.add_argument("--start_file", type=int, default=0,
                        help="Start index of parquet files to process (default: 0)")
    parser.add_argument("--end_file", type=int, default=-1,
                        help="End index of parquet files to process (default: -1 = all)")
    parser.add_argument("--intrinsics_only", action="store_true",
                        help="Only save camera intrinsics (skip RGB and depth extraction)")
    args = parser.parse_args()

    grand_total = 0
    for split in args.splits:
        grand_total += process_split(
            split=split,
            dataset_dir=args.dataset_dir,
            out_folder=args.out_folder,
            num_workers=args.num_workers,
            start_file=args.start_file,
            end_file=args.end_file,
            intrinsics_only=args.intrinsics_only,
        )

    print(f"\nAll done. Grand total frames processed: {grand_total}")
