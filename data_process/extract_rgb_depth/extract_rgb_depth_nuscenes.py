# Extract RGB images, sparse LiDAR depth maps, and camera intrinsics from nuScenes dataset.
# Reference: utils/curate_nuscenes_train.py
# Usage: python extract_rgb_depth_nuscenes.py --dataroot <...> --out_folder <...> [--num_workers 8]
#
# Output structure (scene-based, consistent with Waymo/Argoverse):
#   out_folder/{train,val,test}/rgb/{scene_name}/{timestamp}_{cam_name}.jpg
#   out_folder/{train,val,test}/depth/{scene_name}/{timestamp}_{cam_name}.png  (uint16, depth_m × 256)
#   out_folder/{train,val,test}/intrinsics/{scene_name}/{timestamp}_{cam_name}.json  ([fx, fy, cx, cy, W, H])
#   out_folder/{train,val,test}/index.jsonl  (per-frame metadata for traceability)
"""
Example usage:

python extract_rgb_depth_nuscenes.py \
    --dataroot /path/to/nuscenes_raw \
    --out_folder /path/to/nuscenes \
    --num_workers 32 \
    --splits train val test

python extract_rgb_depth_nuscenes.py \
    --dataroot /path/to/nuscenes_raw \
    --out_folder /path/to/nuscenes \
    --num_workers 16 \
    --intrinsics_only
"""

import argparse
import json
import logging
import os
from functools import partial
from multiprocessing import Pool
from typing import Final, List, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion

logger = logging.getLogger(__name__)

# Depth save scale: depth_m × 256 → uint16 PNG
# Max representable depth: 65535 / 256 ≈ 255.9 m
# Precision: 1/256 ≈ 3.9 mm
# Read back: depth_m = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
DEPTH_SAVE_SCALE: Final[float] = 256.0

CAMERA_NAMES: Final[List[str]] = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]


def map_pointcloud_to_image(
    nusc: NuScenes,
    lidar_data: dict,
    camera_token: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map LiDAR pointcloud to the image plane.

    Returns:
        points_img: (N, 2) projected image coordinates
        depths: (N,) depth values in meters
        image: (H, W, 3) BGR image
    """
    cam = nusc.get("sample_data", camera_token)
    cam_path = os.path.join(nusc.dataroot, cam["filename"])
    im = cv2.imread(cam_path)

    # LiDAR calibration
    lidar_to_world = nusc.get(
        "calibrated_sensor", lidar_data["calibrated_sensor_token"]
    )
    lidar_rotation = Quaternion(lidar_to_world["rotation"])
    lidar_translation = np.array(lidar_to_world["translation"])

    # Camera calibration
    cam_to_world = nusc.get("calibrated_sensor", cam["calibrated_sensor_token"])
    cam_intrinsic = np.array(cam_to_world["camera_intrinsic"])
    cam_rotation = Quaternion(cam_to_world["rotation"])
    cam_translation = np.array(cam_to_world["translation"])

    # Read LiDAR points
    pc = LidarPointCloud.from_file(
        os.path.join(nusc.dataroot, lidar_data["filename"])
    )
    points = pc.points[:3, :]
    points = np.vstack((points, np.ones(points.shape[1])))

    # LiDAR → world
    lidar_to_world_matrix = np.eye(4)
    lidar_to_world_matrix[:3, :3] = lidar_rotation.rotation_matrix
    lidar_to_world_matrix[:3, 3] = lidar_translation

    # World → camera
    world_to_cam_matrix = np.eye(4)
    world_to_cam_matrix[:3, :3] = cam_rotation.rotation_matrix.T
    world_to_cam_matrix[:3, 3] = -np.dot(
        cam_rotation.rotation_matrix.T, cam_translation
    )

    # Transform to camera coordinates
    points_cam = np.dot(world_to_cam_matrix, np.dot(lidar_to_world_matrix, points))

    # Keep only points in front of the camera
    mask = points_cam[2, :] > 0
    points_cam = points_cam[:, mask]

    # Project to image plane
    points_img = np.dot(cam_intrinsic, points_cam[:3, :])
    points_img = points_img / points_img[2, :]
    points_img = points_img[:2, :]

    depths = points_cam[2, :].copy()

    return points_img.T, depths, im


def create_depth_map(
    points_img: np.ndarray,
    depths: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Create a sparse depth map from projected LiDAR points.

    Args:
        points_img: (N, 2) projected image coordinates (x, y)
        depths: (N,) depth values
        image_shape: (height, width)

    Returns:
        depth_map: (H, W) float32 depth map
    """
    h, w = image_shape
    depth_map = np.zeros((h, w), dtype=np.float32)

    # Filter points within image bounds
    mask = (
        (points_img[:, 0] >= 0)
        & (points_img[:, 0] < w)
        & (points_img[:, 1] >= 0)
        & (points_img[:, 1] < h)
    )
    points_img = points_img[mask]
    depths = depths[mask]

    # Convert to integer pixel coordinates
    xs = np.floor(points_img[:, 0]).astype(np.int32)
    ys = np.floor(points_img[:, 1]).astype(np.int32)

    # Populate depth map (keep closest depth when multiple points map to same pixel)
    for i in range(len(xs)):
        x, y = xs[i], ys[i]
        if depth_map[y, x] == 0 or depths[i] < depth_map[y, x]:
            depth_map[y, x] = depths[i]

    return depth_map


# ---- Per-worker global: each worker initialises its own NuScenes instance ----
_worker_nusc = None
_worker_dataroot = None
_worker_version = None


def _worker_init(dataroot: str, version: str):
    """Initializer for each pool worker: create a private NuScenes instance."""
    global _worker_nusc, _worker_dataroot, _worker_version
    _worker_dataroot = dataroot
    _worker_version = version
    _worker_nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)


def process_one_sample(
    task: Tuple[str, str, str, int],
    out_folder: str,
    intrinsics_only: bool = False,
) -> Tuple[int, dict]:
    """Process a single nuScenes sample for all cameras.

    Args:
        task: (sample_token, scene_name, split, frame_idx_in_scene)
        out_folder: output root folder

    Returns:
        (count, index_entry) where index_entry is metadata dict for index.jsonl.
    """
    nusc = _worker_nusc
    sample_token, scene_name, split, frame_idx = task
    sample = nusc.get("sample", sample_token)
    count = 0

    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    timestamp = sample["timestamp"]

    index_entry = {
        "scene_name": scene_name,
        "frame_idx": frame_idx,
        "sample_token": sample_token,
        "timestamp": timestamp,
        "split": split,
        "cameras": [],
        "source_dataset": "nuscenes",
    }

    for cam_name in CAMERA_NAMES:
        try:
            camera_token = sample["data"][cam_name]
            cam_data = nusc.get("sample_data", camera_token)
            cam_calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
            cam_intrinsic = np.array(cam_calib["camera_intrinsic"])

            # Construct relative filename: {scene_name}/{timestamp}_{cam_name}
            relative_name = f"{scene_name}/{timestamp}_{cam_name}.jpg"
            relative_stem = f"{scene_name}/{timestamp}_{cam_name}"

            # --- Save intrinsics ---
            img_w = cam_data["width"]
            img_h = cam_data["height"]
            intrinsics_data = [
                float(cam_intrinsic[0, 0]),  # fx
                float(cam_intrinsic[1, 1]),  # fy
                float(cam_intrinsic[0, 2]),  # cx
                float(cam_intrinsic[1, 2]),  # cy
                img_w,                        # width
                img_h,                        # height
            ]
            intrinsics_out_path = os.path.join(
                out_folder, split, "intrinsics", relative_stem + ".json"
            )
            os.makedirs(os.path.dirname(intrinsics_out_path), exist_ok=True)
            with open(intrinsics_out_path, "w") as f:
                json.dump(intrinsics_data, f)

            if intrinsics_only:
                count += 1
                index_entry["cameras"].append(cam_name)
                continue

            # Project LiDAR to image
            points_img, depths, image = map_pointcloud_to_image(
                nusc, lidar_data, camera_token
            )
            h, w = image.shape[:2]
            depth_map = create_depth_map(points_img, depths, (h, w))

            # Save RGB image
            rgb_out_path = os.path.join(out_folder, split, "rgb", relative_name)
            os.makedirs(os.path.dirname(rgb_out_path), exist_ok=True)
            cv2.imwrite(rgb_out_path, image)

            # Save depth map as uint16 PNG (depth_m × DEPTH_SAVE_SCALE)
            depth_out_path = os.path.join(
                out_folder, split, "depth", relative_stem + ".png"
            )
            os.makedirs(os.path.dirname(depth_out_path), exist_ok=True)
            depth_uint16 = (depth_map * DEPTH_SAVE_SCALE).clip(0, 65535).astype(
                np.uint16
            )
            Image.fromarray(depth_uint16).save(depth_out_path)

            count += 1
            index_entry["cameras"].append(cam_name)

        except Exception as e:
            print(f"Error processing sample {sample_token} camera {cam_name}: {e}")
            continue

    return count, index_entry


def build_scene_tasks(nusc, scene_token_to_split):
    """Build per-sample tasks grouped by scene, preserving temporal order.

    Returns:
        tasks: list of (sample_token, scene_name, split, frame_idx_in_scene)
    """
    tasks = []
    for scene in nusc.scene:
        scene_name = scene["name"]
        scene_token = scene["token"]
        split = scene_token_to_split.get(scene_token)
        if split is None:
            continue

        # Walk through all samples in this scene in temporal order
        sample_token = scene["first_sample_token"]
        frame_idx = 0
        while sample_token:
            tasks.append((sample_token, scene_name, split, frame_idx))
            sample = nusc.get("sample", sample_token)
            sample_token = sample["next"] if sample["next"] else None
            frame_idx += 1

    return tasks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract RGB images and sparse LiDAR depth maps from nuScenes."
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        required=True,
        help="nuScenes dataset root folder (must contain v1.0-trainval/ and samples/)",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        required=True,
        help="Output root folder ({split}/rgb/ and {split}/depth/ will be created inside)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-trainval",
        help="nuScenes version (default: v1.0-trainval)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel worker processes (default: 8)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val"],
        help="Splits to process (default: train val). Options: train, val, test",
    )
    parser.add_argument(
        "--intrinsics_only",
        action="store_true",
        help="Only save camera intrinsics (skip RGB and depth extraction)",
    )
    args = parser.parse_args()

    dataroot = args.dataroot
    out_folder = args.out_folder
    version = args.version
    num_workers = args.num_workers
    requested_splits = set(args.splits)

    # Validate split names
    valid_splits = {"train", "val", "test"}
    invalid = requested_splits - valid_splits
    if invalid:
        parser.error(f"Invalid split(s): {invalid}. Must be subset of {valid_splits}")

    print(f"Requested splits: {sorted(requested_splits)}")

    grand_total = 0
    worker_fn = partial(process_one_sample, out_folder=out_folder, intrinsics_only=args.intrinsics_only)

    # ---- Process train / val (from v1.0-trainval) ----
    trainval_requested = requested_splits & {"train", "val"}
    if trainval_requested:
        print(f"\nLoading nuScenes {version} from {dataroot} ...")
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

        split_names = create_splits_scenes()
        train_scene_names = set(split_names["train"])
        val_scene_names = set(split_names["val"])

        # Build scene_token → split mapping (only for requested splits)
        scene_token_to_split = {}
        for scene in nusc.scene:
            name = scene["name"]
            token = scene["token"]
            if "train" in trainval_requested and name in train_scene_names:
                scene_token_to_split[token] = "train"
            elif "val" in trainval_requested and name in val_scene_names:
                scene_token_to_split[token] = "val"

        all_tasks = build_scene_tasks(nusc, scene_token_to_split)
        for s in sorted(trainval_requested):
            cnt = sum(1 for t in all_tasks if t[2] == s)
            print(f"  {s}: {cnt} samples")

        with Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(dataroot, version),
        ) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(worker_fn, all_tasks),
                    total=len(all_tasks),
                    desc=f"[{','.join(sorted(trainval_requested))}] samples",
                )
            )

        trainval_frame_count = sum(r[0] for r in results)
        trainval_index_entries = [r[1] for r in results]
        grand_total += trainval_frame_count
        print(f"Train/Val done. Frames processed: {trainval_frame_count}")

        # Write index.jsonl per split
        for split_name in sorted(trainval_requested):
            index_path = os.path.join(out_folder, split_name, "index.jsonl")
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            entries = sorted(
                [e for e in trainval_index_entries if e["split"] == split_name],
                key=lambda e: (e["scene_name"], e["frame_idx"]),
            )
            with open(index_path, "w") as f:
                for entry in entries:
                    json.dump(entry, f)
                    f.write("\n")
            print(f"Wrote {len(entries)} entries to {index_path}")

    # ---- Process test split (from v1.0-test) ----
    if "test" in requested_splits:
        test_version = "v1.0-test"
        print(f"\nLoading nuScenes {test_version} from {dataroot} ...")
        nusc_test = NuScenes(
            version=test_version, dataroot=dataroot, verbose=True
        )

        split_names = create_splits_scenes()
        test_scene_names = set(split_names["test"])

        test_scene_token_to_split = {}
        for scene in nusc_test.scene:
            if scene["name"] in test_scene_names:
                test_scene_token_to_split[scene["token"]] = "test"

        test_tasks = build_scene_tasks(nusc_test, test_scene_token_to_split)
        print(f"  test: {len(test_tasks)} samples")

        with Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(dataroot, test_version),
        ) as pool:
            test_results = list(
                tqdm(
                    pool.imap_unordered(worker_fn, test_tasks),
                    total=len(test_tasks),
                    desc="[test] samples",
                )
            )

        test_count = sum(r[0] for r in test_results)
        test_index_entries = [r[1] for r in test_results]
        grand_total += test_count

        index_path = os.path.join(out_folder, "test", "index.jsonl")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        entries = sorted(
            test_index_entries,
            key=lambda e: (e["scene_name"], e["frame_idx"]),
        )
        with open(index_path, "w") as f:
            for entry in entries:
                json.dump(entry, f)
                f.write("\n")
        print(f"Wrote {len(entries)} entries to {index_path}")
        print(f"Test done. Frames processed: {test_count}")

    print(f"\nAll done. Total frames processed: {grand_total}")
