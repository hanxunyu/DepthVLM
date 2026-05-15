#!/usr/bin/env python
"""
Create pixel-level depth dataset metadata for ETH3D (original high-res).

Directory structure:
    eth3d/high-res_train/
    ├── multi_view_training_dslr_undistorted/
    │   └── {scene}/
    │       ├── dslr_calibration_undistorted/
    │       │   ├── cameras.txt   → COLMAP format: CAMERA_ID PINHOLE W H fx fy cx cy
    │       │   └── images.txt    → IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    │       └── images/dslr_images_undistorted/
    │           └── DSC_XXXX.JPG  (original ~6200×4135)
    └── depthmap/
        └── {scene}/ground_truth_depth/dslr_images/
            └── DSC_XXXX.JPG     (binary float32, 6048×4032, unit: meters, inf=invalid)

RGB and depth have different resolutions (RGB ~6205×4135, depth = 6048×4032).

Usage:
    python create_data_pixel_level_eth3d.py \
        --data_root /path/to/eth3d/high-res_train \
        --output_dir ./annotations/eth3d
"""
import argparse
import json
import os
import re

import numpy as np
from tqdm import tqdm

# Depth is binary float32, unit is meters directly, depth_scale = 1.0
DEPTH_SCALE = 1.0
# Depth map fixed resolution
DEPTH_W, DEPTH_H = 6048, 4032
# Target canonical focal length
CANONICAL_FX = 1000.0

# ETH3D scene classification
INDOOR_SCENES = {"delivery_area", "kicker", "office", "pipes", "relief", "relief_2", "terrains"}
OUTDOOR_SCENES = {"courtyard", "electro", "facade", "meadow", "playground", "terrace"}


def parse_cameras_txt(path):
    """Parse COLMAP cameras.txt → {camera_id: (W, H, fx, fy, cx, cy)}."""
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
            cam_id = int(parts[0])
            w, h = int(parts[2]), int(parts[3])
            fx, fy, cx, cy = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            cameras[cam_id] = (w, h, fx, fy, cx, cy)
    return cameras


def parse_images_txt(path):
    """Parse COLMAP images.txt → {image_name: camera_id}.

    images.txt has pairs of lines:
    Line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    Line 2: POINTS2D[] (skipped)
    """
    image_to_cam = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    # Process pairs of lines
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        cam_id = int(parts[8])
        name = parts[9]  # e.g., dslr_images_undistorted/DSC_0286.JPG
        # Extract filename only
        fname = os.path.basename(name)
        image_to_cam[fname] = cam_id
    return image_to_cam


def main():
    parser = argparse.ArgumentParser(description="Create pixel-level depth metadata for ETH3D (original high-res).")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root dir: eth3d/high-res_train/")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    undist_base = os.path.join(args.data_root, "multi_view_training_dslr_undistorted")
    depth_base = os.path.join(args.data_root, "depthmap")

    scenes = sorted([
        d for d in os.listdir(undist_base)
        if os.path.isdir(os.path.join(undist_base, d))
    ])
    print(f"Found {len(scenes)} scenes: {scenes}")

    all_records = []
    for scene in scenes:
        # Parse intrinsics
        calib_dir = os.path.join(undist_base, scene, "dslr_calibration_undistorted")
        cameras_path = os.path.join(calib_dir, "cameras.txt")
        images_path = os.path.join(calib_dir, "images.txt")

        if not os.path.exists(cameras_path) or not os.path.exists(images_path):
            print(f"WARNING: calibration not found for {scene}, skipping")
            continue

        cameras = parse_cameras_txt(cameras_path)
        image_to_cam = parse_images_txt(images_path)

        # RGB directory
        rgb_dir = os.path.join(undist_base, scene, "images", "dslr_images_undistorted")
        # Depth directory
        depth_dir = os.path.join(depth_base, scene, "ground_truth_depth", "dslr_images")

        if not os.path.isdir(rgb_dir):
            print(f"WARNING: rgb dir not found for {scene}, skipping")
            continue
        if not os.path.isdir(depth_dir):
            print(f"WARNING: depth dir not found for {scene}, skipping")
            continue

        rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.upper().endswith(".JPG")])
        depth_files = set(os.listdir(depth_dir))

        scene_type = "indoor" if scene in INDOOR_SCENES else "outdoor"

        for fname in tqdm(rgb_files, desc=scene):
            if fname not in depth_files:
                continue

            stem = os.path.splitext(fname)[0]

            # Get camera_id for this image → intrinsics
            cam_id = image_to_cam.get(fname)
            if cam_id is None:
                continue
            cam_info = cameras.get(cam_id)
            if cam_info is None:
                continue

            rgb_w, rgb_h, fx, fy, cx, cy = cam_info

            # Compute canonical size
            scale = CANONICAL_FX / fx
            canonical_w = round(rgb_w * scale)
            canonical_h = round(rgb_h * scale)

            record = {
                "scene": scene,
                "scene_type": scene_type,
                "image": f"eth3d/high-res_train/multi_view_training_dslr_undistorted/{scene}/images/dslr_images_undistorted/{fname}",
                "depth_path": f"eth3d/high-res_train/depthmap/{scene}/ground_truth_depth/dslr_images/{fname}",
                "depth_format": "binary_float32",
                "depth_scale": DEPTH_SCALE,
                "original_rgb_size": [rgb_w, rgb_h],
                "original_depth_size": [DEPTH_W, DEPTH_H],
                "original_fx": round(fx, 2),
                "canonical_fx": CANONICAL_FX,
                "canonical_size": [canonical_w, canonical_h],
                "prompt": "Given this image, estimate the metric depth (in meters, rounded to two decimal places) for every pixel of the image.",
                "solution": "OK, I will estimate the depth map of this image.",
            }
            all_records.append(record)

    # Group by scene_type
    indoor_records = [r for r in all_records if r["scene_type"] == "indoor"]
    outdoor_records = [r for r in all_records if r["scene_type"] == "outdoor"]

    print(f"\n{'='*60}")
    print(f"Total: {len(all_records)} (indoor={len(indoor_records)}, outdoor={len(outdoor_records)})")

    for records, name, label in [
        (all_records, "eth3d_pixel_depth_all.jsonl", "All"),
        (indoor_records, "eth3d_pixel_depth_indoor.jsonl", "Indoor"),
        (outdoor_records, "eth3d_pixel_depth_outdoor.jsonl", "Outdoor"),
    ]:
        path = os.path.join(args.output_dir, name)
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        scenes_list = sorted(set(r["scene"] for r in records)) if records else []
        print(f"\n  {label}: {len(records)} frames, {len(scenes_list)} scenes → {path}")
        if records:
            print(f"    Scenes: {scenes_list}")
            sizes = set(tuple(r["original_rgb_size"]) for r in records)
            print(f"    RGB resolutions: {sizes}")
            print(f"    Depth size: {DEPTH_W}x{DEPTH_H}")
            canonical_sizes = set(tuple(r["canonical_size"]) for r in records)
            print(f"    Canonical sizes: {canonical_sizes}")
            fxs = [r["original_fx"] for r in records]
            print(f"    fx range: [{min(fxs):.1f}, {max(fxs):.1f}]")


if __name__ == "__main__":
    main()
