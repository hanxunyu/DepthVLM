#!/usr/bin/env python
"""
Create pixel-level depth dataset metadata for Matterport3D.

Directory structure:
    matterport3d/
    ├── {scene_id}/
    │   ├── undistorted_color_images/     ← RGB (jpg, 1280×1024)
    │   ├── undistorted_depth_images/     ← Depth (png, uint16, /4000 = meters)
    │   └── undistorted_camera_parameters/
    │       └── {scene_id}.conf           ← Intrinsics + per-frame extrinsics
    ├── scenes_train.txt
    ├── scenes_val.txt
    └── scenes_test.txt

.conf format:
    intrinsics_matrix fx 0 cx  0 fy cy  0 0 1      ← Shared intrinsics for the scene
    scan depth_file rgb_file [4×4 extrinsics]       ← Per-frame entry

RGB: {pano_id}_i{cam}_{view}.jpg  →  Depth: {pano_id}_d{cam}_{view}.png
Original resolution is 1280×1024 for all frames.

Usage:
    python create_data_pixel_level_matterport3d.py \
        --data_root /path/to/matterport3d \
        --output_dir ./annotations/matterport3d \
        --num_workers 32
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from tqdm import tqdm

DEPTH_SCALE = 4000.0  # Matterport3D: uint16 / 4000 = meters
CANONICAL_FX = 1000.0
DEFAULT_W, DEFAULT_H = 1280, 1024


def load_split(split_path):
    with open(split_path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def parse_conf(conf_path):
    """Parse a Matterport3D .conf file.

    Returns:
        fx: Focal length.
        frames: List of (depth_fname, rgb_fname).
    """
    fx = None
    frames = []
    with open(conf_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("intrinsics_matrix"):
                # intrinsics_matrix fx 0 cx  0 fy cy  0 0 1
                parts = line.split()
                fx = float(parts[1])
            elif line.startswith("scan"):
                # scan depth_file rgb_file [16 floats for 4x4 extrinsics]
                parts = line.split()
                depth_fname = parts[1]
                rgb_fname = parts[2]
                frames.append((depth_fname, rgb_fname))
    return fx, frames


def process_scene(data_root, scene_id):
    """Process a single scene and return a list of records."""
    color_dir = os.path.join(data_root, scene_id, "undistorted_color_images")
    depth_dir = os.path.join(data_root, scene_id, "undistorted_depth_images")
    conf_path = os.path.join(data_root, scene_id, "undistorted_camera_parameters", f"{scene_id}.conf")

    if not os.path.isdir(color_dir) or not os.path.isdir(depth_dir):
        return []

    # Parse conf to get intrinsics and frame list
    if os.path.exists(conf_path):
        fx, conf_frames = parse_conf(conf_path)
    else:
        fx, conf_frames = None, []

    # If conf parsing succeeded, use the frame list from conf
    if conf_frames:
        records = []
        for depth_fname, rgb_fname in conf_frames:
            rgb_path = os.path.join(color_dir, rgb_fname)
            depth_path = os.path.join(depth_dir, depth_fname)
            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                continue

            rgb_w, rgb_h = DEFAULT_W, DEFAULT_H

            if fx and fx > 0:
                scale = CANONICAL_FX / fx
                canonical_w = round(rgb_w * scale)
                canonical_h = round(rgb_h * scale)
            else:
                canonical_w = canonical_h = None

            record = {
                "scene": scene_id,
                "image": f"matterport3d/{scene_id}/undistorted_color_images/{rgb_fname}",
                "depth_path": f"matterport3d/{scene_id}/undistorted_depth_images/{depth_fname}",
                "depth_scale": DEPTH_SCALE,
                "original_rgb_size": [rgb_w, rgb_h],
                "original_depth_size": [rgb_w, rgb_h],
                "original_fx": round(fx, 2) if fx else None,
                "canonical_fx": CANONICAL_FX,
                "canonical_size": [canonical_w, canonical_h] if canonical_w else None,
                "prompt": "Given this image, estimate the metric depth (in meters, rounded to two decimal places) for every pixel of the image.",
                "solution": "OK, I will estimate the depth map of this image.",
            }
            records.append(record)
        return records
    else:
        # Fallback: iterate over RGB directory
        rgb_files = sorted([
            f for f in os.listdir(color_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        records = []
        for fname in rgb_files:
            depth_fname = fname.replace("_i", "_d").replace(".jpg", ".png").replace(".jpeg", ".png")
            depth_path = os.path.join(depth_dir, depth_fname)
            if not os.path.exists(depth_path):
                continue

            rgb_w, rgb_h = DEFAULT_W, DEFAULT_H

            record = {
                "scene": scene_id,
                "image": f"matterport3d/{scene_id}/undistorted_color_images/{fname}",
                "depth_path": f"matterport3d/{scene_id}/undistorted_depth_images/{depth_fname}",
                "depth_scale": DEPTH_SCALE,
                "original_rgb_size": [rgb_w, rgb_h],
                "original_depth_size": [rgb_w, rgb_h],
                "original_fx": None,
                "canonical_fx": CANONICAL_FX,
                "canonical_size": None,
                "prompt": "Given this image, estimate the metric depth (in meters, rounded to two decimal places) for every pixel of the image.",
                "solution": "OK, I will estimate the depth map of this image.",
            }
            records.append(record)
        return records


def _process_scene_wrapper(args_tuple):
    return process_scene(*args_tuple)


def write_jsonl(records, path, label):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    scenes = sorted(set(r["scene"] for r in records)) if records else []
    print(f"  {label}: {len(records)} frames, {len(scenes)} scenes → {path}")
    if records:
        fxs = [r["original_fx"] for r in records if r.get("original_fx")]
        if fxs:
            print(f"    fx range: [{min(fxs):.1f}, {max(fxs):.1f}]")
        canonical_sizes = set(tuple(r["canonical_size"]) for r in records if r.get("canonical_size"))
        if canonical_sizes:
            sizes_list = sorted(canonical_sizes)
            print(f"    Canonical sizes ({len(sizes_list)}): {sizes_list[:5]}..." if len(sizes_list) > 5 else f"    Canonical sizes: {sizes_list}")
        no_fx = sum(1 for r in records if not r.get("original_fx"))
        if no_fx:
            print(f"    WARNING: {no_fx} frames without fx (no conf)")


def main():
    parser = argparse.ArgumentParser(description="Create pixel-level depth metadata for Matterport3D.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load split files
    splits = {}
    for split_name in ["train", "val", "test"]:
        split_file = os.path.join(args.data_root, f"scenes_{split_name}.txt")
        if os.path.exists(split_file):
            splits[split_name] = load_split(split_file)
            print(f"Loaded {split_name} split: {len(splits[split_name])} scenes")

    # Get all scene directories
    all_scene_ids = sorted([
        d for d in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, d))
        and os.path.isdir(os.path.join(args.data_root, d, "undistorted_color_images"))
    ])
    print(f"Found {len(all_scene_ids)} scenes with data")
    print(f"Using {args.num_workers} workers")

    # Process all scenes in parallel
    tasks = [(args.data_root, sid) for sid in all_scene_ids]
    all_results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        all_results = list(tqdm(
            executor.map(_process_scene_wrapper, tasks, chunksize=1),
            total=len(tasks),
            desc="Processing scenes"
        ))

    # Group by split
    split_records = {k: [] for k in splits}
    for scene_id, records in zip(all_scene_ids, all_results):
        if not records:
            continue
        for split_name, scene_set in splits.items():
            if scene_id in scene_set:
                split_records[split_name].extend(records)
                break

    print(f"\n{'='*60}")
    for split_name, records in split_records.items():
        scenes = set(r["scene"] for r in records) if records else set()
        print(f"{split_name}: {len(records)} frames, {len(scenes)} scenes")

    for split_name, records in split_records.items():
        if records:
            output_path = os.path.join(args.output_dir, f"mp3d_pixel_depth_{split_name}.jsonl")
            write_jsonl(records, output_path, split_name.capitalize())


if __name__ == "__main__":
    main()
