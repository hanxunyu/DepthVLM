#!/usr/bin/env python
"""
Create pixel-level depth dataset metadata for HM3D (Omnidata rendered version).
Reads field_of_view_rads from point_info JSON to compute fx, and adds canonical_size.

Directory structure:
    omnidata_hm3d/
    ├── rgb/hm3d/{scene_id}/{point_X_view_Y}_domain_rgb.png              (512×512)
    ├── depth_zbuffer/hm3d/{scene_id}/{point_X_view_Y}_domain_depth_zbuffer.png  (512×512, uint16)
    └── point_info/hm3d/{scene_id}/{point_X_view_Y}_domain_point_info.json

Usage:
    python create_data_pixel_level_hm3d.py \
        --data_root /path/to/omnidata_hm3d \
        --metadata /path/to/hm3d/splits/metadata.csv \
        --output_dir ./annotations/hm3d \
        --num_workers 32

PS:
    metadata can be found in https://github.com/matterport/habitat-matterport-3dresearch/blob/main/metadata.csv
"""
import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

DEPTH_SCALE = 512.0  # Omnidata HM3D: uint16 / 512 = meters
CANONICAL_FX = 1000.0


def load_metadata(csv_path):
    scene_to_split = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_to_split[row["scene"]] = row["split"]
    return scene_to_split


def process_single_frame(args_tuple):
    """Process a single frame and return a record or None."""
    data_root, scene_id, fname = args_tuple
    try:
        rgb_dir = os.path.join(data_root, "rgb", "hm3d", scene_id)
        depth_dir = os.path.join(data_root, "depth_zbuffer", "hm3d", scene_id)
        point_info_dir = os.path.join(data_root, "point_info", "hm3d", scene_id)

        rgb_path = os.path.join(rgb_dir, fname)
        depth_fname = fname.replace("_domain_rgb.png", "_domain_depth_zbuffer.png")
        depth_path = os.path.join(depth_dir, depth_fname)

        if not os.path.exists(depth_path):
            return None

        # Read point_info to get FOV and compute fx
        point_info_fname = fname.replace("_domain_rgb.png", "_domain_fixatedpose.json")
        point_info_path = os.path.join(point_info_dir, point_info_fname)

        if not os.path.exists(point_info_path):
            return None

        with open(point_info_path, "r") as f:
            info = json.loads(f.read())
        fov_rads = info.get("field_of_view_rads")
        resolution = info.get("resolution", 512)
        if not fov_rads or fov_rads <= 0:
            return None

        fx = resolution / (2 * math.tan(fov_rads / 2))
        rgb_w = rgb_h = resolution

        # Compute canonical size
        scale = CANONICAL_FX / fx
        canonical_w = round(rgb_w * scale)
        canonical_h = round(rgb_h * scale)

        record = {
            "scene": scene_id,
            "image": f"omnidata_hm3d/rgb/hm3d/{scene_id}/{fname}",
            "depth_path": f"omnidata_hm3d/depth_zbuffer/hm3d/{scene_id}/{depth_fname}",
            "depth_scale": DEPTH_SCALE,
            "original_rgb_size": [rgb_w, rgb_h],
            "original_depth_size": [rgb_w, rgb_h],
            "original_fx": round(fx, 2),
            "canonical_fx": CANONICAL_FX,
            "canonical_size": [canonical_w, canonical_h],
            "prompt": "Given this image, estimate the metric depth (in meters, rounded to two decimal places) for every pixel of the image.",
            "solution": "OK, I will estimate the depth map of this image.",
        }
        return record
    except Exception:
        return None


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
            print(f"    WARNING: {no_fx} frames without fx (no point_info)")


def main():
    parser = argparse.ArgumentParser(description="Create pixel-level depth metadata for HM3D (Omnidata).")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load split metadata
    scene_to_split = load_metadata(args.metadata)
    split_counts = {}
    for s in scene_to_split.values():
        split_counts[s] = split_counts.get(s, 0) + 1
    print(f"Metadata: {split_counts}")

    # Get all scenes
    rgb_base = os.path.join(args.data_root, "rgb", "hm3d")
    if not os.path.isdir(rgb_base):
        print(f"ERROR: {rgb_base} not found")
        return
    all_scene_ids = sorted([
        d for d in os.listdir(rgb_base)
        if os.path.isdir(os.path.join(rgb_base, d))
    ])
    print(f"Found {len(all_scene_ids)} scenes")

    # Collect tasks
    tasks = []
    scene_split_map = {}
    for scene_id in all_scene_ids:
        split = scene_to_split.get(scene_id, None)
        if split is None or split not in ("train", "val", "test"):
            continue
        scene_split_map[scene_id] = split
        rgb_dir = os.path.join(rgb_base, scene_id)
        rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith(".png") and "domain_rgb" in f]
        for fname in rgb_files:
            tasks.append((args.data_root, scene_id, fname))

    print(f"Total tasks: {len(tasks)} frames from {len(scene_split_map)} scenes")
    print(f"Using {args.num_workers} workers")

    # Multi-process execution
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_frame, tasks, chunksize=100),
            total=len(tasks),
            desc="Processing"
        ))

    split_records = {"train": [], "val": [], "test": []}
    for task, record in zip(tasks, results):
        if record is not None:
            scene_id = task[1]
            split = scene_split_map[scene_id]
            split_records[split].append(record)

    print(f"\n{'='*60}")
    for split_name, records in split_records.items():
        scenes = set(r["scene"] for r in records) if records else set()
        csv_n = split_counts.get(split_name, 0)
        print(f"{split_name}: {len(records)} frames, {len(scenes)}/{csv_n} scenes")

    for split_name, records in split_records.items():
        if records:
            output_path = os.path.join(args.output_dir, f"hm3d_pixel_depth_{split_name}.jsonl")
            write_jsonl(records, output_path, split_name.capitalize())


if __name__ == "__main__":
    main()
