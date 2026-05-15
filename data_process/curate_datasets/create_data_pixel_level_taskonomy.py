#!/usr/bin/env python
"""
Create pixel-level depth dataset metadata for Taskonomy (Omnidata rendered version).
Reads field_of_view_rads from point_info JSON to compute fx, and adds canonical_size.
Uses RGB as the reference (13/530 buildings have fewer RGB than depth files).

Directory structure:
    omnidata_taskonomy/
    ├── rgb/taskonomy/{scene}/{point_X_view_Y}_domain_rgb.png              (512×512)
    ├── depth_zbuffer/taskonomy/{scene}/{point_X_view_Y}_domain_depth_zbuffer.png  (512×512, uint16)
    └── point_info/taskonomy/{scene}/{point_X_view_Y}_domain_point_info.json

Usage:
    python create_data_pixel_level_taskonomy.py \
        --data_root /path/to/omnidata_taskonomy \
        --metadata /path/to/taskonomy/splits/train_val_test_fullplus.csv \
        --output_dir ./annotations/taskonomy \
        --num_workers 32

PS:
    The metadata can be downloaded from https://github.com/StanfordVL/taskonomy/raw/master/data/assets/splits_taskonomy.zip
"""
import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

DEPTH_SCALE = 512.0  # Omnidata: uint16 / 512 = meters
CANONICAL_FX = 1000.0


def load_metadata(csv_path):
    scene_to_split = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = row["id"].strip()
            if row["train"].strip() == "1":
                scene_to_split[s] = "train"
            elif row["val"].strip() == "1":
                scene_to_split[s] = "val"
            elif row["test"].strip() == "1":
                scene_to_split[s] = "test"
    return scene_to_split


def process_single_frame(args_tuple):
    """Process a single frame and return a record or None. Uses RGB as reference."""
    data_root, scene_name, fname = args_tuple
    try:
        # Check if depth file exists
        depth_fname = fname.replace("_domain_rgb.png", "_domain_depth_zbuffer.png")
        depth_path = os.path.join(data_root, "depth_zbuffer", "taskonomy", scene_name, depth_fname)
        if not os.path.exists(depth_path):
            return None

        # Read point_info to get FOV and compute fx
        point_info_fname = fname.replace("_domain_rgb.png", "_domain_point_info.json")
        point_info_path = os.path.join(data_root, "point_info", "taskonomy", scene_name, point_info_fname)

        fx = None
        resolution = 512
        if os.path.exists(point_info_path):
            with open(point_info_path, "r") as f:
                info = json.loads(f.read())
            fov_rads = info.get("field_of_view_rads")
            resolution = info.get("resolution", 512)
            if fov_rads and fov_rads > 0:
                fx = resolution / (2 * math.tan(fov_rads / 2))

        rgb_w = rgb_h = resolution

        # Compute canonical size
        if fx and fx > 0:
            scale = CANONICAL_FX / fx
            canonical_w = round(rgb_w * scale)
            canonical_h = round(rgb_h * scale)
        else:
            canonical_w = canonical_h = None

        record = {
            "scene": scene_name,
            "image": f"omnidata_taskonomy/rgb/taskonomy/{scene_name}/{fname}",
            "depth_path": f"omnidata_taskonomy/depth_zbuffer/taskonomy/{scene_name}/{depth_fname}",
            "depth_scale": DEPTH_SCALE,
            "original_rgb_size": [rgb_w, rgb_h],
            "original_depth_size": [rgb_w, rgb_h],
            "original_fx": round(fx, 2) if fx else None,
            "canonical_fx": CANONICAL_FX,
            "canonical_size": [canonical_w, canonical_h] if canonical_w else None,
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
    parser = argparse.ArgumentParser(description="Create pixel-level depth metadata for Taskonomy (Omnidata).")
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
    print(f"Metadata splits: {split_counts} ({len(scene_to_split)} total in csv)")

    # Local scenes (using RGB as reference)
    rgb_base = os.path.join(args.data_root, "rgb", "taskonomy")
    if not os.path.isdir(rgb_base):
        print(f"ERROR: {rgb_base} not found")
        return
    local_scenes = sorted([
        d for d in os.listdir(rgb_base)
        if os.path.isdir(os.path.join(rgb_base, d))
    ])
    print(f"Found {len(local_scenes)} scenes locally")

    # Collect tasks (using RGB as reference)
    tasks = []
    scene_split_map = {}
    for scene_name in local_scenes:
        split = scene_to_split.get(scene_name, None)
        if split is None or split not in ("train", "val", "test"):
            continue
        scene_split_map[scene_name] = split
        rgb_dir = os.path.join(rgb_base, scene_name)
        rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith(".png") and "_domain_rgb" in f]
        for fname in rgb_files:
            tasks.append((args.data_root, scene_name, fname))

    print(f"Total tasks: {len(tasks)} frames from {len(scene_split_map)} scenes")
    print(f"Using {args.num_workers} workers")

    # Multi-process execution
    split_records = {"train": [], "val": [], "test": []}

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_frame, tasks, chunksize=100),
            total=len(tasks),
            desc="Processing"
        ))

    for task, record in zip(tasks, results):
        if record is not None:
            scene_name = task[1]
            split = scene_split_map[scene_name]
            split_records[split].append(record)

    print(f"\n{'='*60}")
    for split_name in ["train", "val", "test"]:
        records = split_records[split_name]
        scenes = set(r["scene"] for r in records) if records else set()
        csv_n = split_counts.get(split_name, 0)
        print(f"{split_name}: {len(records)} frames, {len(scenes)}/{csv_n} scenes (local/csv)")

    for split_name, records in split_records.items():
        if records:
            output_path = os.path.join(args.output_dir, f"taskonomy_pixel_depth_{split_name}.jsonl")
            write_jsonl(records, output_path, split_name.capitalize())


if __name__ == "__main__":
    main()
