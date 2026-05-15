#!/usr/bin/env python
"""
Create pixel-level depth dataset metadata for DDAD v2.

Directory structure:
    ddad_v2/
    ├── train/
    │   ├── rgb/{scene_id}/{timestamp}_{camera}.jpg
    │   ├── depth/{scene_id}/{timestamp}_{camera}.png
    │   ├── intrinsics/{scene_id}/{timestamp}_{camera}.json  → [fx, fy, cx, cy, W, H]
    │   └── index.jsonl
    └── val/ (same as above)

Based on depth files to build JSONL, reading fx and resolution from intrinsics.

Usage:
    python create_data_pixel_level_ddad.py \
        --data_root /path/to/ddad_v2 \
        --splits train,val \
        --output_dir ./annotations/ddad \
        --num_workers 32
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

# DDAD depth: uint16 PNG, depth_value / DEPTH_SCALE = meters
DEPTH_SCALE = 256.0
# Target canonical focal length
CANONICAL_FX = 1000.0


def process_single_frame(args_tuple):
    """Process a single frame and return a record or None."""
    data_root, split, scene_id, depth_fname = args_tuple
    try:
        stem = os.path.splitext(depth_fname)[0]

        # Check if RGB file exists
        rgb_fname = stem + ".jpg"
        rgb_path = os.path.join(data_root, split, "rgb", scene_id, rgb_fname)
        if not os.path.exists(rgb_path):
            return None

        # Read intrinsics: [fx, fy, cx, cy, W, H]
        intrinsics_path = os.path.join(data_root, split, "intrinsics", scene_id, stem + ".json")
        if not os.path.exists(intrinsics_path):
            return None
        with open(intrinsics_path, "r") as f:
            intr = json.loads(f.read())
        fx = intr[0]
        rgb_w, rgb_h = int(intr[4]), int(intr[5])

        # Extract camera name: {timestamp}_{camera}
        parts = stem.split("_", 1)
        camera = parts[1] if len(parts) > 1 else "unknown"

        # Compute canonical size after rescaling to CANONICAL_FX
        scale = CANONICAL_FX / fx
        canonical_w = round(rgb_w * scale)
        canonical_h = round(rgb_h * scale)

        return {
            "scene": scene_id,
            "camera": camera,
            "split": split,
            "image": f"ddad_v2/{split}/rgb/{scene_id}/{rgb_fname}",
            "depth_path": f"ddad_v2/{split}/depth/{scene_id}/{depth_fname}",
            "depth_scale": DEPTH_SCALE,
            "original_rgb_size": [rgb_w, rgb_h],
            "original_depth_size": [rgb_w, rgb_h],
            "original_fx": round(fx, 2),
            "canonical_fx": CANONICAL_FX,
            "canonical_size": [canonical_w, canonical_h],
            "prompt": "Given this image, estimate the metric depth (in meters, rounded to two decimal places) for every pixel of the image.",
            "solution": "OK, I will estimate the depth map of this image.",
        }
    except Exception:
        return None


def write_jsonl(records, path, label):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    scenes = sorted(set(r["scene"] for r in records)) if records else []
    cameras = sorted(set(r["camera"] for r in records)) if records else []
    print(f"  {label}: {len(records)} frames, {len(scenes)} scenes → {path}")
    if records:
        print(f"    Cameras: {cameras}")
        sizes = set(tuple(r["original_rgb_size"]) for r in records)
        print(f"    Resolutions: {sizes}")
        canonical_sizes = set(tuple(r["canonical_size"]) for r in records)
        print(f"    Canonical sizes: {canonical_sizes}")
        fxs = [r["original_fx"] for r in records if "original_fx" in r]
        if fxs:
            print(f"    fx range: [{min(fxs):.1f}, {max(fxs):.1f}]")
        print(f"    depth_scale: {DEPTH_SCALE}")


def main():
    parser = argparse.ArgumentParser(description="Create pixel-level depth metadata for DDAD v2.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root dir of ddad_v2 (contains train/, val/)")
    parser.add_argument("--splits", type=str, default="train,val",
                        help="Comma-separated splits to process")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",")]

    for split in splits:
        depth_base = os.path.join(args.data_root, split, "depth")
        if not os.path.isdir(depth_base):
            print(f"WARNING: {depth_base} not found, skipping {split}")
            continue

        # Get all scenes
        scene_ids = sorted([
            d for d in os.listdir(depth_base)
            if os.path.isdir(os.path.join(depth_base, d))
        ])
        print(f"\n[{split}] Found {len(scene_ids)} scenes")

        # Collect tasks (using depth files as reference)
        tasks = []
        for scene_id in scene_ids:
            depth_dir = os.path.join(depth_base, scene_id)
            depth_files = [f for f in os.listdir(depth_dir) if f.endswith(".png")]
            for fname in depth_files:
                tasks.append((args.data_root, split, scene_id, fname))

        print(f"[{split}] Total tasks: {len(tasks)} frames")
        print(f"[{split}] Using {args.num_workers} workers")

        # Multi-process execution
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            results = list(tqdm(
                executor.map(process_single_frame, tasks, chunksize=200),
                total=len(tasks),
                desc=f"Processing {split}"
            ))

        records = [r for r in results if r is not None]

        print(f"\n{'='*60}")
        scenes = set(r["scene"] for r in records) if records else set()
        print(f"[{split}] {len(records)} frames from {len(scenes)} scenes")

        if records:
            output_path = os.path.join(args.output_dir, f"ddad_pixel_depth_{split}.jsonl")
            write_jsonl(records, output_path, split.capitalize())


if __name__ == "__main__":
    main()
