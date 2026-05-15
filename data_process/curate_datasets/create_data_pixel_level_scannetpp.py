#!/usr/bin/env python
"""
Create pixel-level depth dataset metadata for ScanNet++ (iPhone captures).

Directory structure:
    scannetpp_1fps/{scene_id}/frame_XXXXXX.jpg          (RGB, 1920×1440)
    scannetpp/data/{scene_id}/iphone/
    ├── depth/frame_XXXXXX.png                           (depth, 256×192, uint16, /1000=meters)
    └── pose_intrinsic_imu.json                          (per-frame intrinsics + pose)

Intrinsic format (3×3 matrix):
    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

Usage:
    python create_data_pixel_level_scannetpp.py \
        --rgb_root /path/to/scannetpp_1fps \
        --depth_root /path/to/scannetpp/data \
        --train_split /path/to/scannetpp/splits/nvs_sem_train.txt \
        --val_split /path/to/scannetpp/splits/nvs_sem_val.txt \
        --output_dir ./annotations/scannetpp \
        --num_workers 32
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

DEPTH_SCALE = 1000.0  # ScanNet++ iPhone depth: uint16 / 1000 = meters
CANONICAL_FX = 1000.0
RGB_W, RGB_H = 1920, 1440
DEPTH_W, DEPTH_H = 256, 192


def load_split(split_path: str) -> list[str]:
    with open(split_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_intrinsics(json_path: str) -> dict:
    """Load pose_intrinsic_imu.json and return {frame_name: fx}."""
    frame_to_fx = {}
    with open(json_path, "r") as f:
        data = json.load(f)
    for frame_key, frame_data in data.items():
        intr = frame_data.get("intrinsic")
        if intr:
            fx = intr[0][0]  # [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            frame_to_fx[frame_key] = fx
    return frame_to_fx


def process_scene(scene_id, rgb_root, depth_root):
    """Process a single scene and return a list of records."""
    results = []

    scene_rgb_dir = os.path.join(rgb_root, scene_id)
    if not os.path.isdir(scene_rgb_dir):
        return results

    # Load intrinsics
    intr_json_path = os.path.join(depth_root, scene_id, "iphone", "pose_intrinsic_imu.json")
    frame_to_fx = {}
    if os.path.exists(intr_json_path):
        try:
            frame_to_fx = load_intrinsics(intr_json_path)
        except Exception:
            pass

    frames = sorted([
        f for f in os.listdir(scene_rgb_dir)
        if f.endswith((".jpg", ".png", ".jpeg"))
    ])

    for frame_name in frames:
        stem = Path(frame_name).stem  # frame_000000

        # Check if depth file exists
        depth_rel = os.path.join(scene_id, "iphone", "depth", f"{stem}.png")
        depth_abs = os.path.join(depth_root, depth_rel)
        if not os.path.exists(depth_abs):
            continue

        # Get fx
        fx = frame_to_fx.get(stem)

        # Compute canonical size
        if fx and fx > 0:
            scale = CANONICAL_FX / fx
            canonical_w = round(RGB_W * scale)
            canonical_h = round(RGB_H * scale)
        else:
            canonical_w = canonical_h = None

        record = {
            "scene_id": scene_id,
            "image": f"{scene_id}/{frame_name}",
            "depth_path": depth_rel,
            "depth_scale": DEPTH_SCALE,
            "original_rgb_size": [RGB_W, RGB_H],
            "original_depth_size": [DEPTH_W, DEPTH_H],
            "original_fx": round(fx, 2) if fx else None,
            "canonical_fx": CANONICAL_FX,
            "canonical_size": [canonical_w, canonical_h] if canonical_w else None,
            "prompt": "Given this image, estimate the metric depth (in meters, rounded to two decimal places) for every pixel of the image.",
            "solution": "OK, I will estimate the depth map of this image.",
        }
        results.append(record)

    return results


def main():
    parser = argparse.ArgumentParser(description="Create pixel-level depth metadata for ScanNet++.")
    parser.add_argument("--rgb_root", type=str, required=True)
    parser.add_argument("--depth_root", type=str, required=True)
    parser.add_argument("--train_split", type=str, required=True)
    parser.add_argument("--val_split", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_scenes = load_split(args.train_split)
    val_scenes = load_split(args.val_split)
    print(f"Train scenes: {len(train_scenes)}, Val scenes: {len(val_scenes)}")

    for split_name, scenes in [("train", train_scenes), ("val", val_scenes)]:
        output_path = os.path.join(args.output_dir, f"scannetpp_pixel_depth_{split_name}.jsonl")
        print(f"\n{'='*60}")
        print(f"Processing {split_name}: {len(scenes)} scenes")

        all_records = []

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(process_scene, sid, args.rgb_root, args.depth_root): sid
                for sid in scenes
            }
            with tqdm(total=len(futures), desc=split_name) as pbar:
                for future in as_completed(futures):
                    scene_id = futures[future]
                    try:
                        records = future.result()
                        all_records.extend(records)
                    except Exception as e:
                        print(f"ERROR {scene_id}: {e}")
                    pbar.update(1)
                    pbar.set_postfix(total=len(all_records))

        all_records.sort(key=lambda x: x["image"])

        with open(output_path, "w") as f:
            for r in all_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"{split_name}: {len(all_records)} frames → {output_path}")
        if all_records:
            fxs = [r["original_fx"] for r in all_records if r.get("original_fx")]
            if fxs:
                print(f"  fx range: [{min(fxs):.1f}, {max(fxs):.1f}]")
            canonical_sizes = set(tuple(r["canonical_size"]) for r in all_records if r.get("canonical_size"))
            if canonical_sizes:
                sizes_list = sorted(canonical_sizes)
                print(f"  Canonical sizes ({len(sizes_list)}): {sizes_list[:5]}..." if len(sizes_list) > 5 else f"  Canonical sizes: {sizes_list}")
            no_fx = sum(1 for r in all_records if not r.get("original_fx"))
            if no_fx:
                print(f"  WARNING: {no_fx} frames without fx")


if __name__ == "__main__":
    main()
