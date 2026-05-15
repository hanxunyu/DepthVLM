#!/usr/bin/env python
"""
Create pixel-level depth dataset metadata for iBims-1.

Directory structure:
    ibims1_core_raw/
    ├── rgb/{name}.png           (640×480)
    ├── depth/{name}.png         (uint16, depth_m = raw * 50.0 / 65535)
    ├── calib/{name}.txt         → fx,fy,cx,cy
    ├── mask_invalid/{name}.png  (1-bit, >0 = valid region)
    └── mask_transp/{name}.png   (1-bit, >0 = non-transparent object region)

iBims-1 contains only 100 images, used for testing only (no training set).

Usage:
    python create_data_pixel_level_ibims1.py \
        --data_root /path/to/ibims1/ibims1_core_raw \
        --output_dir ./annotations/ibims1
"""
import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

# iBims-1 depth encoding: depth_m = raw_uint16 * 50.0 / 65535
# Equivalent to depth_m = raw_uint16 / (65535 / 50.0) = raw_uint16 / 1310.7
DEPTH_SCALE = 65535.0 / 50.0  # ≈ 1310.7
CANONICAL_FX = 1000.0


def read_calib(calib_path):
    """Read calibration txt → (fx, fy, cx, cy)."""
    with open(calib_path, "r") as f:
        line = f.read().strip()
    parts = line.split(",")
    fx, fy, cx, cy = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    return fx, fy, cx, cy


def process_frame(data_root: str, name: str) -> dict | None:
    """Process a single frame and return a metadata record."""
    rgb_path = os.path.join(data_root, "rgb", f"{name}.png")
    depth_path = os.path.join(data_root, "depth", f"{name}.png")
    calib_path = os.path.join(data_root, "calib", f"{name}.txt")

    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        return None

    # Get RGB size
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        return None
    rgb_h, rgb_w = rgb.shape[:2]

    # Get depth size
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        return None
    depth_h, depth_w = depth_raw.shape[:2]

    # Read intrinsics
    fx = None
    if os.path.exists(calib_path):
        fx, fy, cx, cy = read_calib(calib_path)

    # Compute canonical size
    if fx and fx > 0:
        scale = CANONICAL_FX / fx
        canonical_w = round(rgb_w * scale)
        canonical_h = round(rgb_h * scale)
    else:
        canonical_w = canonical_h = None

    record = {
        "image": f"ibims1/ibims1_core_raw/rgb/{name}.png",
        "depth_path": f"ibims1/ibims1_core_raw/depth/{name}.png",
        "mask_valid_path": f"ibims1/ibims1_core_raw/mask_invalid/{name}.png",
        "mask_transp_path": f"ibims1/ibims1_core_raw/mask_transp/{name}.png",
        "depth_scale": DEPTH_SCALE,
        "original_rgb_size": [rgb_w, rgb_h],
        "original_depth_size": [depth_w, depth_h],
        "original_fx": round(fx, 2) if fx else None,
        "canonical_fx": CANONICAL_FX,
        "canonical_size": [canonical_w, canonical_h] if canonical_w else None,
        "prompt": "Given this image, estimate the metric depth (in meters, rounded to two decimal places) for every pixel of the image.",
        "solution": "OK, I will estimate the depth map of this image.",
    }
    return record


def main():
    parser = argparse.ArgumentParser(description="Create pixel-level depth metadata for iBims-1.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root dir of iBims-1 (contains rgb/, depth/, calib/ subdirs)")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rgb_dir = os.path.join(args.data_root, "rgb")
    names = sorted([
        os.path.splitext(f)[0] for f in os.listdir(rgb_dir)
        if f.endswith(".png") and not f.startswith(".")
    ])
    print(f"Found {len(names)} images in {rgb_dir}")

    output_path = os.path.join(args.output_dir, "ibims1_pixel_depth_test.jsonl")

    records = []
    for name in tqdm(names, desc="ibims1"):
        record = process_frame(args.data_root, name)
        if record is not None:
            records.append(record)

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nTest: {len(records)} frames saved to {output_path}")
    if records:
        fxs = [r["original_fx"] for r in records if r.get("original_fx")]
        if fxs:
            print(f"  fx range: [{min(fxs):.1f}, {max(fxs):.1f}]")
        sizes = set(tuple(r["original_rgb_size"]) for r in records)
        print(f"  RGB resolutions: {sizes}")
        canonical_sizes = set(tuple(r["canonical_size"]) for r in records if r.get("canonical_size"))
        if canonical_sizes:
            print(f"  Canonical sizes ({len(canonical_sizes)}): {sorted(canonical_sizes)[:5]}...")


if __name__ == "__main__":
    main()
