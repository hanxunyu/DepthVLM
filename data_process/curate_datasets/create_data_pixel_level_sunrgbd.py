#!/usr/bin/env python
"""
Create pixel-level depth dataset metadata for SUN RGB-D.

Directory structure:
    sun_rgbd_official_link/
    ├── SUNRGBD/
    │   ├── kv1/   (b3dodata, NYUdata)
    │   ├── kv2/   (align_kv2, kinect2data)
    │   ├── realsense/ (lg, sa, sh, shr)
    │   └── xtion/ (sun3ddata, xtion_align_data)
    └── SUNRGBDtoolbox/
        └── traintestSUNRGBD/
            └── allsplit.mat

Each scene directory contains:
    {scene}/image/xxx.jpg           ← RGB
    {scene}/depth_bfx/xxx.png       ← depth (uint16, /10000 = meters)
    {scene}/intrinsics.txt          ← 3×3 intrinsic matrix: fx 0 cx 0 fy cy 0 0 1

Usage:
    python create_data_pixel_level_sunrgbd.py \
        --data_root /path/to/sun_rgbd_official_link \
        --output_dir ./annotations/sunrgbd
"""
import argparse
import json
import os
from glob import glob

import cv2
import numpy as np
import scipy.io
from tqdm import tqdm

DEPTH_SCALE = 8000.0
CANONICAL_FX = 1000.0


def find_all_scenes(sunrgbd_root):
    """Recursively find all scene directories containing an image/ subdirectory."""
    scenes = []
    for root, dirs, files in os.walk(sunrgbd_root):
        if "image" in dirs:
            img_dir = os.path.join(root, "image")
            imgs = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
            if imgs:
                scenes.append(root)
    return sorted(scenes)


def get_split_paths(split_mat_path):
    """Extract train/test path sets from allsplit.mat."""
    mat = scipy.io.loadmat(split_mat_path)
    train_paths = set()
    test_paths = set()
    for s in mat['alltrain'][0]:
        p = s[0]
        key = p.split("SUNRGBD/")[-1].rstrip("/")
        train_paths.add(key)
    for s in mat['alltest'][0]:
        p = s[0]
        key = p.split("SUNRGBD/")[-1].rstrip("/")
        test_paths.add(key)
    return train_paths, test_paths


def scene_to_key(scene_path, sunrgbd_root):
    return os.path.relpath(scene_path, sunrgbd_root)


def read_intrinsics(txt_path):
    """Read intrinsics.txt → (fx, fy, cx, cy).

    Format: fx 0 cx 0 fy cy 0 0 1 (3×3 row-major)
    """
    with open(txt_path, "r") as f:
        vals = f.read().strip().split()
    vals = [float(v) for v in vals]
    fx = vals[0]
    cx = vals[2]
    fy = vals[4]
    cy = vals[5]
    return fx, fy, cx, cy


def process_scene(scene_dir, sunrgbd_root, args_data_root):
    """Process a single scene and return a metadata record or None."""
    # Find RGB
    img_dir = os.path.join(scene_dir, "image")
    imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    if not imgs:
        return None
    image_fname = imgs[0]
    image_path = os.path.join(img_dir, image_fname)

    # Find depth (prefer depth_bfx)
    depth_dir = os.path.join(scene_dir, "depth_bfx")
    if not os.path.isdir(depth_dir):
        depth_dir = os.path.join(scene_dir, "depth")
    if not os.path.isdir(depth_dir):
        return None
    depths = sorted([f for f in os.listdir(depth_dir) if f.lower().endswith((".png", ".jpg"))])
    if not depths:
        return None
    depth_fname = depths[0]
    depth_path = os.path.join(depth_dir, depth_fname)

    # Read RGB size (varies by sensor)
    rgb = cv2.imread(image_path)
    if rgb is None:
        return None
    rgb_h, rgb_w = rgb.shape[:2]

    # Read depth size
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        return None
    depth_h, depth_w = depth_raw.shape[:2]

    # Read intrinsics
    intr_path = os.path.join(scene_dir, "intrinsics.txt")
    fx = None
    if os.path.exists(intr_path):
        try:
            fx, fy, cx, cy = read_intrinsics(intr_path)
        except Exception:
            pass

    # Compute canonical size
    if fx and fx > 0:
        scale = CANONICAL_FX / fx
        canonical_w = round(rgb_w * scale)
        canonical_h = round(rgb_h * scale)
    else:
        canonical_w = canonical_h = None

    # Relative path
    data_root_parent = os.path.dirname(args_data_root)
    image_rel = os.path.relpath(image_path, data_root_parent)
    depth_rel = os.path.relpath(depth_path, data_root_parent)

    record = {
        "image": image_rel,
        "depth_path": depth_rel,
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


def write_jsonl(records, path, label):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {label}: {len(records)} frames → {path}")
    if records:
        fxs = [r["original_fx"] for r in records if r.get("original_fx")]
        if fxs:
            print(f"    fx range: [{min(fxs):.1f}, {max(fxs):.1f}]")
        sizes = set(tuple(r["original_rgb_size"]) for r in records)
        print(f"    RGB resolutions: {sizes}")
        canonical_sizes = set(tuple(r["canonical_size"]) for r in records if r.get("canonical_size"))
        if canonical_sizes:
            sizes_list = sorted(canonical_sizes)
            print(f"    Canonical sizes ({len(sizes_list)}): {sizes_list[:5]}..." if len(sizes_list) > 5 else f"    Canonical sizes: {sizes_list}")
        no_fx = sum(1 for r in records if not r.get("original_fx"))
        if no_fx:
            print(f"    WARNING: {no_fx} frames without fx")


def main():
    parser = argparse.ArgumentParser(description="Create pixel-level depth metadata for SUN RGB-D.")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root dir (contains SUNRGBD/ and SUNRGBDtoolbox/)")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sunrgbd_root = os.path.join(args.data_root, "SUNRGBD")
    split_mat = os.path.join(args.data_root, "SUNRGBDtoolbox", "traintestSUNRGBD", "allsplit.mat")

    print(f"Loading split from {split_mat}")
    train_keys, test_keys = get_split_paths(split_mat)
    print(f"Split: train={len(train_keys)}, test={len(test_keys)}")

    print(f"Scanning scenes in {sunrgbd_root}...")
    all_scenes = find_all_scenes(sunrgbd_root)
    print(f"Found {len(all_scenes)} scenes")

    train_records = []
    test_records = []
    unmatched = 0

    for scene_dir in tqdm(all_scenes, desc="sunrgbd"):
        key = scene_to_key(scene_dir, sunrgbd_root)

        if key in train_keys:
            split = "train"
        elif key in test_keys:
            split = "test"
        else:
            matched = False
            for tk in train_keys:
                if key.endswith(tk.split("/")[-1]) or tk.endswith(key.split("/")[-1]):
                    split = "train"
                    matched = True
                    break
            if not matched:
                for tk in test_keys:
                    if key.endswith(tk.split("/")[-1]) or tk.endswith(key.split("/")[-1]):
                        split = "test"
                        matched = True
                        break
            if not matched:
                unmatched += 1
                continue

        record = process_scene(scene_dir, sunrgbd_root, args.data_root)
        if record is None:
            continue

        if split == "train":
            train_records.append(record)
        else:
            test_records.append(record)

    print(f"\n{'='*60}")
    print(f"Total: train={len(train_records)}, test={len(test_records)}, unmatched={unmatched}")

    write_jsonl(train_records,
                os.path.join(args.output_dir, "sunrgbd_pixel_depth_train.jsonl"), "Train")
    write_jsonl(test_records,
                os.path.join(args.output_dir, "sunrgbd_pixel_depth_test.jsonl"), "Test")


if __name__ == "__main__":
    main()
