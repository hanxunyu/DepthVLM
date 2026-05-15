#!/usr/bin/env python
"""
Extract NYU v2 RGB, depth, and intrinsics from SUN RGB-D into a unified directory structure.

Source data:
    sun_rgbd_official_link/SUNRGBD/kv1/NYUdata/NYU{0001..1449}/
    ├── image/NYU{xxxx}.jpg
    ├── depth_bfx/NYU{xxxx}.png      (uint16, /10000 = meters)
    └── intrinsics.txt                (3×3 intrinsic matrix, space-separated)

Split information:
    sun_rgbd_official_link/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat

Output:
    nyuv2_extracted/
    ├── train/
    │   ├── rgb/NYU{xxxx}.jpg
    │   ├── depth/NYU{xxxx}.png
    │   └── intrinsics/NYU{xxxx}.json   → [fx, fy, cx, cy, W, H]
    └── test/
        └── (same as above)

Usage:
    python extract_nyuv2_from_sunrgbd.py \
        --sunrgbd_root /path/to/sun_rgbd_official_link \
        --output_dir /path/to/nyuv2_extracted
"""
import argparse
import json
import os
import shutil

import scipy.io as sio
import numpy as np
from tqdm import tqdm


def load_allsplit(mat_path):
    """Load train/test split from allsplit.mat.

    Returns:
        train_set: Set of sample paths containing 'NYUdata'.
        test_set: Set of sample paths containing 'NYUdata'.
    """
    mat = sio.loadmat(mat_path)

    train_set = set()
    test_set = set()

    # allsplit.mat may contain trainval/test or train/test
    for key in mat.keys():
        if key.startswith("_"):
            continue
        arr = mat[key]
        items = []
        for item in arr.flat:
            if hasattr(item, "__len__") and len(item) > 0:
                s = str(item[0]) if hasattr(item[0], '__len__') else str(item)
                items.append(s)
            else:
                items.append(str(item))

        # Keep only NYUdata-related entries
        nyu_items = [s for s in items if "NYUdata" in s]

        if "train" in key.lower():
            train_set.update(nyu_items)
        elif "test" in key.lower():
            test_set.update(nyu_items)

    return train_set, test_set


def extract_nyu_id(path_str):
    """Extract NYU ID from a path string, e.g. 'NYU0001'."""
    for part in path_str.replace("\\", "/").split("/"):
        if part.startswith("NYU") and part[3:].isdigit():
            return part
    return None


def read_intrinsics(txt_path):
    """Read intrinsics.txt → (fx, fy, cx, cy).

    Format: fx 0 cx 0 fy cy 0 0 1 (3×3 matrix flattened row-major)
    """
    with open(txt_path, "r") as f:
        vals = f.read().strip().split()
    vals = [float(v) for v in vals]
    # 3×3 matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    fx = vals[0]
    cx = vals[2]
    fy = vals[4]
    cy = vals[5]
    return fx, fy, cx, cy


def main():
    parser = argparse.ArgumentParser(description="Extract NYUv2 from SUN RGB-D into unified structure.")
    parser.add_argument("--sunrgbd_root", type=str, required=True,
                        help="Root of sun_rgbd_official_link/")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output dir for extracted nyuv2 data")
    args = parser.parse_args()

    nyu_data_dir = os.path.join(args.sunrgbd_root, "SUNRGBD", "kv1", "NYUdata")
    allsplit_path = os.path.join(args.sunrgbd_root, "SUNRGBDtoolbox", "traintestSUNRGBD", "allsplit.mat")

    # Load split
    print("Loading allsplit.mat...")
    train_paths, test_paths = load_allsplit(allsplit_path)
    print(f"  Train paths with NYU: {len(train_paths)}")
    print(f"  Test paths with NYU: {len(test_paths)}")

    # Extract NYU IDs
    train_ids = set()
    for p in train_paths:
        nid = extract_nyu_id(p)
        if nid:
            train_ids.add(nid)

    test_ids = set()
    for p in test_paths:
        nid = extract_nyu_id(p)
        if nid:
            test_ids.add(nid)

    print(f"  Train NYU IDs: {len(train_ids)}")
    print(f"  Test NYU IDs: {len(test_ids)}")
    overlap = train_ids & test_ids
    if overlap:
        print(f"  WARNING: {len(overlap)} IDs in both train and test!")

    # Get all NYU directories
    all_nyu_dirs = sorted([
        d for d in os.listdir(nyu_data_dir)
        if d.startswith("NYU") and os.path.isdir(os.path.join(nyu_data_dir, d))
    ])
    print(f"\nFound {len(all_nyu_dirs)} NYU directories")

    # Create output directories
    for split in ["train", "test"]:
        for sub in ["rgb", "depth", "intrinsics"]:
            os.makedirs(os.path.join(args.output_dir, split, sub), exist_ok=True)

    stats = {"train": 0, "test": 0, "unknown": 0}

    for nyu_id in tqdm(all_nyu_dirs, desc="Extracting"):
        src_dir = os.path.join(nyu_data_dir, nyu_id)

        # Determine split
        if nyu_id in train_ids:
            split = "train"
        elif nyu_id in test_ids:
            split = "test"
        else:
            stats["unknown"] += 1
            continue

        # Source file paths
        rgb_src = os.path.join(src_dir, "image", f"{nyu_id}.jpg")
        depth_src = os.path.join(src_dir, "depth_bfx", f"{nyu_id}.png")
        intr_src = os.path.join(src_dir, "intrinsics.txt")

        if not os.path.exists(rgb_src) or not os.path.exists(depth_src):
            continue

        # Destination paths
        rgb_dst = os.path.join(args.output_dir, split, "rgb", f"{nyu_id}.jpg")
        depth_dst = os.path.join(args.output_dir, split, "depth", f"{nyu_id}.png")
        intr_dst = os.path.join(args.output_dir, split, "intrinsics", f"{nyu_id}.json")

        # Copy RGB and depth (use hard link for speed)
        try:
            os.link(rgb_src, rgb_dst)
        except OSError:
            shutil.copy2(rgb_src, rgb_dst)

        try:
            os.link(depth_src, depth_dst)
        except OSError:
            shutil.copy2(depth_src, depth_dst)

        # Read intrinsics and save as JSON
        if os.path.exists(intr_src):
            fx, fy, cx, cy = read_intrinsics(intr_src)
            # Get image size
            # SUN RGB-D NYU: 561×427
            from PIL import Image
            img = Image.open(rgb_src)
            w, h = img.size
            with open(intr_dst, "w") as f:
                json.dump([fx, fy, cx, cy, w, h], f)

        stats[split] += 1

    print(f"\n{'='*60}")
    print(f"Extracted: train={stats['train']}, test={stats['test']}, unknown={stats['unknown']}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
