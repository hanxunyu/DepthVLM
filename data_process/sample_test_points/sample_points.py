#!/usr/bin/env python
"""
sample_points.py — Sample depth points for evaluation.

Reads a JSONL file (produced by curate_datasets/ and sampled by sample_test_images/),
and for each record:
  1. Loads the raw depth map; applies mask/range filtering (no resize).
  2. Randomly samples N valid pixels at the original resolution.
  3. Reads the z-depth values directly from the depth map.
  4. Appends pixel_coords / depth / depth_type fields to the record.

Output JSONL format:
  {
    ...all original record fields...,
    "pixel_coords": [[col, row], ...],    # sampled coordinates at original resolution
    "depth": [d1, d2, ...],               # corresponding z-depth values in meters
    "depth_type": "z_depth"
  }

Usage:
    python sample_points.py \\
        --input_jsonl /path/to/test_sampled.jsonl \\
        --data_root /path/to/datasets \\
        --output_dir /path/to/output \\
        --points_per_image 10 \\
        --seed 42
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm


# ===== Per-dataset depth range configuration =====
DATASET_CONFIGS = {
    "argoverse":   {"min_depth": 0.05,  "max_depth": 120.0},
    "waymo":       {"min_depth": 0.05,  "max_depth": 70.0},
    "nuscenes":    {"min_depth": 0.05,  "max_depth": 80.0},
    "ddad":        {"min_depth": 0.05,  "max_depth": 120.0},
    "scannetpp":   {"min_depth": 0.001, "max_depth": 10.0},
    "scannet":     {"min_depth": 0.001, "max_depth": 10.0},
    "taskonomy":   {"min_depth": 0.005, "max_depth": 15.0},
    "hm3d":        {"min_depth": 0.01,  "max_depth": 10.0},
    "matterport":  {"min_depth": 0.01,  "max_depth": 10.0},
    "sunrgbd":     {"min_depth": 0.005, "max_depth": 8.0},
    "sun_rgbd":    {"min_depth": 0.005, "max_depth": 8.0},
    "ibims":       {"min_depth": 0.005, "max_depth": 25.0},
    "nyuv2":       {"min_depth": 0.005, "max_depth": 10.0},
    "eth3d":       {"min_depth": 0.01,  "max_depth": 50.0},
}


def match_dataset_config(jsonl_path: str) -> dict:
    """Match dataset config by checking if a known key appears in the JSONL path."""
    path_lower = jsonl_path.lower()
    for key, cfg in DATASET_CONFIGS.items():
        if key in path_lower:
            return cfg
    return {"min_depth": 0.0, "max_depth": float("inf")}


def load_depth_raw(depth_abs_path, depth_scale, min_depth, max_depth,
                   mask_invalid_path=None, mask_transp_path=None,
                   depth_format=None):
    """Load depth map at original resolution, apply masks and range filtering.

    Returns:
        depth_m: (H, W) float32 depth in meters, or None on failure.
    """
    try:
        if depth_format == "binary_float32":
            with open(depth_abs_path, "rb") as f:
                depth_m = np.fromfile(f, dtype=np.float32)
            total = len(depth_m)
            for w, h in [(6048, 4032)]:
                if w * h == total:
                    depth_m = depth_m.reshape(h, w)
                    break
            else:
                return None
            depth_m[~np.isfinite(depth_m)] = 0.0
        else:
            depth_raw = cv2.imread(depth_abs_path, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                return None
            depth_m = depth_raw.astype(np.float32) / depth_scale

        if mask_invalid_path and os.path.exists(mask_invalid_path):
            mask = cv2.imread(mask_invalid_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape[:2] != depth_m.shape[:2]:
                    mask = cv2.resize(mask, (depth_m.shape[1], depth_m.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                depth_m[mask == 0] = 0.0

        if mask_transp_path and os.path.exists(mask_transp_path):
            mask_t = cv2.imread(mask_transp_path, cv2.IMREAD_GRAYSCALE)
            if mask_t is not None:
                if mask_t.shape[:2] != depth_m.shape[:2]:
                    mask_t = cv2.resize(mask_t, (depth_m.shape[1], depth_m.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                depth_m[mask_t == 0] = 0.0

        depth_m[(depth_m < min_depth) | (depth_m > max_depth)] = 0.0
        return depth_m
    except Exception as e:
        print(f"ERROR loading depth {depth_abs_path}: {e}")
        return None


def sample_and_compute(depth_m, points_per_image, rng):
    """Sample points on the depth map and read z-depth values.

    Args:
        depth_m: (H, W) float32, meters, 0=invalid.
        points_per_image: number of points to sample.
        rng: numpy RandomState instance.

    Returns:
        pixel_coords: [[col, row], ...] coordinates at original resolution.
        depths: [float, ...] z-depth values in meters.
    """
    valid_indices = np.argwhere(depth_m > 0)  # (N, 2) [row, col]
    if len(valid_indices) == 0:
        return [], []

    n = min(points_per_image, len(valid_indices))
    sampled_idx = rng.choice(len(valid_indices), size=n, replace=False)
    sampled_pixels = valid_indices[sampled_idx]

    rows = sampled_pixels[:, 0]
    cols = sampled_pixels[:, 1]
    depth_values = depth_m[rows, cols]

    pixel_coords = [[int(c), int(r)] for r, c in zip(rows, cols)]
    depths = [round(float(d), 4) for d in depth_values]
    return pixel_coords, depths


def process_record(record, data_root, depth_root, min_depth, max_depth,
                   points_per_image, seed, idx):
    """Process a single record. Returns (output_record, None) or (None, skip_reason)."""
    rng = np.random.RandomState(seed + idx)

    depth_scale = record.get("depth_scale")
    if not depth_scale:
        return None, f"[idx={idx}] no depth_scale"

    depth_rel = record.get("depth_path")
    if not depth_rel:
        return None, f"[idx={idx}] no depth_path"

    depth_abs = os.path.join(depth_root, depth_rel.lstrip("/"))
    depth_format = record.get("depth_format")

    mask_invalid_path = None
    mask_rel = record.get("mask_valid_path") or record.get("mask_invalid_path")
    if mask_rel:
        mask_invalid_path = os.path.join(depth_root, mask_rel.lstrip("/"))

    mask_transp_path = None
    mask_transp_rel = record.get("mask_transp_path")
    if mask_transp_rel:
        mask_transp_path = os.path.join(depth_root, mask_transp_rel.lstrip("/"))

    depth_m = load_depth_raw(
        depth_abs, depth_scale, min_depth, max_depth,
        mask_invalid_path, mask_transp_path, depth_format,
    )
    if depth_m is None:
        return None, f"[idx={idx}] failed to load depth: {depth_abs}"

    pixel_coords, depths = sample_and_compute(depth_m, points_per_image, rng)
    if len(pixel_coords) == 0:
        return None, f"[idx={idx}] no valid depth pixels"

    # Append sampling results to the original record
    out = dict(record)
    out["pixel_coords"] = pixel_coords
    out["depth"] = depths
    out["depth_type"] = "z_depth"
    return out, None


def _process_one(args_tuple):
    """Wrapper for multiprocessing Pool."""
    (idx, record, data_root, depth_root, min_depth, max_depth,
     points_per_image, seed) = args_tuple
    return process_record(
        record, data_root, depth_root, min_depth, max_depth,
        points_per_image, seed, idx,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Sample depth points at original resolution without processing/saving images.",
    )
    parser.add_argument("--input_jsonl", required=True,
                        help="Path to input JSONL file (from sample_test_images/)")
    parser.add_argument("--data_root", required=True,
                        help="Dataset root directory (common prefix for image/depth/mask paths)")
    parser.add_argument("--depth_root", default=None,
                        help="Depth root directory (defaults to data_root if not specified)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory")
    parser.add_argument("--output_jsonl_name", default="sampled.jsonl",
                        help="Output JSONL filename")
    parser.add_argument("--points_per_image", type=int, default=10,
                        help="Number of points to sample per image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_points", type=int, default=0,
                        help="Target total number of sampled points. If > 0, automatically "
                             "adjusts points_per_image so that the total equals this value.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of worker processes (default: 0 = single process)")
    args = parser.parse_args()

    if args.depth_root is None:
        args.depth_root = args.data_root

    cfg = match_dataset_config(args.input_jsonl)
    min_depth = cfg["min_depth"]
    max_depth = cfg["max_depth"]

    print(f"Input:  {args.input_jsonl}")
    print(f"  data_root:  {args.data_root}")
    print(f"  depth_root: {args.depth_root}")
    print(f"  output_dir: {args.output_dir}")
    print(f"  depth range: [{min_depth}, {max_depth}]")
    print(f"  points_per_image: {args.points_per_image}")
    print(f"  total_points: {args.total_points}")
    print(f"  seed: {args.seed}")

    with open(args.input_jsonl, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"  Loaded {len(records)} records")

    # --- Compute actual points per image ---
    num_records = len(records)
    if args.total_points > 0 and num_records > 0:
        # Distribute total_points evenly across all records
        base_ppi = args.total_points // num_records
        remainder = args.total_points % num_records
        per_image_points = [base_ppi + (1 if i < remainder else 0)
                            for i in range(num_records)]
        print(f"  total_points={args.total_points}, num_records={num_records}")
        print(f"  -> base points_per_image={base_ppi}, "
              f"remainder={remainder} images get {base_ppi+1} points")
    else:
        per_image_points = [args.points_per_image] * num_records

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_jsonl_name)

    task_args = [
        (idx, record, args.data_root, args.depth_root, min_depth, max_depth,
         per_image_points[idx], args.seed)
        for idx, record in enumerate(records)
    ]

    success = 0
    skipped = 0
    skip_reasons = []

    if args.num_workers > 0:
        with Pool(processes=args.num_workers) as pool, \
             open(output_path, "w") as out_f:
            for out_record, skip_reason in tqdm(
                pool.imap(_process_one, task_args),
                total=len(task_args), desc="Sampling",
            ):
                if out_record is None:
                    skipped += 1
                    skip_reasons.append(skip_reason)
                    continue
                out_f.write(json.dumps(out_record) + "\n")
                success += 1
    else:
        with open(output_path, "w") as out_f:
            for t in tqdm(task_args, desc="Sampling"):
                out_record, skip_reason = _process_one(t)
                if out_record is None:
                    skipped += 1
                    skip_reasons.append(skip_reason)
                    continue
                out_f.write(json.dumps(out_record) + "\n")
                success += 1

    print(f"\nDone! {success} records saved, {skipped} skipped.")
    if skip_reasons:
        from collections import Counter
        cats = Counter()
        for r in skip_reasons:
            if "no depth_scale" in r: cats["no depth_scale"] += 1
            elif "no depth_path" in r: cats["no depth_path"] += 1
            elif "failed to load" in r: cats["failed to load depth"] += 1
            elif "no valid" in r: cats["no valid depth pixels"] += 1
            else: cats["other"] += 1
        print("  Skip reasons:")
        for reason, count in cats.most_common():
            print(f"    {reason}: {count}")
        for r in skip_reasons[:5]:
            print(f"    {r}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
