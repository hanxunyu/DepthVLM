"""
Shared depth validation module: check whether a depth map exists, is readable,
and contains at least one valid pixel.

All sampling scripts pass depth data root directories via --depth_root and
--depth_root2; during sampling, check_depth_valid() is called to filter out
samples with invalid depth maps.
"""
import json
import os

import cv2
import numpy as np


# ===== Per-dataset depth range configuration =====
DATASET_DEPTH_RANGES = {
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


def match_depth_range(jsonl_path: str) -> dict:
    """Auto-match depth range config based on the JSONL file path."""
    path_lower = jsonl_path.lower()
    for key, cfg in DATASET_DEPTH_RANGES.items():
        if key in path_lower:
            return cfg
    return {"min_depth": 0.0, "max_depth": float("inf")}


def check_depth_valid(record_json_str, depth_root, depth_root2=None, min_depth=0.0, max_depth=float("inf")):
    """Check whether the depth map for a JSONL record is valid.

    Validity conditions:
      1. The record contains depth_path and depth_scale fields.
      2. The depth file exists and is readable.
      3. After applying depth_scale to convert to meters, at least 1 pixel
         falls within [min_depth, max_depth].

    Args:
        record_json_str: raw JSONL line string (str or bytes).
        depth_root: depth map root directory.
        depth_root2: fallback depth root directory (optional).
        min_depth: minimum valid depth in meters.
        max_depth: maximum valid depth in meters.

    Returns:
        True if the depth map is valid, False otherwise.
    """
    if depth_root is None:
        return True  # No depth_root specified; skip validation

    try:
        if isinstance(record_json_str, bytes):
            record_json_str = record_json_str.decode("utf-8", errors="replace")
        record = json.loads(record_json_str)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False

    depth_rel = record.get("depth_path")
    if not depth_rel:
        return False

    depth_scale = record.get("depth_scale")
    if depth_scale is None:
        return False

    depth_format = record.get("depth_format")

    # Locate depth file
    depth_abs = os.path.join(depth_root, depth_rel.lstrip("/"))
    if not os.path.exists(depth_abs) and depth_root2:
        depth_abs = os.path.join(depth_root2, depth_rel.lstrip("/"))
    if not os.path.exists(depth_abs):
        return False

    # Load depth map
    try:
        if depth_format == "binary_float32":
            with open(depth_abs, "rb") as f:
                depth_m = np.fromfile(f, dtype=np.float32)
            total = len(depth_m)
            for w, h in [(6048, 4032)]:
                if w * h == total:
                    depth_m = depth_m.reshape(h, w)
                    break
            else:
                return False
            depth_m[~np.isfinite(depth_m)] = 0.0
        else:
            depth_raw = cv2.imread(depth_abs, cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                return False
            depth_m = depth_raw.astype(np.float32) / depth_scale
    except Exception:
        return False

    # Check if any pixel falls within valid range
    in_range = np.sum((depth_m >= min_depth) & (depth_m <= max_depth))
    return int(in_range) > 0
