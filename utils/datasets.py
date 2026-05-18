# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import logging
import os
from typing import Any

import cv2
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

logger: logging.Logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ==================== Pixel-Level Depth Map Datasets ====================

DEFAULT_DEPTH_PROMPT = "Given this image, estimate the metric depth (in meters, rounded to two decimal places) for every pixel of the image."

# ===== Dataset configuration table =====
DATASET_CONFIGS = {
    "argoverse":   {"min_depth": 0.05,  "max_depth": 120.0},
    "waymo":       {"min_depth": 0.05,  "max_depth": 70.0},
    "nuscenes":    {"min_depth": 0.05,  "max_depth": 80.0},
    "ddad":        {"min_depth": 0.05,  "max_depth": 120.0},
    "scannetpp":   {"min_depth": 0.001, "max_depth": 10.0},
    "taskonomy":   {"min_depth": 0.005, "max_depth": 15.0, "max_canonical_edge": 1024},
    "hm3d":        {"min_depth": 0.01,  "max_depth": 10.0, "max_canonical_edge": 1024},
    "matterport":  {"min_depth": 0.01,  "max_depth": 10.0},
    "sunrgbd":     {"min_depth": 0.005, "max_depth": 8.0},
    "ibims":       {"min_depth": 0.005, "max_depth": 25.0},
    "nyuv2":       {"min_depth": 0.005, "max_depth": 10.0},
    "eth3d":       {"min_depth": 0.01,  "max_depth": 50.0},
}


def _match_dataset_config(jsonl_path: str) -> dict:
    """Auto-match the dataset configuration from a jsonl path."""
    path_lower = jsonl_path.lower()
    for key, cfg in DATASET_CONFIGS.items():
        if key in path_lower:
            return cfg
    return {"min_depth": 0.0, "max_depth": float("inf")}


def _clamp_canonical_size(canonical_size: list | None, max_edge: int | None = None):
    """Return crop info if canonical_size exceeds max_edge.

    Args:
        canonical_size: [W, H] or None.
        max_edge: max-edge length cap; None means no limit.

    Returns:
        (canonical_size, crop_size)
        - canonical_size: target size for resize.
        - crop_size: post-crop [W, H], or None if no cropping is needed.
    """
    if not canonical_size:
        return canonical_size, None
    if max_edge is None:
        return canonical_size, None
    tw, th = canonical_size
    if max(tw, th) <= max_edge:
        return canonical_size, None
    crop_w = min(tw, max_edge)
    crop_h = min(th, max_edge)
    return canonical_size, [crop_w, crop_h]


def _load_and_resize_image(image_path: str, canonical_size: list | None, crop_size: list | None = None) -> Image.Image | None:
    """Load RGB, resize to canonical_size, optionally center-crop."""
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"ERROR loading image {image_path}: {e}")
        return None

    if canonical_size:
        tw, th = canonical_size
        if image.size != (tw, th):
            image = image.resize((tw, th), Image.BILINEAR)

    if crop_size:
        # center crop
        w, h = image.size
        cw, ch = crop_size
        left = (w - cw) // 2
        top = (h - ch) // 2
        image = image.crop((left, top, left + cw, top + ch))

    return image


def _load_depth(
    depth_abs_path: str,
    depth_scale: float,
    min_depth: float,
    max_depth: float,
    canonical_size: list | None,
    crop_size: list | None = None,
    mask_valid_path: str | None = None,
    mask_transp_path: str | None = None,
    depth_format: str | None = None,
) -> np.ndarray | None:
    """Load depth, apply mask/range filtering, resize to canonical_size, optionally center-crop."""
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

        # Load mask_valid (iBims-1 mask_invalid directory; mask>0 = valid region)
        if mask_valid_path and os.path.exists(mask_valid_path):
            mask = cv2.imread(mask_valid_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape[:2] != depth_m.shape[:2]:
                    mask = cv2.resize(mask, (depth_m.shape[1], depth_m.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                depth_m[mask == 0] = 0.0  # mask==0 region is invalid

        # Load mask_transp (iBims-1 transparent-object mask; mask>0 = opaque/trustworthy region)
        if mask_transp_path and os.path.exists(mask_transp_path):
            mask_t = cv2.imread(mask_transp_path, cv2.IMREAD_GRAYSCALE)
            if mask_t is not None:
                if mask_t.shape[:2] != depth_m.shape[:2]:
                    mask_t = cv2.resize(mask_t, (depth_m.shape[1], depth_m.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                depth_m[mask_t == 0] = 0.0  # mask_transp==0 region is transparent; depth is unreliable

        # Zero out values outside the [min_depth, max_depth] range
        depth_m[(depth_m < min_depth) | (depth_m > max_depth)] = 0.0

        # Resize to canonical_size
        if canonical_size:
            tw, th = canonical_size
            if depth_m.shape[1] != tw or depth_m.shape[0] != th:
                depth_m = cv2.resize(depth_m, (tw, th), interpolation=cv2.INTER_NEAREST)

        # Center crop (keeps fx unchanged)
        if crop_size:
            h, w = depth_m.shape[:2]
            cw, ch = crop_size
            left = (w - cw) // 2
            top = (h - ch) // 2
            depth_m = depth_m[top:top+ch, left:left+cw]

        # Check valid pixels: all-zero depth maps cannot contribute to loss; skip with None
        if (depth_m > 0).sum() == 0:
            print(f"WARNING [_load_depth] depth map has NO valid pixels (all <= 0), skipping. "
                  f"path={depth_abs_path}, shape={depth_m.shape}, "
                  f"min={depth_m.min():.4f}, max={depth_m.max():.4f}, "
                  f"depth_range=[{min_depth}, {max_depth}]")
            return None

        return depth_m
    except Exception as e:
        print(f"ERROR loading depth {depth_abs_path}: {e}")
        return None


class dataset_pixel_depth_train(Dataset):
    """Training dataset for pixel-level depth prediction.

    Both RGB and depth are resized to the `canonical_size` from jsonl
    (the resolution after focal-length unification).
    """

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        depth_root: str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        print(f"[dataset_pixel_depth_train] reading data from {data_path}")
        print(f"  image_folder = {image_folder}")
        print(f"  depth_root = {depth_root}")

        data_paths = data_path.split(";")
        image_folders = image_folder.split(";")
        if len(image_folders) == 1:
            image_folders = image_folders * len(data_paths)
        if depth_root:
            depth_roots = depth_root.split(";")
            if len(depth_roots) == 1:
                depth_roots = depth_roots * len(data_paths)
        else:
            depth_roots = [None] * len(data_paths)

        self.list_data_dict = []
        self.dataset_configs = []
        for dp in data_paths:
            cfg = _match_dataset_config(dp)
            self.dataset_configs.append(cfg)
            if ".jsonl" in dp:
                with open(dp, "r") as f:
                    records = [json.loads(line) for line in f if line.strip()]
                self.list_data_dict.append(records)
            else:
                self.list_data_dict.append(json.load(open(dp, "r")))
            print(f"  [{os.path.basename(dp)}] {len(self.list_data_dict[-1])} samples, "
                  f"depth range=[{cfg['min_depth']}, {cfg['max_depth']}]")

        self.data_path = data_paths
        self.image_folder = image_folders
        self.depth_roots = depth_roots
        print(f"[dataset_pixel_depth_train] loaded {self.__len__()} samples")

    def __len__(self) -> int:
        return sum(len(d) for d in self.list_data_dict)

    def __getitem__(self, index):
        id_dataset = 0
        local_idx = index
        for i, d in enumerate(self.list_data_dict):
            if local_idx < len(d):
                id_dataset = i
                break
            local_idx -= len(d)

        record = self.list_data_dict[id_dataset][local_idx]
        cfg = self.dataset_configs[id_dataset]
        min_depth = cfg["min_depth"]
        max_depth = cfg["max_depth"]
        max_edge = cfg.get("max_canonical_edge")  # only hm3d/taskonomy have a value
        canonical_size, crop_size = _clamp_canonical_size(record.get("canonical_size"), max_edge)

        # Load RGB: resize to canonical_size, then center crop
        image_path = os.path.join(self.image_folder[id_dataset], record["image"].lstrip("/"))
        image = _load_and_resize_image(image_path, canonical_size, crop_size)
        if image is None:
            return self.__getitem__((index + 1) % self.__len__())

        result = {
            "image": image,
            "problem": record.get("prompt", DEFAULT_DEPTH_PROMPT),
            "solution": record.get("solution", ""),
            "system": "You are a helpful assistant.",
            "min_depth": min_depth,
            "max_depth": max_depth,
        }

        # Load depth: resize to canonical_size, then center crop
        depth_path_rel = record.get("depth_path")
        current_depth_root = self.depth_roots[id_dataset]
        if depth_path_rel and current_depth_root:
            depth_abs_path = os.path.join(current_depth_root, depth_path_rel)

            mask_valid_path = None
            mask_valid_rel = record.get("mask_valid_path")
            if mask_valid_rel and current_depth_root:
                mask_valid_path = os.path.join(current_depth_root, mask_valid_rel)

            mask_transp_path = None
            mask_transp_rel = record.get("mask_transp_path")
            if mask_transp_rel and current_depth_root:
                mask_transp_path = os.path.join(current_depth_root, mask_transp_rel)

            depth_m = _load_depth(
                depth_abs_path,
                depth_scale=record.get("depth_scale", 1000.0),
                min_depth=min_depth,
                max_depth=max_depth,
                canonical_size=canonical_size,
                crop_size=crop_size,
                mask_valid_path=mask_valid_path,
                mask_transp_path=mask_transp_path,
                depth_format=record.get("depth_format"),
            )
            if depth_m is not None:
                result["pixel_depth_labels"] = depth_m  # (H, W) np.float32
            else:
                # Depth failed to load or is all-zero; skip to the next sample
                return self.__getitem__((index + 1) % self.__len__())

        return result


class dataset_pixel_depth_eval(Dataset):
    """Evaluation dataset for pixel-level depth prediction.

    Both RGB and depth are resized to the `canonical_size` from jsonl.
    """

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        depth_root: str = None,
        **kwargs,
    ) -> None:
        super().__init__()
        print(f"[dataset_pixel_depth_eval] reading data from {data_path}")

        data_paths = data_path.split(";")
        image_folders = image_folder.split(";")
        if len(image_folders) == 1:
            image_folders = image_folders * len(data_paths)
        if depth_root:
            depth_roots = depth_root.split(";")
            if len(depth_roots) == 1:
                depth_roots = depth_roots * len(data_paths)
        else:
            depth_roots = [None] * len(data_paths)

        self.list_data_dict = []
        self.dataset_configs = []
        for dp in data_paths:
            dp = dp.strip()
            cfg = _match_dataset_config(dp)
            self.dataset_configs.append(cfg)
            if ".jsonl" in dp:
                with open(dp, "r") as f:
                    records = [json.loads(line) for line in f if line.strip()]
                self.list_data_dict.append(records)
            else:
                self.list_data_dict.append(json.load(open(dp, "r")))
            print(f"  [{os.path.basename(dp)}] {len(self.list_data_dict[-1])} samples, "
                  f"depth range=[{cfg['min_depth']}, {cfg['max_depth']}]")

        self.data_path = data_paths
        self.image_folder = [f.strip() for f in image_folders]
        self.depth_roots = [r.strip() if r else None for r in depth_roots]
        print(f"[dataset_pixel_depth_eval] loaded {self.__len__()} samples from {len(data_paths)} jsonl(s)")

    def __len__(self) -> int:
        return sum(len(d) for d in self.list_data_dict)

    def __getitem__(self, index) -> dict[str, Any]:
        id_dataset = 0
        local_idx = index
        for i, d in enumerate(self.list_data_dict):
            if local_idx < len(d):
                id_dataset = i
                break
            local_idx -= len(d)

        record = self.list_data_dict[id_dataset][local_idx]
        cfg = self.dataset_configs[id_dataset]
        min_depth = cfg["min_depth"]
        max_depth = cfg["max_depth"]
        max_edge = cfg.get("max_canonical_edge")
        canonical_size, crop_size = _clamp_canonical_size(record.get("canonical_size"), max_edge)
        cur_image_folder = self.image_folder[id_dataset]
        cur_depth_root = self.depth_roots[id_dataset]

        # Load RGB: resize to canonical_size, then center crop
        image_path = os.path.join(cur_image_folder, record["image"].lstrip("/"))
        image = _load_and_resize_image(image_path, canonical_size, crop_size)
        if image is None:
            return None

        result = {
            "image": image,
            "problem": record.get("prompt", DEFAULT_DEPTH_PROMPT),
            "image_name": record["image"].lstrip("/"),
            "system": "You are a helpful assistant.",
            "min_depth": min_depth,
            "max_depth": max_depth,
        }

        # ===== New format: pixel_coords + depth (sparse-points GT) =====
        if "pixel_coords" in record and "depth" in record:
            orig_coords = np.array(record["pixel_coords"], dtype=np.float64)  # (N, 2), each row [x, y]
            gt_depths_jsonl = np.array(record["depth"], dtype=np.float32)     # (N,) values in the original jsonl resolution

            # --- Coordinate transform: original image -> canonical_size -> crop ---
            orig_rgb_size = record.get("original_rgb_size")  # [W_orig, H_orig]
            if orig_rgb_size and canonical_size:
                w_orig, h_orig = orig_rgb_size
                w_canon, h_canon = canonical_size
                # Scale by ratio
                scale_x = w_canon / w_orig
                scale_y = h_canon / h_orig
                transformed = orig_coords.copy()
                transformed[:, 0] = orig_coords[:, 0] * scale_x
                transformed[:, 1] = orig_coords[:, 1] * scale_y

                # If there is a center crop, subtract the offset as well
                if crop_size:
                    cw, ch = crop_size
                    offset_x = (w_canon - cw) / 2.0
                    offset_y = (h_canon - ch) / 2.0
                    transformed[:, 0] -= offset_x
                    transformed[:, 1] -= offset_y

                pixel_coords = np.round(transformed).astype(np.int64)
            else:
                # No resize info; use the original coordinates directly
                pixel_coords = orig_coords.astype(np.int64)

            depth_path_rel = record.get("depth_path")
            if depth_path_rel and cur_depth_root:
                depth_abs_path = os.path.join(cur_depth_root, depth_path_rel)
                gt_depth_map = _load_depth(
                    depth_abs_path,
                    depth_scale=record.get("depth_scale", 1000.0),
                    min_depth=min_depth,
                    max_depth=max_depth,
                    canonical_size=canonical_size,
                    crop_size=crop_size,
                    depth_format=record.get("depth_format"),
                )
                if gt_depth_map is not None:
                    # On the resized+cropped depth map, sample GT at the transformed coordinates
                    map_h, map_w = gt_depth_map.shape
                    gt_from_map = []
                    for k in range(len(pixel_coords)):
                        x, y = int(pixel_coords[k][0]), int(pixel_coords[k][1])
                        if 0 <= x < map_w and 0 <= y < map_h:
                            val = gt_depth_map[y, x]
                            if val > 0:
                                gt_from_map.append(val)
                            else:
                                # Point is 0 on the resized depth map (sparse depth lost during resize);
                                # fall back to the original depth value from jsonl.
                                gt_from_map.append(gt_depths_jsonl[k])
                        else:
                            gt_from_map.append(0.0)  # mark out-of-bounds points as invalid
                    gt_depths = np.array(gt_from_map, dtype=np.float32)

            result["pixel_coords"] = pixel_coords           # transformed coordinates (N, 2)
            result["pixel_coords_orig"] = orig_coords.astype(np.int64)  # original coordinates, for debugging
            result["gt_depths"] = gt_depths                 # GT sampled from the resized depth map
            result["gt_depths_jsonl"] = gt_depths_jsonl     # original jsonl GT, for comparison
            result["depth_type"] = record.get("depth_type", "z_depth")
            return result

        # ===== Legacy format: full depth-map GT =====
        depth_path_rel = record.get("depth_path")
        if depth_path_rel and cur_depth_root:
            depth_abs_path = os.path.join(cur_depth_root, depth_path_rel)

            mask_valid_path = None
            mask_valid_rel = record.get("mask_valid_path")
            if mask_valid_rel and cur_depth_root:
                mask_valid_path = os.path.join(cur_depth_root, mask_valid_rel)

            mask_transp_path = None
            mask_transp_rel = record.get("mask_transp_path")
            if mask_transp_rel and cur_depth_root:
                mask_transp_path = os.path.join(cur_depth_root, mask_transp_rel)

            depth_m = _load_depth(
                depth_abs_path,
                depth_scale=record.get("depth_scale", 1000.0),
                min_depth=min_depth,
                max_depth=max_depth,
                canonical_size=canonical_size,
                crop_size=crop_size,
                mask_valid_path=mask_valid_path,
                mask_transp_path=mask_transp_path,
                depth_format=record.get("depth_format"),
            )
            if depth_m is not None:
                result["pixel_depth_labels"] = depth_m

        return result
