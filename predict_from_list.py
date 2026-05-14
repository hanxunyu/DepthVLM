"""DepthVLM demo: predict depth maps and point clouds for images in a folder.

Inputs:
  --image_dir   Folder containing input RGB images and ONE .jsonl file with
                per-image annotations (canonical_size, canonical_fx, prompt,
                depth_path, depth_scale).

Outputs (under --output_dir):
  depth_map/<stem>.png     side-by-side colored depth: GT (left) | Pred (right)
  pointcloud/<stem>.ply    camera-frame XYZ + RGB point cloud (binary PLY)
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

from qwen3_vl import Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

DEFAULT_PROMPT = (
    "Given this image, estimate the metric depth (in meters, rounded to two "
    "decimal places) for every pixel of the image."
)
MIN_DEPTH = 0.001
MAX_DEPTH = 100.0
COLORMAP = cv2.COLORMAP_TURBO


def load_annotations(jsonl_path: str) -> dict:
    """Build a {image_basename: record} index from the jsonl file."""
    idx = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = os.path.basename(r.get("image", ""))
            if key:
                idx[key] = r
    return idx


def find_annotation(annos: dict, img_path: str) -> dict | None:
    base = os.path.basename(img_path)
    if base in annos:
        return annos[base]
    stem = os.path.splitext(base)[0]
    return next((v for k, v in annos.items() if os.path.splitext(k)[0] == stem), None)


def load_gt_depth(image_dir: str, anno: dict) -> np.ndarray | None:
    """Load GT depth (meters, float32) using `depth_path` from the annotation."""
    depth_path = anno.get("depth_path")
    if not depth_path:
        return None
    depth_scale = float(anno.get("depth_scale", 1000.0))
    candidates = [
        depth_path if os.path.isabs(depth_path) else None,
        os.path.join(image_dir, depth_path),
        os.path.join(image_dir, os.path.basename(depth_path)),
    ]
    p = next((c for c in candidates if c and os.path.exists(c)), None)
    if p is None:
        return None
    raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3:
        raw = raw[..., 0]
    depth = raw.astype(np.float32) / depth_scale
    depth[~np.isfinite(depth)] = 0.0
    return depth


def percentile_range(depth: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0):
    valid = depth[depth > 0]
    if valid.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(valid, p_lo))
    hi = float(np.percentile(valid, p_hi))
    return lo, max(hi, lo + 1e-6)


def colorize_depth(depth: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Colorize a depth map; invalid pixels (<= 0) -> black."""
    valid = depth > 0
    normed = np.zeros_like(depth, dtype=np.float64)
    if vmax > vmin:
        normed = (depth.astype(np.float64) - vmin) / (vmax - vmin)
    normed = np.clip(normed, 0.0, 1.0)
    bgr = cv2.applyColorMap((normed * 255.0).astype(np.uint8), COLORMAP)
    bgr[~valid] = 0
    return bgr


def resize_depth_keep_invalid(depth: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    if depth.shape[1] == target_w and depth.shape[0] == target_h:
        return depth.astype(np.float32)
    invalid = (depth <= 0).astype(np.uint8)
    out = cv2.resize(depth.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    invalid_up = cv2.resize(invalid, (target_w, target_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    out[invalid_up] = 0.0
    return out


def save_side_by_side(gt: np.ndarray | None, pred: np.ndarray, save_path: str) -> None:
    """Save a side-by-side colorized depth comparison: GT (left) | Pred (right)."""
    h, w = pred.shape
    if gt is not None:
        gt_resized = resize_depth_keep_invalid(gt, w, h)
        vmin, vmax = percentile_range(gt_resized)
    else:
        gt_resized = np.zeros_like(pred)
        vmin, vmax = percentile_range(pred)
    gt_color = colorize_depth(gt_resized, vmin, vmax)
    pred_color = colorize_depth(pred, vmin, vmax)
    cv2.imwrite(save_path, np.concatenate([gt_color, pred_color], axis=1))


def depth_to_pointcloud(depth: np.ndarray, K: np.ndarray, rgb_bgr: np.ndarray | None):
    """Backproject a depth map to camera-frame XYZ points with optional RGB colors."""
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    valid = (depth > MIN_DEPTH) & (depth < MAX_DEPTH) & np.isfinite(depth)
    if not valid.any():
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    ys, xs = np.nonzero(valid)
    zs = depth[ys, xs].astype(np.float32)
    Xs = (xs.astype(np.float32) - cx) * zs / fx
    Ys = (ys.astype(np.float32) - cy) * zs / fy
    pts = np.stack([Xs, Ys, zs], axis=1).astype(np.float32)
    if rgb_bgr is not None:
        if rgb_bgr.shape[:2] != (H, W):
            rgb_bgr = cv2.resize(rgb_bgr, (W, H), interpolation=cv2.INTER_AREA)
        cols = rgb_bgr[ys, xs][:, ::-1].astype(np.uint8)
    else:
        cols = np.full((pts.shape[0], 3), 200, np.uint8)
    return pts, cols


def save_ply(path: str, pts: np.ndarray, cols: np.ndarray) -> None:
    """Write a binary little-endian PLY file (XYZ float32 + RGB uint8)."""
    n = pts.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("r", "u1"), ("g", "u1"), ("b", "u1")])
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr["r"], arr["g"], arr["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(arr.tobytes())


def main():
    parser = argparse.ArgumentParser(description="DepthVLM demo: depth map + point cloud.")
    parser.add_argument("--model_path", required=True, help="DepthVLM checkpoint dir.")
    parser.add_argument(
        "--image_dir",
        default="demo_images",
        help="Folder with RGB images and a .jsonl annotations file.",
    )
    parser.add_argument("--output_dir", default="demo_outputs")
    args = parser.parse_args()

    jsonls = sorted(glob.glob(os.path.join(args.image_dir, "*.jsonl")))
    if not jsonls:
        raise FileNotFoundError(f"No .jsonl found in {args.image_dir}")
    annos = load_annotations(jsonls[0])
    print(f"[anno] {len(annos)} entries from {jsonls[0]}")

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        images.extend(glob.glob(os.path.join(args.image_dir, ext)))
    images = sorted(set(images))
    if not images:
        raise FileNotFoundError(f"No images found in {args.image_dir}")
    print(f"[input] {len(images)} images")

    depth_dir = os.path.join(args.output_dir, "depth_map")
    pc_dir = os.path.join(args.output_dir, "pointcloud")
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pc_dir, exist_ok=True)

    print(f"[model] loading {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()

    with torch.no_grad():
        for img_path in tqdm(images, desc="predict"):
            anno = find_annotation(annos, img_path)
            if anno is None:
                print(f"[skip] no annotation for {img_path}")
                continue

            stem = os.path.splitext(os.path.basename(img_path))[0]
            canonical_size = anno.get("canonical_size")
            canonical_fx = float(anno.get("canonical_fx", 1000.0))

            img = Image.open(img_path).convert("RGB")
            if canonical_size and tuple(img.size) != tuple(canonical_size):
                img = img.resize(tuple(canonical_size), Image.BILINEAR)

            messages = [
                {"role": "system",
                 "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user",
                 "content": [
                     {"type": "image", "image": img},
                     {"type": "text", "text": anno.get("prompt", DEFAULT_PROMPT)},
                 ]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            imgs, _ = process_vision_info(messages)
            inputs = processor(
                text=[text], images=[imgs], padding=True, return_tensors="pt",
            ).to("cuda")

            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
            )
            preds = outputs.depth_pred
            if not preds:
                print(f"[warn] model returned no depth for {img_path}")
                continue

            pred = preds[0].detach().cpu().float().numpy()
            while pred.ndim > 2:
                pred = pred[0]
            pred = pred.astype(np.float32)

            gt = load_gt_depth(args.image_dir, anno)
            save_side_by_side(gt, pred, os.path.join(depth_dir, f"{stem}.png"))

            H, W = pred.shape
            ref_w, ref_h = canonical_size if canonical_size else (W, H)
            sx = W / float(ref_w)
            sy = H / float(ref_h)
            fx = canonical_fx * sx
            fy = canonical_fx * sy
            cx = (ref_w / 2.0) * sx
            cy = (ref_h / 2.0) * sy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

            rgb_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
            pts, cols = depth_to_pointcloud(pred, K, rgb_bgr)
            if pts.shape[0] > 0:
                save_ply(os.path.join(pc_dir, f"{stem}.ply"), pts, cols)

    print(f"[done] depth maps  -> {depth_dir}")
    print(f"[done] pointclouds -> {pc_dir}")


if __name__ == "__main__":
    main()
