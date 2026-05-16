"""Run DepthVLM inference on a JSONL annotation file and save outputs.

Outputs (under ``<output_dir>/<timestamp>/``):
    depth_map/<frame>.png   -- side-by-side RGB | GT depth | predicted depth
    point_cloud/<frame>.ply -- colored point cloud from the predicted depth

Usage:
    python examples/run_demo.py \
        --model_path <model_dir> \
        --annotations_jsonl examples/examples.jsonl \
        --rgb_root examples/rgb \
        --depth_root examples/depth
"""

import argparse
import json
import os
import sys
from datetime import datetime
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model import Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Valid depth range (meters) for ScanNet++ indoor scenes.
SCANNETPP_MIN_DEPTH = 0.001
SCANNETPP_MAX_DEPTH = 10.0

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_rgb(rgb_path: str, canonical_size=None) -> Image.Image:
    """Load an RGB image, optionally resizing to *canonical_size* ``(W, H)``."""
    img = Image.open(rgb_path).convert("RGB")
    if canonical_size is not None:
        tw, th = canonical_size
        if img.size != (tw, th):
            img = img.resize((tw, th), Image.BILINEAR)
    return img


def load_gt_depth(depth_root: str, depth_rel_path: str = None,
                  depth_scale: float = 1000.0):
    """Load a 16-bit GT depth map and convert to float32 meters.

    Returns:
        (depth, path): *depth* is ``None`` when the file is missing.
    """
    if not depth_rel_path or not depth_root:
        return None, None
    p = os.path.join(depth_root, depth_rel_path)
    if not os.path.exists(p):
        return None, p
    raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None, p
    if raw.ndim == 3:
        raw = raw[..., 0]
    depth = raw.astype(np.float32) / float(depth_scale)
    depth[(depth < SCANNETPP_MIN_DEPTH) | (depth > SCANNETPP_MAX_DEPTH)] = 0.0
    return depth, p


# ---------------------------------------------------------------------------
# Camera intrinsics & point-cloud back-projection
# ---------------------------------------------------------------------------


def build_intrinsic_from_jsonl(rec: dict):
    """Build a 3x3 intrinsic matrix from JSONL fields.

    Assumes square pixels (fx == fy) and a centered principal point.

    Returns:
        (K, rgb_w, rgb_h) or (None, None, None) when fields are missing.
    """
    fx = rec.get("original_fx")
    if fx is None or fx <= 0:
        return None, None, None
    rgb_size = rec.get("original_rgb_size")
    if not rgb_size or len(rgb_size) != 2:
        return None, None, None
    rgb_w, rgb_h = int(rgb_size[0]), int(rgb_size[1])
    K = np.array([
        [fx,  0.0, rgb_w / 2.0],
        [0.0, fx,  rgb_h / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K, rgb_w, rgb_h


def scale_intrinsic(K: np.ndarray, src_w: int, src_h: int,
                    dst_w: int, dst_h: int) -> np.ndarray:
    """Scale an intrinsic matrix from (src_w, src_h) to (dst_w, dst_h)."""
    sx, sy = dst_w / src_w, dst_h / src_h
    Ks = K.copy()
    Ks[0, 0] *= sx
    Ks[0, 2] *= sx
    Ks[1, 1] *= sy
    Ks[1, 2] *= sy
    return Ks


def depth_to_pointcloud(depth: np.ndarray, K: np.ndarray,
                        rgb_bgr: np.ndarray = None,
                        min_depth: float = SCANNETPP_MIN_DEPTH,
                        max_depth: float = SCANNETPP_MAX_DEPTH):
    """Back-project a depth map into a colored point cloud.

    The coordinate convention is X-right, Y-up, Z-forward so that common
    3-D viewers display the cloud upright from the camera's viewpoint.

    Returns:
        (points, colors): each shaped ``(N, 3)``.
    """
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    valid = (depth > min_depth) & (depth < max_depth) & np.isfinite(depth)
    if valid.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    ys, xs = np.nonzero(valid)
    zs = depth[ys, xs].astype(np.float32)
    Xs = (xs.astype(np.float32) - cx) * zs / fx
    Ys = -((ys.astype(np.float32) - cy) * zs / fy)
    points = np.stack([Xs, Ys, zs], axis=1)

    if rgb_bgr is not None:
        if rgb_bgr.shape[:2] != (H, W):
            rgb_bgr = cv2.resize(rgb_bgr, (W, H), interpolation=cv2.INTER_AREA)
        colors = rgb_bgr[ys, xs][:, ::-1].astype(np.uint8)
    else:
        colors = np.full((len(points), 3), 200, dtype=np.uint8)

    return points.astype(np.float32), colors


def save_ply_binary(path: str, points: np.ndarray, colors: np.ndarray):
    """Write a binary little-endian PLY file (XYZ float32 + RGB uint8)."""
    assert points.shape[0] == colors.shape[0]
    n = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ])
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = points[:, 0], points[:, 1], points[:, 2]
    arr["r"], arr["g"], arr["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as fp:
        fp.write(header)
        fp.write(arr.tobytes())


# ---------------------------------------------------------------------------
# Depth-map visualization
# ---------------------------------------------------------------------------


def percentile_range(depth: np.ndarray, p_lo: float = 2.0,
                     p_hi: float = 98.0):
    """Return (lo, hi) depth values at the given percentiles."""
    valid = depth[depth > 0]
    if valid.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(valid, p_lo))
    hi = float(np.percentile(valid, p_hi))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def depth_to_color(depth: np.ndarray, vmin: float, vmax: float,
                   invalid_color=(0, 0, 0)) -> np.ndarray:
    """Apply a Turbo colormap to a depth map. Invalid pixels (<=0) are black."""
    normed = np.clip((depth.astype(np.float64) - vmin) / max(vmax - vmin, 1e-6),
                     0.0, 1.0)
    colored = cv2.applyColorMap((normed * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[depth <= 0] = invalid_color
    return colored


def resize_depth(depth: np.ndarray, target_w: int, target_h: int,
                 interp=cv2.INTER_LINEAR) -> np.ndarray:
    """Resize a depth map while preserving invalid (<=0) regions."""
    if depth.shape[1] == target_w and depth.shape[0] == target_h:
        return depth.astype(np.float32)
    invalid = (depth <= 0).astype(np.uint8)
    resized = cv2.resize(depth.astype(np.float32), (target_w, target_h),
                         interpolation=interp)
    invalid_up = cv2.resize(invalid, (target_w, target_h),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    resized[invalid_up] = 0.0
    return resized


def _add_labels_bar(canvas_bgr: np.ndarray, labels: list,
                    n_cols: int) -> np.ndarray:
    """Prepend a black label bar on top of *canvas_bgr*."""
    from PIL import Image as PILImage, ImageDraw, ImageFont

    _, total_w = canvas_bgr.shape[:2]
    col_w = total_w // n_cols
    font_size = max(10, min(int(col_w * 0.07), 28))

    font = None
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
    ]:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue
    if font is None:
        try:
            font = ImageFont.truetype("DejaVuSans", font_size)
        except Exception:
            font = ImageFont.load_default()

    bar_h = font_size + max(8, font_size // 2)
    bar_img = PILImage.new("RGB", (total_w, bar_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(bar_img)
    for i, label in enumerate(labels):
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = col_w * i + (col_w - tw) // 2
        y = (bar_h - th) // 2 - bbox[1]
        draw.text((x, y), label, fill=(255, 255, 255), font=font)

    bar_np = np.array(bar_img)[:, :, ::-1]
    return np.concatenate([bar_np, canvas_bgr], axis=0)


def make_hconcat_rgb_gt_pred(rgb_path: str, gt_color_bgr: np.ndarray,
                              pred_color_bgr: np.ndarray) -> np.ndarray:
    """Horizontally concatenate RGB | GT depth | predicted depth with labels."""
    gt_h, gt_w = gt_color_bgr.shape[:2]
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        rgb_bgr = np.zeros((gt_h, gt_w, 3), dtype=np.uint8)
    else:
        rgb_bgr = cv2.resize(rgb_bgr, (gt_w, gt_h), interpolation=cv2.INTER_AREA)
    canvas = np.concatenate([rgb_bgr, gt_color_bgr, pred_color_bgr], axis=1)
    return _add_labels_bar(canvas,
                           ["RGB Image", "GT Depth Map", "Predicted Depth Map"], 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="DepthVLM demo: predict depth, save visualization & point cloud.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the DepthVLM model directory")
    parser.add_argument("--annotations_jsonl", type=str, required=True,
                        help="JSONL annotation file (one record per image)")
    parser.add_argument("--rgb_root", type=str, default=None,
                        help="Root directory for RGB images")
    parser.add_argument("--depth_root", type=str, default=None,
                        help="Root directory for GT depth maps (optional)")
    parser.add_argument("--output_dir", type=str, default="examples/output",
                        help="Output base directory")
    args = parser.parse_args()

    # Output directories
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(args.output_dir, ts)
    dir_depth_map = os.path.join(root, "depth_map")
    dir_point_cloud = os.path.join(root, "point_cloud")
    os.makedirs(dir_depth_map, exist_ok=True)
    os.makedirs(dir_point_cloud, exist_ok=True)
    print(f"Output root: {root}")

    # Load annotations
    records = []
    with open(args.annotations_jsonl, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} entries from {args.annotations_jsonl}")

    # Load model
    print(f"Loading model from: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    model.eval()

    n_ok, n_skip, n_pc_saved = 0, 0, 0

    with torch.no_grad():
        for rec in tqdm(records, desc="predict"):
            scene_id = rec.get("scene_id", "")
            img_rel = rec.get("image", "")
            frame_stem = os.path.splitext(os.path.basename(img_rel))[0]
            if not scene_id or not frame_stem:
                n_skip += 1
                continue

            rgb_path = (os.path.join(args.rgb_root, img_rel)
                        if args.rgb_root else img_rel)
            if not os.path.exists(rgb_path):
                print(f"[skip] RGB not found: {rgb_path}")
                n_skip += 1
                continue

            canonical_size = rec.get("canonical_size")
            if not canonical_size:
                print(f"[skip] no canonical_size for {scene_id}/{frame_stem}")
                n_skip += 1
                continue
            canonical_size = tuple(canonical_size)

            try:
                img = load_rgb(rgb_path, canonical_size)
            except Exception as e:
                print(f"[skip] failed to load RGB {rgb_path}: {e}")
                n_skip += 1
                continue

            prompt = rec.get("prompt", "")
            messages = [
                {"role": "system",
                 "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user",
                 "content": [
                     {"type": "image", "image": img},
                     {"type": "text",  "text": prompt},
                 ]},
            ]

            # Inference
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=[image_inputs],
                               padding=True, return_tensors="pt").to("cuda")

            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
            )
            if outputs.depth_pred is None:
                print(f"[warn] no depth prediction for {scene_id}/{frame_stem}")
                n_skip += 1
                continue

            pred = outputs.depth_pred[0].detach().cpu().float().numpy()
            while pred.ndim > 2:
                pred = pred[0]
            pred = pred.astype(np.float32)

            # Resize prediction to GT resolution
            gt_size = rec.get("original_depth_size")
            gt_w, gt_h = (int(gt_size[0]), int(gt_size[1])) \
                if gt_size and len(gt_size) == 2 else (256, 192)
            pred_resized = resize_depth(pred, gt_w, gt_h)

            # Depth-map visualization (requires GT)
            gt, _ = load_gt_depth(
                args.depth_root,
                depth_rel_path=rec.get("depth_path"),
                depth_scale=float(rec.get("depth_scale", 1000.0)),
            )
            if gt is not None:
                gt_color = depth_to_color(gt, *percentile_range(gt))
                pred_color = depth_to_color(pred_resized,
                                            *percentile_range(pred_resized))
                concat = make_hconcat_rgb_gt_pred(rgb_path, gt_color, pred_color)
                cv2.imwrite(os.path.join(dir_depth_map, f"{frame_stem}.png"),
                            concat)

            # Point cloud
            K_orig, K_rgb_w, K_rgb_h = build_intrinsic_from_jsonl(rec)
            if K_orig is not None:
                rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
                dH, dW = pred_resized.shape
                K_scaled = scale_intrinsic(K_orig, K_rgb_w, K_rgb_h, dW, dH)
                pts, cols = depth_to_pointcloud(pred_resized, K_scaled, rgb_bgr)
                if pts.shape[0] > 0:
                    save_ply_binary(
                        os.path.join(dir_point_cloud, f"{frame_stem}.ply"),
                        pts, cols)
                    n_pc_saved += 1

            n_ok += 1

    print(f"\nDone. ok={n_ok}  skipped={n_skip}")
    print(f"Point clouds saved: {n_pc_saved}")
    print(f"Outputs: {root}")


if __name__ == "__main__":
    main()
