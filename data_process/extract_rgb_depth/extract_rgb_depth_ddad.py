# Extract RGB images, sparse LiDAR depth maps, and camera intrinsics from DDAD dataset.
# Reference: utils/curate_ddad.py
# Usage: python extract_rgb_depth_ddad.py --ddad_json <...> --out_folder <...> --path_to_dgp_lib <...> [--num_workers 8]
#
# Output structure (scene-based, consistent with Waymo/Argoverse):
#   out_folder/{train,val}/rgb/{scene_name}/{timestamp}_{cam_name}.jpg
#   out_folder/{train,val}/depth/{scene_name}/{timestamp}_{cam_name}.png  (uint16, depth_m × 256)
#   out_folder/{train,val}/intrinsics/{scene_name}/{timestamp}_{cam_name}.json  ([fx, fy, cx, cy, W, H])
#   out_folder/{train,val}/index.jsonl  (per-frame metadata for traceability)
#
# DDAD uses Luminar-H2 LiDAR with max range ~250 m.
# depth_scale=256 → max representable depth = 65535/256 ≈ 255.9 m, which covers the full range.
"""
Example usage:

python extract_rgb_depth_ddad.py \
    --ddad_json /path/to/ddad/ddad_train_val/ddad.json \
    --out_folder /path/to/ddad \
    --path_to_dgp_lib /path/to/dgp \
    --num_workers 32

python extract_rgb_depth_ddad.py \
    --ddad_json /path/to/ddad/ddad_train_val/ddad.json \
    --out_folder /path/to/ddad \
    --path_to_dgp_lib /path/to/dgp \
    --num_workers 32 \
    --intrinsics_only

PS:
    Install the dgp library following the "How to Use" section in https://github.com/TRI-ML/DDAD
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Final, List

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Depth save scale: depth_m × 256 → uint16 PNG
# Max representable depth: 65535 / 256 ≈ 255.9 m (covers Luminar-H2 250 m range)
# Precision: 1/256 ≈ 3.9 mm (well below Luminar sub-1cm precision)
# Read back: depth_m = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
DEPTH_SAVE_SCALE: Final[float] = 256.0

DATUMS: Final[List[str]] = ["lidar"] + ["CAMERA_%02d" % idx for idx in [1, 5, 6, 7, 8, 9]]

CAMERA_DATUMS: Final[List[str]] = ["CAMERA_%02d" % idx for idx in [1, 5, 6, 7, 8, 9]]


def save_rgb_and_depth(
    rgb_image: Image.Image,
    depth_map: np.ndarray,
    intrinsics: np.ndarray,
    scene_name: str,
    timestamp: int,
    datum_name: str,
    split: str,
    out_folder: str,
    intrinsics_only: bool = False,
) -> bool:
    """Save one RGB + depth + intrinsics to disk. Safe to call from a thread."""
    try:
        # Construct relative path: {scene_name}/{timestamp}_{datum_name}
        relative_stem = f"{scene_name}/{timestamp}_{datum_name}"

        # --- Save intrinsics ---
        img_h, img_w = rgb_image.size[1], rgb_image.size[0]
        intrinsics_data = [
            float(intrinsics[0, 0]),  # fx
            float(intrinsics[1, 1]),  # fy
            float(intrinsics[0, 2]),  # cx
            float(intrinsics[1, 2]),  # cy
            img_w,                     # width
            img_h,                     # height
        ]
        intrinsics_out_path = os.path.join(out_folder, split, "intrinsics", relative_stem + ".json")
        os.makedirs(os.path.dirname(intrinsics_out_path), exist_ok=True)
        with open(intrinsics_out_path, "w") as f:
            json.dump(intrinsics_data, f)

        if intrinsics_only:
            return True

        # --- Save RGB image ---
        rgb_out_path = os.path.join(out_folder, split, "rgb", relative_stem + ".jpg")
        os.makedirs(os.path.dirname(rgb_out_path), exist_ok=True)
        rgb_image.save(rgb_out_path)

        # --- Save depth map as uint16 PNG (depth_m × DEPTH_SAVE_SCALE) ---
        depth_out_path = os.path.join(out_folder, split, "depth", relative_stem + ".png")
        os.makedirs(os.path.dirname(depth_out_path), exist_ok=True)
        depth_float = depth_map.astype(np.float32)
        depth_uint16 = (depth_float * DEPTH_SAVE_SCALE).clip(0, 65535).astype(np.uint16)
        Image.fromarray(depth_uint16).save(depth_out_path)

        return True
    except Exception as e:
        print(f"Error saving scene {scene_name} timestamp {timestamp} camera {datum_name}: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract RGB images and sparse LiDAR depth maps from DDAD."
    )
    parser.add_argument(
        "--ddad_json",
        type=str,
        required=True,
        help="Path to ddad.json (e.g. /path/to/ddad_train_val/ddad.json)",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        required=True,
        help="Output root folder ({split}/rgb/ and {split}/depth/ will be created inside)",
    )
    parser.add_argument(
        "--path_to_dgp_lib",
        type=str,
        required=True,
        help="Path to TRI DGP library (will be prepended to sys.path)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of I/O threads for saving files (default: 8)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val"],
        help="Splits to process (default: train val)",
    )
    parser.add_argument(
        "--intrinsics_only",
        action="store_true",
        help="Only save camera intrinsics (skip RGB and depth extraction)",
    )
    args = parser.parse_args()

    # Ensure dgp is importable
    if args.path_to_dgp_lib not in sys.path:
        sys.path.insert(0, args.path_to_dgp_lib)

    from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset

    grand_total = 0

    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"Loading DDAD split: {split}")
        print(f"{'='*60}")

        # Load dataset in main process (this internally uses multiprocessing.Pool)
        dataset = SynchronizedSceneDataset(
            args.ddad_json,
            split=split,
            datum_names=DATUMS,
            generate_depth_from_datum="lidar",
        )
        num_samples = len(dataset)
        num_scenes = len(dataset.scenes)
        print(f"Split [{split}]: {num_samples} samples across {num_scenes} scenes")

        if num_samples == 0:
            print(f"No samples found for split '{split}', skipping.")
            continue

        # Pre-create output directories
        os.makedirs(os.path.join(args.out_folder, split, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(args.out_folder, split, "depth"), exist_ok=True)
        os.makedirs(os.path.join(args.out_folder, split, "intrinsics"), exist_ok=True)

        split_count = 0
        index_entries = []

        # Use ThreadPoolExecutor for async I/O (saving files), but iterate dataset
        # sequentially in the main process to avoid nested multiprocessing issues.
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []

            for global_idx in tqdm(range(num_samples), desc=f"Reading ({split})"):
                try:
                    # dataset_item_index[global_idx] = (scene_idx, sample_idx_in_scene, datum_names)
                    scene_idx, sample_idx_in_scene, _ = dataset.dataset_item_index[global_idx]

                    # Get scene name from the scene container's directory basename
                    scene_container = dataset.scenes[scene_idx]
                    scene_name = os.path.basename(scene_container.directory)

                    # Get the sample to extract timestamp
                    sample_pb = scene_container.get_sample(sample_idx_in_scene)
                    timestamp = sample_pb.id.timestamp.ToMicroseconds()

                    sample = dataset[global_idx]
                except Exception as e:
                    print(f"Error loading sample {global_idx}: {e}")
                    continue

                # sample is a list of datum groups
                datum_list = sample[0] if isinstance(sample, (list, tuple)) else sample

                index_entry = {
                    "scene_name": scene_name,
                    "frame_idx": int(sample_idx_in_scene),
                    "timestamp": timestamp,
                    "split": split,
                    "scene_idx": int(scene_idx),
                    "cameras": [],
                    "source_dataset": "ddad",
                }

                for i in range(len(datum_list)):
                    datum = datum_list[i]
                    datum_name = datum.get("datum_name", "")

                    if "CAMERA" not in datum_name:
                        continue

                    try:
                        rgb_image: Image.Image = datum["rgb"]       # PIL.Image
                        depth_map: np.ndarray = datum["depth"]       # (H, W) float, meters
                        cam_intrinsics: np.ndarray = datum["intrinsics"]  # (3, 3) matrix

                        # Submit I/O to thread pool
                        fut = executor.submit(
                            save_rgb_and_depth,
                            rgb_image,
                            depth_map.copy(),  # copy to avoid dgp internal buffer reuse
                            cam_intrinsics.copy(),
                            scene_name,
                            timestamp,
                            datum_name,
                            split,
                            args.out_folder,
                            args.intrinsics_only,
                        )
                        futures.append(fut)
                        index_entry["cameras"].append(datum_name)

                    except Exception as e:
                        print(f"Error processing scene {scene_name} timestamp {timestamp} camera {datum_name}: {e}")
                        continue

                index_entries.append(index_entry)

            # Wait for all I/O to complete
            print(f"Waiting for {len(futures)} file writes to complete...")
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Writing ({split})"):
                if fut.result():
                    split_count += 1

        # Write index.jsonl for this split
        index_path = os.path.join(args.out_folder, split, "index.jsonl")
        entries = sorted(
            index_entries,
            key=lambda e: (e["scene_name"], e["frame_idx"]),
        )
        with open(index_path, "w") as f:
            for entry in entries:
                json.dump(entry, f)
                f.write("\n")
        print(f"Wrote {len(entries)} entries to {index_path}")

        grand_total += split_count
        print(f"Split [{split}] done. Frames saved: {split_count}")

        del dataset  # free memory before next split

    print(f"\nAll done. Grand total frames saved: {grand_total}")
