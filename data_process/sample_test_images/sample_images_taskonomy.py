#!/usr/bin/env python
"""
Uniformly sample a specified number of entries from a JSONL file,
ensuring all scenes in the split file are covered with equal allocation.
Supports --target_split to select train/val/test split.

Supports --depth_root: when enabled, uses a "sample-then-validate" strategy.
Pass 1 performs a lightweight grouping scan without depth validation; after
sampling, each selected entry is validated. Invalid samples are replaced by
resampling from the remaining candidates in the same scene.

Usage:
    python sample_images_taskonomy.py \
        --input_jsonl annotations/taskonomy/taskonomy_pixel_depth_test.jsonl \
        --split_file annotations/taskonomy/train_val_test_fullplus.csv \
        --target_split test \
        --total_samples 1000 \
        --output_jsonl annotations/taskonomy/test/taskonomy_pixel_depth_test_1000.jsonl \
        --depth_root /path/to/datasets
"""
import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_check import check_depth_valid, match_depth_range

MAX_RESAMPLE_ROUNDS = 5


def _allocate_budget(found_scenes, group_sizes, total_samples):
    """Multi-round reallocation: distribute budget evenly; overflow from
    small groups is redistributed to larger ones."""
    group_alloc = {s: 0 for s in found_scenes}
    remaining_budget = total_samples
    remaining_groups = set(found_scenes)

    for _ in range(10):
        if not remaining_groups or remaining_budget <= 0:
            break
        sorted_remaining = sorted(remaining_groups)
        per_scene = remaining_budget // len(sorted_remaining)
        remainder = remaining_budget % len(sorted_remaining)

        newly_saturated = set()
        budget_used = 0
        for i, s in enumerate(sorted_remaining):
            alloc = per_scene + (1 if i < remainder else 0)
            available = group_sizes[s] - group_alloc[s]
            actual = min(alloc, available)
            group_alloc[s] += actual
            budget_used += actual
            if group_alloc[s] >= group_sizes[s]:
                newly_saturated.add(s)

        remaining_budget -= budget_used
        remaining_groups -= newly_saturated
        if not newly_saturated:
            break

    return group_alloc


def main():
    parser = argparse.ArgumentParser(description="Uniformly sample from jsonl, balanced across scenes.")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--split_file", type=str, required=True)
    parser.add_argument("--target_split", type=str, default="test",
                        help="Which split column to filter: train/val/test (default: test). "
                             "The CSV has columns 'train', 'val', 'test' with value '1'.")
    parser.add_argument("--total_samples", type=int, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--depth_root", type=str, default=None,
                        help="Depth map root directory. Enables depth validation (sample-then-validate).")
    parser.add_argument("--depth_root2", type=str, default=None,
                        help="Fallback depth root directory (tried when depth_root misses).")
    args = parser.parse_args()

    random.seed(args.seed)

    depth_cfg = match_depth_range(args.input_jsonl) if args.depth_root else None
    if depth_cfg:
        print(f"Depth validation enabled (lazy): depth_root={args.depth_root}")
        print(f"  depth range: [{depth_cfg['min_depth']}, {depth_cfg['max_depth']}]")

    # Read split file (supports both txt and csv formats)
    split_scenes = set()
    if args.split_file.endswith(".csv"):
        import csv
        with open(args.split_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Taskonomy CSV: 'id' column + train/val/test columns (value "1")
                if row.get(args.target_split, "").strip() == "1":
                    split_scenes.add(row["id"].strip())
    else:
        with open(args.split_file, "r") as f:
            split_scenes = set(line.strip() for line in f if line.strip())
    print(f"Split scenes ({args.target_split}): {len(split_scenes)}")

    print("Pass 1: Scanning scene IDs (no depth check)...")
    scene_line_indices = defaultdict(list)
    all_lines = []
    pattern = re.compile(r'"scene_id"\s*:\s*"([^"]+)"')
    pattern2 = re.compile(r'"scene"\s*:\s*"([^"]+)"')

    with open(args.input_jsonl, "r") as f:
        for line_no, line in enumerate(f):
            all_lines.append(line)
            if not line.strip():
                continue
            m = pattern.search(line)
            if not m:
                m = pattern2.search(line)
            if m:
                scene_id = m.group(1)
                if scene_id in split_scenes:
                    scene_line_indices[scene_id].append(line_no)

    total_matched = sum(len(v) for v in scene_line_indices.values())
    print(f"Matched {total_matched} lines from {len(scene_line_indices)} scenes")

    missing = split_scenes - set(scene_line_indices.keys())
    if missing:
        print(f"WARNING: {len(missing)} scenes not found: {sorted(missing)[:5]}...")

    found_scenes = sorted(scene_line_indices.keys())
    if not found_scenes:
        print("ERROR: No matching scenes!")
        return

    group_sizes = {s: len(scene_line_indices[s]) for s in found_scenes}
    group_alloc = _allocate_budget(found_scenes, group_sizes, args.total_samples)
    print(f"Sampling {args.total_samples}: allocated total={sum(group_alloc.values())}")

    scene_selected_set = {}
    for s in found_scenes:
        indices = scene_line_indices[s]
        n = group_alloc[s]
        if n <= 0:
            scene_selected_set[s] = set()
        elif n >= len(indices):
            scene_selected_set[s] = set(range(len(indices)))
        else:
            scene_selected_set[s] = set(int(j * len(indices) / n) for j in range(n))

    selected_lines = set()
    for s in found_scenes:
        for idx in scene_selected_set[s]:
            selected_lines.add(scene_line_indices[s][idx])
    print(f"Selected {len(selected_lines)} lines")

    if depth_cfg:
        min_d, max_d = depth_cfg["min_depth"], depth_cfg["max_depth"]
        print("Depth validation (lazy): checking selected samples...")

        scene_valid = {s: set() for s in found_scenes}
        invalid_count = 0

        for s in found_scenes:
            for idx in sorted(scene_selected_set[s]):
                line_no = scene_line_indices[s][idx]
                if check_depth_valid(all_lines[line_no], args.depth_root, args.depth_root2, min_d, max_d):
                    scene_valid[s].add(line_no)
                else:
                    invalid_count += 1

        total_valid = sum(len(v) for v in scene_valid.values())
        print(f"  Round 0: valid={total_valid}, invalid={invalid_count}")

        for round_no in range(1, MAX_RESAMPLE_ROUNDS + 1):
            deficit = sum(max(0, group_alloc[s] - len(scene_valid[s])) for s in found_scenes)
            if deficit == 0:
                break
            new_valid_count = 0
            new_invalid_count = 0
            for s in found_scenes:
                need = group_alloc[s] - len(scene_valid[s])
                if need <= 0:
                    continue
                pool = [i for i in range(len(scene_line_indices[s]))
                        if i not in scene_selected_set[s]]
                random.shuffle(pool)
                for idx in pool:
                    if len(scene_valid[s]) >= group_alloc[s]:
                        break
                    scene_selected_set[s].add(idx)
                    line_no = scene_line_indices[s][idx]
                    if check_depth_valid(all_lines[line_no], args.depth_root, args.depth_root2, min_d, max_d):
                        scene_valid[s].add(line_no)
                        new_valid_count += 1
                    else:
                        new_invalid_count += 1
                        invalid_count += 1

            total_valid = sum(len(v) for v in scene_valid.values())
            print(f"  Round {round_no}: +{new_valid_count} valid, +{new_invalid_count} invalid, "
                  f"total valid={total_valid}")
            if new_valid_count == 0:
                break

        print(f"  Depth validation total: checked={total_valid + invalid_count}, "
              f"valid={total_valid}, invalid={invalid_count}")
        selected_lines = set()
        for s in found_scenes:
            selected_lines.update(scene_valid[s])

    print("Pass 2: Reading selected lines...")
    sampled = [all_lines[i] for i in sorted(selected_lines)]

    random.shuffle(sampled)
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w") as f:
        f.writelines(sampled)

    scene_counts = defaultdict(int)
    for line in sampled:
        m = pattern.search(line)
        if not m:
            m = pattern2.search(line)
        if m:
            scene_counts[m.group(1)] += 1

    actual = len(sampled)
    final_path = args.output_jsonl
    if actual != args.total_samples:
        new_path = args.output_jsonl.replace(
            f"_{args.total_samples}.jsonl", f"_{actual}.jsonl")
        if new_path != args.output_jsonl:
            os.rename(args.output_jsonl, new_path)
            final_path = new_path

    counts = sorted(scene_counts.values())
    print(f"\nOutput: {actual} samples from {len(scene_counts)} scenes")
    if counts:
        print(f"  Per-scene: min={counts[0]}, max={counts[-1]}, mean={sum(counts)/len(counts):.1f}")
    print(f"  Saved to: {final_path}")


if __name__ == "__main__":
    main()
