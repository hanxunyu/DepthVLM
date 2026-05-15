#!/usr/bin/env python
"""
Uniformly sample a specified number of entries from a JSONL file,
balanced across (scene, camera) groups.

Supports --depth_root: when enabled, uses a "sample-then-validate" strategy.
Pass 1 performs a lightweight grouping scan without depth validation; after
sampling, each selected entry is validated. Invalid samples are replaced by
resampling from the remaining candidates in the same group.

Usage:
    python sample_images_ddad.py \
        --input_jsonl annotations/ddad/ddad_pixel_depth_val.jsonl \
        --total_samples 1000 \
        --output_jsonl annotations/ddad/test/ddad_pixel_depth_val_1000.jsonl \
        --depth_root /path/to/datasets
"""
import argparse
import os
import random
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_check import check_depth_valid, match_depth_range

MAX_RESAMPLE_ROUNDS = 5


def _allocate_budget(found_groups, group_sizes, total_samples):
    """Multi-round reallocation: distribute budget evenly; overflow from
    small groups is redistributed to larger ones."""
    group_alloc = {g: 0 for g in found_groups}
    remaining_budget = total_samples
    remaining_groups = set(found_groups)

    for _ in range(10):
        if not remaining_groups or remaining_budget <= 0:
            break
        sorted_remaining = sorted(remaining_groups)
        per_group = remaining_budget // len(sorted_remaining)
        remainder = remaining_budget % len(sorted_remaining)

        newly_saturated = set()
        budget_used = 0
        for i, g in enumerate(sorted_remaining):
            alloc = per_group + (1 if i < remainder else 0)
            available = group_sizes[g] - group_alloc[g]
            actual = min(alloc, available)
            group_alloc[g] += actual
            budget_used += actual
            if group_alloc[g] >= group_sizes[g]:
                newly_saturated.add(g)

        remaining_budget -= budget_used
        remaining_groups -= newly_saturated
        if not newly_saturated:
            break

    return group_alloc


def main():
    parser = argparse.ArgumentParser(description="Uniformly sample from jsonl, balanced across scene+camera.")
    parser.add_argument("--input_jsonl", type=str, required=True)
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

    print("Pass 1: Scanning (buffered read, no depth check)...")
    group_offsets = defaultdict(list)
    pattern_scene = re.compile(r'"scene"\s*:\s*"([^"]+)"')
    pattern_camera = re.compile(r'"camera"\s*:\s*"([^"]+)"')

    with open(args.input_jsonl, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            line_str = line.decode("utf-8", errors="replace")
            m_scene = pattern_scene.search(line_str)
            m_camera = pattern_camera.search(line_str)
            if m_scene and m_camera:
                group_offsets[(m_scene.group(1), m_camera.group(1))].append((offset, len(line)))

    total = sum(len(v) for v in group_offsets.values())
    scenes = set(k[0] for k in group_offsets)
    cameras = set(k[1] for k in group_offsets)
    print(f"Total {total} lines from {len(scenes)} scenes x {len(cameras)} cameras = {len(group_offsets)} groups")

    found_groups = sorted(group_offsets.keys())
    if not found_groups:
        print("ERROR: No groups found!")
        return

    group_sizes = {g: len(group_offsets[g]) for g in found_groups}
    group_alloc = _allocate_budget(found_groups, group_sizes, args.total_samples)
    print(f"Sampling {args.total_samples}: allocated total={sum(group_alloc.values())}")

    group_selected_set = {}
    for g in found_groups:
        n = group_alloc[g]
        sz = len(group_offsets[g])
        if n <= 0:
            group_selected_set[g] = set()
        elif n >= sz:
            group_selected_set[g] = set(range(sz))
        else:
            group_selected_set[g] = set(int(j * sz / n) for j in range(n))

    selected = []
    for g in found_groups:
        for idx in sorted(group_selected_set[g]):
            selected.append(group_offsets[g][idx])
    print(f"Selected {len(selected)} lines")

    if depth_cfg:
        min_d, max_d = depth_cfg["min_depth"], depth_cfg["max_depth"]
        print("Depth validation (lazy): checking selected samples...")

        group_valid = {g: [] for g in found_groups}
        invalid_count = 0

        with open(args.input_jsonl, "rb") as f:
            for g in found_groups:
                for idx in sorted(group_selected_set[g]):
                    off, length = group_offsets[g][idx]
                    f.seek(off)
                    data = f.read(length)
                    if check_depth_valid(data, args.depth_root, args.depth_root2, min_d, max_d):
                        group_valid[g].append((off, length))
                    else:
                        invalid_count += 1

        total_valid = sum(len(v) for v in group_valid.values())
        print(f"  Round 0: valid={total_valid}, invalid={invalid_count}")

        for round_no in range(1, MAX_RESAMPLE_ROUNDS + 1):
            deficit = sum(max(0, group_alloc[g] - len(group_valid[g])) for g in found_groups)
            if deficit == 0:
                break
            new_valid_count = 0
            new_invalid_count = 0
            with open(args.input_jsonl, "rb") as f:
                for g in found_groups:
                    need = group_alloc[g] - len(group_valid[g])
                    if need <= 0:
                        continue
                    pool = [i for i in range(len(group_offsets[g]))
                            if i not in group_selected_set[g]]
                    random.shuffle(pool)
                    for idx in pool:
                        if len(group_valid[g]) >= group_alloc[g]:
                            break
                        group_selected_set[g].add(idx)
                        off, length = group_offsets[g][idx]
                        f.seek(off)
                        data = f.read(length)
                        if check_depth_valid(data, args.depth_root, args.depth_root2, min_d, max_d):
                            group_valid[g].append((off, length))
                            new_valid_count += 1
                        else:
                            new_invalid_count += 1
                            invalid_count += 1

            total_valid = sum(len(v) for v in group_valid.values())
            print(f"  Round {round_no}: +{new_valid_count} valid, +{new_invalid_count} invalid, "
                  f"total valid={total_valid}")
            if new_valid_count == 0:
                break

        print(f"  Depth validation total: checked={total_valid + invalid_count}, "
              f"valid={total_valid}, invalid={invalid_count}")
        selected = []
        for g in found_groups:
            selected.extend(group_valid[g])

    print("Pass 2: Reading selected lines (seek)...")
    sampled = []
    with open(args.input_jsonl, "rb") as f:
        for offset, length in selected:
            f.seek(offset)
            sampled.append(f.read(length))

    random.shuffle(sampled)
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "wb") as f:
        for line in sampled:
            f.write(line)

    group_counts = defaultdict(int)
    for line in sampled:
        line_str = line.decode("utf-8", errors="replace")
        m_scene = pattern_scene.search(line_str)
        m_camera = pattern_camera.search(line_str)
        if m_scene and m_camera:
            group_counts[(m_scene.group(1), m_camera.group(1))] += 1

    scene_counts = defaultdict(int)
    camera_counts = defaultdict(int)
    for (s, c), cnt in group_counts.items():
        scene_counts[s] += cnt
        camera_counts[c] += cnt

    actual = len(sampled)
    final_path = args.output_jsonl
    if actual != args.total_samples:
        new_path = args.output_jsonl.replace(
            f"_{args.total_samples}.jsonl", f"_{actual}.jsonl")
        if new_path != args.output_jsonl:
            os.rename(args.output_jsonl, new_path)
            final_path = new_path

    counts = sorted(group_counts.values())
    print(f"\nOutput: {actual} samples from {len(scene_counts)} scenes x {len(camera_counts)} cameras")
    if counts:
        print(f"  Per-group: min={counts[0]}, max={counts[-1]}, mean={sum(counts)/len(counts):.1f}")
        print(f"  Per-camera: {dict(sorted(camera_counts.items()))}")
    print(f"  Saved to: {final_path}")


if __name__ == "__main__":
    main()
