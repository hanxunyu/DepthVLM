#!/usr/bin/env python
"""
SUN RGB-D sampling script:
  1. Excludes NYUv2 data (kv1/NYUdata), since NYUv2 is treated as a separate dataset.
  2. Stratified sampling by sensor/source (e.g. kv2/kinect2data, xtion/sun3ddata)
     to ensure all sensor types are represented.

Supports --depth_root: when enabled, uses a "sample-then-validate" strategy.
Pass 1 performs a lightweight grouping scan without depth validation; after
sampling, each selected entry is validated. Invalid samples are replaced by
resampling from the remaining candidates in the same group.

Usage:
    python sample_images_sunrgbd.py \
        --input_jsonl annotations/sun_rgbd/sunrgbd_pixel_depth_test.jsonl \
        --total_samples 1000 \
        --output_jsonl annotations/sun_rgbd/test_1000/sunrgbd_pixel_depth_test_1000.jsonl \
        --depth_root /path/to/datasets
"""
import argparse
import os
import re
import random
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
    parser = argparse.ArgumentParser(
        description="SUN RGB-D sampling: exclude NYUv2, stratified by sensor/source.")
    parser.add_argument("--input_jsonl", type=str, required=True,
                        help="Input JSONL file path")
    parser.add_argument("--total_samples", type=int, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude_nyu", action="store_true", default=True,
                        help="Exclude NYUv2 data (kv1/NYUdata), default: True")
    parser.add_argument("--no_exclude_nyu", dest="exclude_nyu", action="store_false",
                        help="Do not exclude NYUv2 data")
    parser.add_argument("--depth_root", type=str, default=None,
                        help="Depth map root directory. Enables depth validation (sample-then-validate).")
    parser.add_argument("--depth_root2", type=str, default=None,
                        help="Fallback depth root directory (tried when depth_root misses).")
    args = parser.parse_args()

    random.seed(args.seed)

    # Depth validation config
    depth_cfg = match_depth_range(args.input_jsonl) if args.depth_root else None
    if depth_cfg:
        print(f"Depth validation enabled (lazy): depth_root={args.depth_root}")
        print(f"  depth range: [{depth_cfg['min_depth']}, {depth_cfg['max_depth']}]")

    # ------ 1. Scan JSONL, group by sensor/source (no depth check) ------
    print(f"Scanning {args.input_jsonl} (no depth check)...")
    pat_sensor_source = re.compile(r'SUNRGBD/([^/]+/[^/]+)/')
    pat_image_path = re.compile(r'"image"\s*:\s*"([^"]+)"')

    group_offsets = defaultdict(list)
    total = 0
    excluded_nyu = 0

    with open(args.input_jsonl, "rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            line_str = line.decode("utf-8", errors="replace")
            if not line_str.strip():
                continue

            total += 1

            m_img = pat_image_path.search(line_str)
            if not m_img:
                continue
            img_path = m_img.group(1)

            m_ss = pat_sensor_source.search(img_path)
            if not m_ss:
                sensor_source = "unknown"
            else:
                sensor_source = m_ss.group(1)

            # Exclude NYUv2
            if args.exclude_nyu and "NYUdata" in sensor_source:
                excluded_nyu += 1
                continue

            group_offsets[sensor_source].append((offset, len(line)))

    kept = sum(len(v) for v in group_offsets.values())
    print(f"  Total lines: {total}")
    if args.exclude_nyu:
        print(f"  Excluded NYUv2: {excluded_nyu}")
    print(f"  Kept: {kept}")
    print(f"  Sensor/source distribution:")
    for ss, offsets in sorted(group_offsets.items(), key=lambda x: -len(x[1])):
        print(f"    {ss}: {len(offsets)}")

    # ------ 2. Stratified sampling with redistribution ------
    found_groups = sorted(group_offsets.keys())
    num_groups = len(found_groups)

    if num_groups == 0:
        print("ERROR: No valid groups found!")
        return

    print(f"\nSampling {args.total_samples} from {num_groups} sensor/source groups:")

    group_sizes = {g: len(group_offsets[g]) for g in found_groups}
    group_alloc = _allocate_budget(found_groups, group_sizes, args.total_samples)

    # Initial selection
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

    print(f"  Allocated per-group: min={min(group_alloc.values())}, "
          f"max={max(group_alloc.values())}, total={sum(group_alloc.values())}")
    print(f"  Selected: {len(selected)} lines")

    # ------ 3. Depth validation + resampling ------
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

    # ------ 4. Read selected lines ------
    sampled = []
    with open(args.input_jsonl, "rb") as f:
        for offset, length in selected:
            f.seek(offset)
            sampled.append(f.read(length))

    # Shuffle
    random.shuffle(sampled)

    # ------ 5. Write output ------
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "wb") as f:
        for line in sampled:
            f.write(line)

    # ------ 6. Statistics + rename ------
    actual = len(sampled)
    final_path = args.output_jsonl
    if actual != args.total_samples:
        new_path = args.output_jsonl.replace(
            f"_{args.total_samples}.jsonl", f"_{actual}.jsonl")
        if new_path != args.output_jsonl:
            os.rename(args.output_jsonl, new_path)
            final_path = new_path

    final_ss = defaultdict(int)
    for line in sampled:
        line_str = line.decode("utf-8", errors="replace")
        m_img = pat_image_path.search(line_str)
        if m_img:
            m_ss = pat_sensor_source.search(m_img.group(1))
            if m_ss:
                final_ss[m_ss.group(1)] += 1

    print(f"\nOutput: {actual} samples")
    print(f"  Sensor/source distribution:")
    for ss, cnt in sorted(final_ss.items(), key=lambda x: -x[1]):
        print(f"    {ss}: {cnt}")
    print(f"  Saved to: {final_path}")


if __name__ == "__main__":
    main()
