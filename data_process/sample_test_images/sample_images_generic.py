#!/usr/bin/env python
"""
Generic JSONL sampling script: randomly sample N records from any JSONL file
(no scene-based grouping). If the total line count <= N, all lines are kept.

Supports --depth_root: when enabled, uses a "sample-then-validate" strategy.
First samples at uniform intervals, then validates each selected entry's depth
map. Invalid samples are replaced by resampling from the remaining candidates.

Suitable for datasets without scene grouping, such as NYUv2, SUN RGB-D,
ETH3D, and iBims-1.

Usage:
    python sample_images_generic.py \
        --input_jsonl annotations/nyuv2/nyuv2_pixel_depth_test.jsonl \
        --total_samples 1000 \
        --output_jsonl annotations/nyuv2/test/nyuv2_pixel_depth_test_1000.jsonl \
        --depth_root /path/to/datasets
"""
import argparse
import os
import random
import sys

# Ensure depth_check in the same directory is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_check import check_depth_valid, match_depth_range

MAX_RESAMPLE_ROUNDS = 5


def main():
    parser = argparse.ArgumentParser(description="Randomly sample N lines from a JSONL file.")
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

    # Read all non-empty lines
    with open(args.input_jsonl, "r") as f:
        lines = [line for line in f if line.strip()]

    total = len(lines)
    print(f"Total lines: {total}")

    # Depth validation config
    depth_cfg = None
    if args.depth_root:
        depth_cfg = match_depth_range(args.input_jsonl)
        print(f"Depth validation enabled (lazy): depth_root={args.depth_root}")
        print(f"  depth range: [{depth_cfg['min_depth']}, {depth_cfg['max_depth']}]")

    if len(lines) <= args.total_samples:
        print(f"Total ({len(lines)}) <= requested ({args.total_samples}), using all lines.")
        if depth_cfg:
            # Total <= requested: validate all lines
            valid_lines = []
            invalid_count = 0
            for line in lines:
                if check_depth_valid(line, args.depth_root, args.depth_root2,
                                     depth_cfg["min_depth"], depth_cfg["max_depth"]):
                    valid_lines.append(line)
                else:
                    invalid_count += 1
            print(f"  Depth valid: {len(valid_lines)}, invalid (removed): {invalid_count}")
            sampled = valid_lines
        else:
            sampled = lines
    else:
        # Uniform-interval sampling (reproducible and evenly distributed)
        n_lines = len(lines)
        indices = [int(i * n_lines / args.total_samples) for i in range(args.total_samples)]
        sampled_indices = set(indices)
        sampled = [lines[i] for i in indices]
        print(f"Sampled {len(sampled)} lines (uniform interval).")

        if depth_cfg:
            # Sample-then-validate + resampling
            min_d, max_d = depth_cfg["min_depth"], depth_cfg["max_depth"]
            valid = []
            invalid_count = 0
            for line in sampled:
                if check_depth_valid(line, args.depth_root, args.depth_root2, min_d, max_d):
                    valid.append(line)
                else:
                    invalid_count += 1
            print(f"  Round 0: valid={len(valid)}, invalid={invalid_count}")

            # Resample from unselected lines
            remaining_pool = [i for i in range(n_lines) if i not in sampled_indices]
            random.shuffle(remaining_pool)
            pool_cursor = 0

            for round_no in range(1, MAX_RESAMPLE_ROUNDS + 1):
                deficit = args.total_samples - len(valid)
                if deficit <= 0 or pool_cursor >= len(remaining_pool):
                    break

                new_invalid = 0
                candidates = remaining_pool[pool_cursor:pool_cursor + deficit * 2]
                pool_cursor += len(candidates)

                for idx in candidates:
                    if len(valid) >= args.total_samples:
                        break
                    line = lines[idx]
                    if check_depth_valid(line, args.depth_root, args.depth_root2, min_d, max_d):
                        valid.append(line)
                        sampled_indices.add(idx)
                    else:
                        new_invalid += 1
                        invalid_count += 1

                print(f"  Round {round_no}: +{len(valid) - (args.total_samples - deficit)} valid, "
                      f"+{new_invalid} invalid, total valid={len(valid)}")

            total_checked = len(valid) + invalid_count
            print(f"  Depth validation total: checked={total_checked}, "
                  f"valid={len(valid)}, invalid={invalid_count}")
            sampled = valid

    # Shuffle and write
    random.shuffle(sampled)
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w") as f:
        f.writelines(sampled)

    # Rename output file if actual count differs from requested
    actual = len(sampled)
    final_path = args.output_jsonl
    if actual != args.total_samples:
        new_path = args.output_jsonl.replace(
            f"_{args.total_samples}.jsonl", f"_{actual}.jsonl")
        if new_path != args.output_jsonl:
            os.rename(args.output_jsonl, new_path)
            final_path = new_path

    print(f"\nOutput: {actual} samples")
    print(f"  Saved to: {final_path}")


if __name__ == "__main__":
    main()
