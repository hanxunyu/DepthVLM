"""Merge per-shard evaluation jsonl files and report aggregated metrics.

Inputs:
    --result_dir   directory containing test_eval_results_shard{i}.jsonl
    --num_shards   number of shard files to read

Outputs:
    <result_dir>/test_eval_results_merged.jsonl   concatenation of all shards
    Console:
        Total samples: <N>
        final delta_1 = <X.XXXXXX>
"""
import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Merge per-shard eval result jsonls.")
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    args = parser.parse_args()

    shard_files = []
    for i in range(args.num_shards):
        p = os.path.join(args.result_dir, f"test_eval_results_shard{i}.jsonl")
        if os.path.exists(p):
            shard_files.append(p)
        else:
            print(f"WARNING: missing shard file {p}")

    if not shard_files:
        # Fall back to glob in case shard ids differ from {0..N-1}
        shard_files = sorted(glob.glob(os.path.join(args.result_dir, "test_eval_results_shard*.jsonl")))

    merged_records = []
    for sf in shard_files:
        with open(sf, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                merged_records.append(json.loads(line))

    merged_path = os.path.join(args.result_dir, "test_eval_results_merged.jsonl")
    with open(merged_path, "w") as f:
        for r in merged_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    deltas = [r["delta1"] for r in merged_records if isinstance(r.get("delta1"), (int, float))]
    n = len(deltas)
    avg = sum(deltas) / n if n > 0 else 0.0

    print(f"Merged {len(shard_files)} shard(s) -> {merged_path}")
    print(f"Total samples: {n}")
    print(f"final delta_1 = {avg:.6f}")


if __name__ == "__main__":
    main()
