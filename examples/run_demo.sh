#!/bin/bash
# Run DepthVLM example inference on the provided JSONL annotations.
#
# Usage (from project root):
#   bash examples/run_demo.sh                # default (GPU 0)
#   bash examples/run_demo.sh --gpus 0,1     # specify GPUs

set -e

# ---------- Configuration ----------
MODEL_PATH=JonnyYu828/DepthVLM-4B
RGB_ROOT="examples/rgb"
DEPTH_ROOT="examples/depth"
ANNOTATIONS_JSONL="examples/examples.jsonl"
OUTPUT_DIR="examples/output"

# ---------- Parse arguments ----------
GPUS="0"
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus) GPUS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

export CUDA_VISIBLE_DEVICES="${GPUS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ---------- Run ----------
python -u examples/run_demo.py \
    --model_path        "$MODEL_PATH" \
    --annotations_jsonl "$ANNOTATIONS_JSONL" \
    --rgb_root          "$RGB_ROOT" \
    --depth_root        "$DEPTH_ROOT" \
    --output_dir        "$OUTPUT_DIR"
