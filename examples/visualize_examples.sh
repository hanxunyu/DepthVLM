#!/bin/bash
# Run DepthVLM on images in demo_images/ and save:
#   - depth maps  (left: GT | right: predicted, side-by-side colored PNG)
#   - point clouds (.ply, camera frame, RGB-colored)
set -e

# Always run from the project root and make local packages importable
# regardless of where this script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ========== Configuration ==========
MODEL_PATH=JonnyYu828/DepthVLM-4B   # HuggingFace repo or a local checkpoint dir
IMAGE_DIR=demo_images               # folder with input RGBs + a .jsonl annotation file
OUTPUT_DIR=demo_outputs

# ========== GPU selection ==========
GPUS="0"
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus) GPUS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
export CUDA_VISIBLE_DEVICES="$GPUS"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

python -u predict_from_list.py \
    --model_path "$MODEL_PATH" \
    --image_dir  "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR"
