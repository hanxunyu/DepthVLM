#!/bin/bash
# Run DepthVLM on images in demo_images/ and save:
#   - predicted depth maps (.npy)
#   - colored point clouds (.ply, camera frame)
set -e

MODEL_PATH=JonnyYu828/DepthVLM-8B
IMAGE_DIR=demo_images/
OUTPUT_DIR=demo_outputs/

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
