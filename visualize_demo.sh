#!/bin/bash
# Run DepthVLM on images in demo_images/ and save:
#   - predicted depth maps (.npy)
#   - colored point clouds (.ply, camera frame)
set -e

MODEL_PATH=/apdcephfs/share_300000800/datamultimodal/hanxun2_data/ft-v4-multinodes-2000k-unifocal-DepthLM-qwen3-vl-8b-pixel-DPT-stage1b_20260423_1445-dptv1
IMAGE_DIR=/apdcephfs_cq11/share_1603164/user/jonnyhxyu/DepthLM_Official/DepthVLM/demo_images
OUTPUT_DIR=demo_outputs

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
