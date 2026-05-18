#!/bin/bash

# Always run from the project root and make local packages (model/, utils/, ...)
# importable regardless of where this script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ========== Configuration ==========

model_path=JonnyYu828/DepthVLM-4B

# Load dataset path configuration
source configs/eval_datasets.conf

# ===== Datasets to evaluate (space-separated) =====
# Names must match the variable prefix (uppercase) in eval_datasets.conf
EVAL_DATASETS="ARGOVERSE2 WAYMO DDAD NUSCENES SCANNETPP ETH3D SUNRGBD IBIMS1 NYUV2"


# GENERATE_TEXT=false # Stage 1 -> false
GENERATE_TEXT=true # Stage 2 -> true

SAVE_DEPTH_MAPS=false
max_save_depth_maps=20
BSZ=24

# ===== Specify GPUs to use =====
# Option 1: edit the variable below, e.g. GPUS="0,1,2,3" or GPUS="2,5,7"
# Option 2: pass at runtime, e.g. bash eval.sh --gpus 0,1,2,3
# Option 3: environment variable, e.g. CUDA_VISIBLE_DEVICES=0,1,2,3 bash eval.sh
# Leave empty to auto-use all available GPUs
GPUS=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Determine GPU list
if [ -n "$GPUS" ]; then
    # Use the specified GPUs
    IFS=',' read -ra GPU_LIST <<< "$GPUS"
elif [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    # Auto-detect all GPUs
    NUM_ALL=$(nvidia-smi -L | wc -l)
    GPU_LIST=()
    for ((i=0; i<NUM_ALL; i++)); do
        GPU_LIST+=($i)
    done
fi

NUM_GPUS=${#GPU_LIST[@]}
echo "Using ${NUM_GPUS} GPUs: ${GPU_LIST[*]}"

# ========== Output directory ==========
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_OUTPUT="output_log/eval_all_${TIMESTAMP}"
mkdir -p "${BASE_OUTPUT}"
echo "All outputs -> ${BASE_OUTPUT}"

# ========== Signal handler ==========
cleanup() {
    echo ""
    echo "Caught Ctrl+C! Killing all processes..."
    pkill -f "eval-debug.py.*--output_dir ${BASE_OUTPUT}" 2>/dev/null
    exit 1
}
trap cleanup SIGINT SIGTERM

# ========== Summary file ==========
SUMMARY_FILE="${BASE_OUTPUT}/summary.txt"
cat > "${SUMMARY_FILE}" <<EOF
========== Evaluation Summary ==========
Model: ${model_path}
Time: $(date)
Datasets: ${EVAL_DATASETS}

$(printf "%-20s %10s %10s\n" "Dataset" "delta1" "N_samples")
$(printf "%-20s %10s %10s\n" "-------" "------" "---------")
EOF

# ========== Evaluate per dataset ==========
for ds in ${EVAL_DATASETS}; do
    jsonl_var="${ds}_JSONL"
    image_var="${ds}_IMAGE"
    depth_var="${ds}_DEPTH"
    samples_var="${ds}_SAMPLES"
    json_path="${!jsonl_var}"
    image_folder="${!image_var}"
    depth_folder="${!depth_var}"
    ds_samples="${!samples_var:-0}"

    # Use lowercase as display name
    dataset_name=$(echo "${ds}" | tr '[:upper:]' '[:lower:]')

    if [ -z "${json_path}" ]; then
        echo "WARNING: ${jsonl_var} not defined in conf, skipping ${ds}"
        continue
    fi
    if [ ! -f "${json_path}" ]; then
        echo "WARNING: ${json_path} not found, skipping ${dataset_name}"
        continue
    fi

    OUTPUT_DIR="${BASE_OUTPUT}/${dataset_name}"
    mkdir -p "${OUTPUT_DIR}"

    echo ""
    echo "============================================================"
    echo "Evaluating: ${dataset_name}"
    echo "  jsonl: ${json_path}"
    echo "  image: ${image_folder}"
    echo "  depth: ${depth_folder}"
    echo "  samples: ${ds_samples} (0=all)"
    echo "============================================================"

    # Multi-GPU parallel inference
    pids=()
    for shard_id in $(seq 0 $((NUM_GPUS - 1))); do
        GPU_ID=${GPU_LIST[$shard_id]}
        CUDA_VISIBLE_DEVICES=${GPU_ID} python -u eval/eval.py \
            --model_path $model_path \
            --image_folder "$image_folder" \
            --json_path "$json_path" \
            --depth_root "$depth_folder" \
            $([ "$GENERATE_TEXT" = "true" ] && echo "--generate_text") \
            $([ "$SAVE_DEPTH_MAPS" = "true" ] && echo "--save_depth_maps") \
            --max_save_depth_maps $max_save_depth_maps \
            --bsz $BSZ \
            --samples_to_eval $ds_samples \
            --num_shards $NUM_GPUS \
            --shard_id $shard_id \
            --output_dir "${OUTPUT_DIR}" \
            2>&1 | tee "${OUTPUT_DIR}/eval_shard_${shard_id}.log" | sed "s/^/[${dataset_name}:${shard_id}] /" &
        pids+=($!)
    done

    # Wait for the current dataset to finish
    failed=0
    for i in "${!pids[@]}"; do
        wait ${pids[$i]}
        status=$?
        if [ $status -ne 0 ]; then
            echo "[${dataset_name}:Shard ${i}] FAILED (exit ${status})"
            failed=$((failed + 1))
        fi
    done

    if [ $failed -gt 0 ]; then
        echo "WARNING: ${dataset_name} had ${failed} failed shard(s)"
    fi

    # Merge results
    echo "Merging results for ${dataset_name}..."
    merge_output=$(python eval/merge_eval_results.py \
        --result_dir "${OUTPUT_DIR}" \
        --num_shards $NUM_GPUS 2>&1)
    echo "${merge_output}"

    # Extract delta1 and write into the summary
    delta1=$(echo "${merge_output}" | grep -oP 'delta_1.*?=\s*\K[0-9.]+' | tail -1 || echo "N/A")
    n_samples=$(echo "${merge_output}" | grep -oP 'Total samples:\s*\K[0-9]+' || echo "N/A")
    printf "%-20s %10s %10s\n" "${dataset_name}" "${delta1}" "${n_samples}" >> "${SUMMARY_FILE}"

    echo "[${dataset_name}] delta1=${delta1}, n=${n_samples}"
done

# ========== Final summary ==========
echo ""
echo "============================================================"
echo "ALL EVALUATIONS COMPLETE"
echo "============================================================"
cat "${SUMMARY_FILE}"
echo ""
echo "Full results: ${BASE_OUTPUT}/"
