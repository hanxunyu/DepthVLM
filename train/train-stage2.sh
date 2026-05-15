#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

############################
# Multi-node distributed training arguments
############################

# Usage:
#   Single node:  bash train/train-stage2.sh
#   Multi node:   bash train/train-stage2.sh <NNODES> <NPROC_PER_NODE> <MASTER_ADDR> <MASTER_PORT> <NODE_RANK>
#
# Example: 2 nodes, 8 GPUs per node
#   export RUN_TIMESTAMP=20260515_1445
#   Node 0: RUN_TIMESTAMP=$RUN_TIMESTAMP bash train/train-stage2.sh <NNODES> <NPROC_PER_NODE> <MASTER_ADDR> <MASTER_PORT> <NODE_RANK>

NNODES=${1:-1}
NPROC_PER_NODE=${2:-8}
MASTER_ADDR=${3:-127.0.0.1}
MASTER_PORT=${4:-29500}
NODE_RANK=${5:-0}

echo "[dist] NNODES=$NNODES  NPROC=$NPROC_PER_NODE  MASTER=$MASTER_ADDR:$MASTER_PORT  RANK=$NODE_RANK"

export DEPTHLM_DEPTH_HEAD_TYPE=dpt 

if [ "$NNODES" -gt 1 ]; then
    NCCL_IF=${NCCL_IFNAME_OVERRIDE:-bond1}

    # Bind the control channel to the physical NIC to avoid the k8s overlay
    export NCCL_SOCKET_IFNAME="$NCCL_IF"
    export GLOO_SOCKET_IFNAME="$NCCL_IF"
    export TP_SOCKET_IFNAME="$NCCL_IF"

    # Keep IB/RoCE by default; disable only when explicitly requested
    if [ "${NCCL_DISABLE_IB:-0}" = "1" ]; then
        export NCCL_IB_DISABLE=1
        # Falling back to TCP after disabling IB; enable multi-socket parallelism to compensate bandwidth loss
        export NCCL_SOCKET_NTHREADS=${NCCL_SOCKET_NTHREADS:-8}
        export NCCL_NSOCKS_PERTHREAD=${NCCL_NSOCKS_PERTHREAD:-8}
        export NCCL_BUFFSIZE=${NCCL_BUFFSIZE:-8388608}
    fi

    # --- Bootstrap timing hardening ---
    # Avoid Connection refused on port 48827 caused by slow model loading on rank 0.
    # NCCL bootstrap defaults to ~2 minutes; raise to 10 minutes here.
    export NCCL_BOOTSTRAP_TIMEOUT=${NCCL_BOOTSTRAP_TIMEOUT:-600}

    # --- Watchdog / Heartbeat timeouts ---
    # FSDP broadcasts the whole model to all ranks on the first iteration, which may take >10 min.
    # Turn off the heartbeat monitor to prevent misjudged hangs from killing the process.
    export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-0}
    # As a safety net, set heartbeat timeout to 60 minutes, enough for the first FSDP sync
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}
    # Watchdog timeout for a single NCCL collective (default 30 min, explicitly raised here)
    export TORCH_NCCL_BLOCKING_WAIT=0
    export NCCL_TIMEOUT=${NCCL_TIMEOUT:-3600}

    # Slave node startup wait: avoid slaves reaching NCCL init before master and hitting "Connection refused".
    # Set NODE_START_DELAY=0 to disable (e.g., when you can guarantee strict ordered startup externally).
    NODE_START_DELAY=${NODE_START_DELAY:-60}
    if [ "$NODE_RANK" != "0" ] && [ "$NODE_START_DELAY" -gt 0 ]; then
        echo "[wait] slave node_rank=$NODE_RANK sleeping ${NODE_START_DELAY}s for master to be ready ..."
        sleep "$NODE_START_DELAY"
    fi
fi

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "[nccl] SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<unset>}  IB_DISABLE=${NCCL_IB_DISABLE:-0}  BOOTSTRAP_TIMEOUT=${NCCL_BOOTSTRAP_TIMEOUT:-default}  HEARTBEAT=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-default}s  MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-default}  DEBUG=$NCCL_DEBUG"

############################
# CUDA memory allocator tuning (mandatory for 8B unfrozen-LLM scenario)
############################
# Use expandable segments to avoid fragmentation; cap single-split size to reduce tail fragmentation
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}
# Limit cuBLAS workspace to prevent the first cublasCreate from allocating too much and hitting CUBLAS_STATUS_ALLOC_FAILED
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}

echo "[mem] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF  CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"

############################
# Training hyperparameters
############################
GLOBAL_BATCH_SIZE=640
LOCAL_BATCH_SIZE=8       # 8B unfrozen-LLM scenario; keep 8 to leave headroom for optimizer state memory

TOTAL_GPUS=$(( NNODES * NPROC_PER_NODE ))
GLOBAL_STEP_BATCH=$(( TOTAL_GPUS * LOCAL_BATCH_SIZE ))
if (( GLOBAL_BATCH_SIZE % GLOBAL_STEP_BATCH == 0 )); then
    GRAD_ACC_STEPS=$(( GLOBAL_BATCH_SIZE / GLOBAL_STEP_BATCH ))
else
    GRAD_ACC_STEPS=$(( (GLOBAL_BATCH_SIZE + GLOBAL_STEP_BATCH - 1) / GLOBAL_STEP_BATCH ))
fi
GRAD_ACC_STEPS=$(( GRAD_ACC_STEPS > 0 ? GRAD_ACC_STEPS : 1 ))

echo "[train] TOTAL_GPUS=$TOTAL_GPUS  GLOBAL_STEP_BATCH=$GLOBAL_STEP_BATCH  GRAD_ACC=$GRAD_ACC_STEPS"

############################
# Model & output
############################
# TODO: replace with the stage 1 checkpoint path
model_path=outputs/DepthVLM-4b-stage1_${TIMESTAMP}/
TIMESTAMP=${RUN_TIMESTAMP:-$(date +"%Y%m%d_%H%M%S")}
output_path=outputs/DepthVLM-4b-stage2_${TIMESTAMP}/
echo "[output] ${output_path}"

############################
# Data configuration
############################
source configs/train_datasets.conf

TRAIN_DATASETS="ARGOVERSE2 WAYMO DDAD NUSCENES SCANNETPP HM3D TASKONOMY MATTERPORT3D"

echo "Building train args for: ${TRAIN_DATASETS}"
build_train_args

############################
# Launch training
############################
python -m torch.distributed.run \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$NODE_RANK \
    train/train.py \
--model_name_or_path $model_path \
--image_folder "$IMAGE_FOLDER" \
--dataset_name "$DATASET_NAME" \
--depth_root "$DEPTH_ROOT" \
--freeze_vision \
--with_text_reply \
--depth_loss_weight 1.0 \
--max_seq_length 4096 \
--learning_rate 2e-5 \
--lr_scheduler_type cosine \
--per_device_train_batch_size $LOCAL_BATCH_SIZE \
--gradient_accumulation_steps $GRAD_ACC_STEPS \
--warmup_ratio 0.05 \
--max_grad_norm 1.0 \
--logging_steps 50 \
--report_to tensorboard \
--gradient_checkpointing true \
--attn_implementation "flash_attention_2" \
--num_train_epochs 1 \
--log_level info \
--logging_strategy steps \
--output_dir $output_path \
--save_steps 10000 \
--save_strategy steps \
--save_total_limit 3 \
--eval_strategy no \
--torch_dtype bfloat16 \
--seed 42 \
--fsdp "full_shard auto_wrap" \
--fsdp_config '{"transformer_layer_cls_to_wrap": "Qwen3VLTextDecoderLayer", "limit_all_gathers": true}'