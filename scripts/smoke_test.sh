#!/bin/bash
# One-click smoke test for DepthVLM (environment, depth heads, demo, eval, optional training).
#
# Usage (from anywhere):
#   bash scripts/smoke_test.sh                 # env + head unit test + demo + sparse/dense eval
#   bash scripts/smoke_test.sh --train         # also run tiny stage-1 -> stage-2 training on examples/
#   bash scripts/smoke_test.sh --gpus 0,1,2,3  # GPUs to use (default: all visible)
#
# Env overrides:
#   MODEL_PATH        DepthVLM checkpoint (HF id or local dir), default JonnyYu828/DepthVLM-4B
#   BASE_MODEL        base model for training, default Qwen/Qwen3-VL-4B-Instruct
#   HF_ENDPOINT       e.g. https://hf-mirror.com when Hugging Face is not reachable
#   DEPTHVLM_ATTN_IMPL  flash_attention_2 (default when flash-attn is installed) or sdpa
#
# Every step runs even if an earlier one fails; a PASS/FAIL summary is printed at the end.
# Logs and outputs go to smoke_logs/<timestamp>/.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

RUN_TRAIN=0
GPUS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --train) RUN_TRAIN=1; shift ;;
        --gpus)  GPUS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

MODEL_PATH=${MODEL_PATH:-JonnyYu828/DepthVLM-4B}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-VL-4B-Instruct}

if [ -n "$GPUS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
fi
NGPU=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
FIRST_GPU=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d, -f1)

TS=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="smoke_logs/${TS}"
mkdir -p "${LOG_DIR}"

declare -a STEP_NAMES=()
declare -a STEP_RESULTS=()

# run_step <name> <command...>: run a command, tee its log, record PASS/FAIL
run_step() {
    local name="$1"; shift
    local log="${LOG_DIR}/${name}.log"
    echo ""
    echo "================ [${name}] ================"
    local start=$SECONDS
    "$@" 2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    local secs=$(( SECONDS - start ))
    STEP_NAMES+=("${name}")
    if [ "$rc" -eq 0 ]; then
        STEP_RESULTS+=("PASS (${secs}s)")
    else
        STEP_RESULTS+=("FAIL rc=${rc} (${secs}s) -> ${log}")
    fi
    return "$rc"
}

############################
# 1. Environment check
############################
check_env() {
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || return 1
    python - <<'EOF'
import sys, importlib
print("python", sys.version.split()[0])
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA not available"
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name}  {p.total_memory/2**30:.0f} GiB  sm_{p.major}{p.minor}")
import transformers
print("transformers", transformers.__version__)
for m in ["trl", "accelerate", "qwen_vl_utils", "cv2", "plyfile", "math_verify"]:
    mod = importlib.import_module(m)
    print(m, getattr(mod, "__version__", "ok"))
try:
    import flash_attn
    print("flash_attn", flash_attn.__version__)
except ImportError:
    print("flash_attn NOT installed (will fall back to sdpa)")
import model  # repo model package; fails on transformers < 5.x
print("import model OK")
EOF
}
run_step 01_env check_env

if [ -z "${DEPTHVLM_ATTN_IMPL:-}" ]; then
    if python -c "import flash_attn" 2>/dev/null; then
        export DEPTHVLM_ATTN_IMPL=flash_attention_2
    else
        export DEPTHVLM_ATTN_IMPL=sdpa
    fi
fi
echo "[attn] DEPTHVLM_ATTN_IMPL=${DEPTHVLM_ATTN_IMPL}"

############################
# 2. Depth head unit test (random features, no checkpoint needed)
############################
test_heads() {
    python - <<'EOF'
import torch
from model.dpt_depth_head import DPTDepthHead
from model.mlp_depth_head import MLPDepthHead

dev = "cuda"
ph, pw, C = 12, 16, 2560
feats = [torch.randn(1, ph * pw, C, device=dev, dtype=torch.bfloat16) for _ in range(4)]
for name, head in [("dpt", DPTDepthHead(C)), ("mlp", MLPDepthHead(C))]:
    head = head.to(dev, torch.bfloat16)
    out = head(feats, ph, pw)
    assert out.shape == (1, ph * 32, pw * 32), (name, out.shape)
    assert (out >= 0).all() and torch.isfinite(out).all(), name
    out.float().mean().backward()
    print(f"{name}: out {tuple(out.shape)}  params {sum(p.numel() for p in head.parameters())/1e6:.1f}M  backward OK")
EOF
}
run_step 02_heads test_heads

############################
# 3. Download checkpoint
############################
download_model() {
    if [ -d "${MODEL_PATH}" ]; then
        echo "Using local checkpoint ${MODEL_PATH}"
        return 0
    fi
    python -c "from huggingface_hub import snapshot_download as s; print(s('${MODEL_PATH}'))"
}
run_step 03_download download_model

############################
# 4. Demo inference on examples/ (depth maps + point clouds)
############################
DEMO_OUT="${LOG_DIR}/demo"
run_demo() {
    CUDA_VISIBLE_DEVICES="${FIRST_GPU}" python -u examples/run_demo.py \
        --model_path "${MODEL_PATH}" \
        --annotations_jsonl examples/examples.jsonl \
        --rgb_root examples/rgb \
        --depth_root examples/depth \
        --output_dir "${DEMO_OUT}" || return 1
    local n_png n_ply
    n_png=$(find "${DEMO_OUT}" -name "*.png" | wc -l)
    n_ply=$(find "${DEMO_OUT}" -name "*.ply" | wc -l)
    echo "[check] png=${n_png} ply=${n_ply}"
    [ "${n_png}" -gt 0 ] && [ "${n_ply}" -gt 0 ]
}
run_step 04_demo run_demo

############################
# 5. Eval on examples/ (sparse points and full depth map)
############################
run_eval() {
    local mode="$1"
    CUDA_VISIBLE_DEVICES="${FIRST_GPU}" python -u eval/eval.py \
        --model_path "${MODEL_PATH}" \
        --image_folder examples/rgb \
        --json_path examples/examples.jsonl \
        --depth_root examples/depth \
        --eval_mode "${mode}" \
        --generate_text \
        --bsz 4 \
        --output_dir "${LOG_DIR}/eval_${mode}" || return 1
}
run_step 05_eval_sparse run_eval sparse
run_step 06_eval_dense run_eval dense

############################
# 6. Optional: tiny two-stage training on examples/
############################
if [ "$RUN_TRAIN" -eq 1 ]; then
    # Data config pointing both RGB and depth to examples/
    SMOKE_CONF="${LOG_DIR}/smoke_train_datasets.conf"
    cat > "${SMOKE_CONF}" <<EOF
source configs/train_datasets.conf
EXAMPLES_JSONL="examples/examples.jsonl"
EXAMPLES_IMAGE="examples/rgb"
EXAMPLES_DEPTH="examples/depth"
EOF
    # Note: train.py only uses AutoProcessor's VL path when "vl" is in the model path
    STAGE1_DIR="${LOG_DIR}/depthvlm_stage1"
    STAGE2_DIR="${LOG_DIR}/depthvlm_stage2"
    TRAIN_GPUS=$(( NGPU > 0 ? NGPU : 1 ))

    train_stage() {
        local stage="$1"
        TRAIN_DATA_CONF="${SMOKE_CONF}" \
        TRAIN_DATASETS="EXAMPLES" \
        LOCAL_BATCH_SIZE=2 \
        GLOBAL_BATCH_SIZE=$(( TRAIN_GPUS * 2 )) \
        BASE_MODEL="${BASE_MODEL}" \
        STAGE1_CKPT="${STAGE1_DIR}" \
        OUTPUT_PATH="$([ "$stage" = 1 ] && echo "${STAGE1_DIR}" || echo "${STAGE2_DIR}")" \
        EXTRA_TRAIN_ARGS="--num_train_epochs 2 --logging_steps 1 --save_strategy no" \
            bash "train/train-stage${stage}.sh" 1 "${TRAIN_GPUS}" 127.0.0.1 $(( 29500 + stage )) 0 || return 1
        ls "$([ "$stage" = 1 ] && echo "${STAGE1_DIR}" || echo "${STAGE2_DIR}")"/*.safetensors >/dev/null
    }
    run_step 07_train_stage1 train_stage 1
    run_step 08_train_stage2 train_stage 2
    # Training saves fp32 master weights; export to bf16 and make sure the result still evaluates
    export_and_eval() {
        python scripts/export_bf16.py "${STAGE2_DIR}" "${STAGE2_DIR}_bf16" || return 1
        CUDA_VISIBLE_DEVICES="${FIRST_GPU}" python -u eval/eval.py \
            --model_path "${STAGE2_DIR}_bf16" \
            --image_folder examples/rgb \
            --json_path examples/examples.jsonl \
            --depth_root examples/depth \
            --eval_mode dense \
            --bsz 4 \
            --output_dir "${LOG_DIR}/eval_stage2_bf16"
    }
    run_step 09_export_bf16 export_and_eval
fi

############################
# Summary
############################
echo ""
echo "================ SUMMARY ================"
echo "logs: ${LOG_DIR}   attn: ${DEPTHVLM_ATTN_IMPL}   gpus: ${NGPU}"
n_fail=0
for i in "${!STEP_NAMES[@]}"; do
    printf "  %-18s %s\n" "${STEP_NAMES[$i]}" "${STEP_RESULTS[$i]}"
    [[ "${STEP_RESULTS[$i]}" == FAIL* ]] && n_fail=$(( n_fail + 1 ))
done
for mode in sparse dense; do
    line=$(grep -h "final delta_1" "${LOG_DIR}/05_eval_${mode}.log" "${LOG_DIR}/06_eval_${mode}.log" 2>/dev/null | tail -1)
    [ -n "$line" ] && echo "  [${mode}] ${line}"
done
for s in 07_train_stage1 08_train_stage2; do
    [ -f "${LOG_DIR}/${s}.log" ] || continue
    loss=$(grep -oE "'pixel_depth_loss': '?[0-9.e-]+" "${LOG_DIR}/${s}.log" | sed -n '1p;$p' | awk '{print $2}' | tr -d "'" | paste -sd' ' -)
    [ -n "$loss" ] && echo "  [${s}] pixel_depth_loss first/last: ${loss}"
done
exit $(( n_fail > 0 ? 1 : 0 ))
