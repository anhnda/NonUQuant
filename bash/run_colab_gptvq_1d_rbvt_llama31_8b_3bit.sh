#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.env"
  set +a
fi

RUN_SETUP="${RUN_SETUP:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/gptvq_1d_rbvt_llama31_8b_3bit}"

N_CALIB="${N_CALIB:-128}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
CALIB_DATASET="${CALIB_DATASET:-c4}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-2048}"
EVAL_STRIDE="${EVAL_STRIDE:-512}"
GROUPSIZE="${GROUPSIZE:-1024}"
KMEANS_ITERS="${KMEANS_ITERS:-100}"
ASSIGNMENT_CHUNK_SIZE="${ASSIGNMENT_CHUNK_SIZE:-4096}"
RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
DIAGNOSTIC_LAYER_LIMIT="${DIAGNOSTIC_LAYER_LIMIT:-6}"
DIAGNOSTIC_MAX_TOKENS="${DIAGNOSTIC_MAX_TOKENS:-4096}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-arc_challenge arc_easy boolq hellaswag lambada_openai openbookqa piqa rte winogrande mmlu}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== GPTVQ-1D vs GPTVQ-1D+RBVT | Llama-3.1-8B | 3-bit | single pass ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT_ROOT"
echo "Calibration: $CALIB_DATASET n=$N_CALIB len=$MAX_LENGTH"
echo "GPTVQ EM/k-means iterations: $KMEANS_ITERS"
echo "Activation diagnostics: first $DIAGNOSTIC_LAYER_LIMIT Linear layers, max_tokens=$DIAGNOSTIC_MAX_TOKENS"
echo "PPL: WikiText-2/C4 eval_samples=$EVAL_SAMPLES len=$EVAL_MAX_LENGTH stride=$EVAL_STRIDE"
echo "LM-eval tasks: $LM_EVAL_TASKS"
echo "GSM8K: disabled"

if [ "$RUN_SETUP" = "1" ]; then
  "$PYTHON_BIN" -m pip install -q -r requirements.txt
fi

if [[ ! -d GPTVQ/.git ]]; then
  git clone https://github.com/Qualcomm-AI-research/gptvq.git GPTVQ
else
  git -C GPTVQ pull --ff-only
fi

"$PYTHON_BIN" - <<'PY'
import sys
import transformers

if not hasattr(transformers, "Conv1D"):
    from transformers.pytorch_utils import Conv1D

    transformers.Conv1D = Conv1D

sys.path.insert(0, "GPTVQ")
from gptq import GPTQ  # noqa: F401
from modelutils import find_layers  # noqa: F401
from vq_quant import VQQuantizer  # noqa: F401

print("GPTVQ import smoke check passed.")
PY

COMMON_ARGS=(
  --model-path "$MODEL"
  --device "$DEVICE"
  --output-root "$OUTPUT_ROOT"
  --single-pass-compare
  --keep-model-on-device
  --wbits 3
  --groupsize "$GROUPSIZE"
  --gptq-blocksize 128
  --percdamp 0.01
  --kmeans-iters "$KMEANS_ITERS"
  --kmeans-init-method mahalanobis
  --assignment-chunk-size "$ASSIGNMENT_CHUNK_SIZE"
  --n-calib "$N_CALIB"
  --max-length "$MAX_LENGTH"
  --calib-dataset "$CALIB_DATASET"
  --eval-samples "$EVAL_SAMPLES"
  --eval-max-length "$EVAL_MAX_LENGTH"
  --eval-stride "$EVAL_STRIDE"
  --include-lm-eval
  --lm-eval-batch-size "$LM_EVAL_BATCH_SIZE"
  --lm-eval-output-dir "$OUTPUT_ROOT/lm_eval"
  --rbvt-lambda "$RBVT_LAMBDA"
  --diagnostic-layer-limit "$DIAGNOSTIC_LAYER_LIMIT"
  --diagnostic-max-tokens "$DIAGNOSTIC_MAX_TOKENS"
  --cleanup-model-artifacts
)

if [ -n "$LM_EVAL_LIMIT" ]; then
  COMMON_ARGS+=(--lm-eval-limit "$LM_EVAL_LIMIT")
fi
read -r -a LM_EVAL_TASK_ARRAY <<< "$LM_EVAL_TASKS"
COMMON_ARGS+=(--lm-eval-tasks "${LM_EVAL_TASK_ARRAY[@]}")

"$PYTHON_BIN" gptvq_rbvt_benchmark.py "${COMMON_ARGS[@]}"
