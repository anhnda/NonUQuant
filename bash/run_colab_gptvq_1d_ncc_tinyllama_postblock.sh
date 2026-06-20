#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_SETUP="${RUN_SETUP:-1}"
MODEL="${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/gptvq_1d_ncc_tinyllama_postblock}"

N_CALIB="${N_CALIB:-32}"
MAX_LENGTH="${MAX_LENGTH:-512}"
EVAL_SAMPLES="${EVAL_SAMPLES:-64}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-1024}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-100}"
GROUPSIZE="${GROUPSIZE:-128}"
GPTQ_BLOCKSIZE="${GPTQ_BLOCKSIZE:-128}"
KMEANS_ITERS="${KMEANS_ITERS:-20}"
ASSIGNMENT_CHUNK_SIZE="${ASSIGNMENT_CHUNK_SIZE:-4096}"
NCC_BUDGET_P="${NCC_BUDGET_P:-0.02}"
NCC_SWEEPS="${NCC_SWEEPS:-1}"
NCC_STOP_EPS="${NCC_STOP_EPS:-0.0}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-arc_easy arc_challenge}"
DIAGNOSTIC_LAYER_LIMIT="${DIAGNOSTIC_LAYER_LIMIT:-3}"
DIAGNOSTIC_MAX_TOKENS="${DIAGNOSTIC_MAX_TOKENS:-4096}"

echo "=== GPTVQ-1D+NCC TinyLlama post-block Colab benchmark ==="
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Output: $OUTPUT_ROOT"
echo "Bits: 4"
echo "GPTQ blocksize: $GPTQ_BLOCKSIZE | groupsize=$GROUPSIZE"
echo "NCC placement: post_block | budget_p=$NCC_BUDGET_P | sweeps=$NCC_SWEEPS | stop_eps=$NCC_STOP_EPS"
echo "M-step: disabled so post-block NCC is not overwritten"
echo "PPL: WikiText-2 + C4 | eval_samples=$EVAL_SAMPLES"
echo "LM-eval tasks: $LM_EVAL_TASKS | limit=$LM_EVAL_LIMIT"
echo "Activation diagnostics: first $DIAGNOSTIC_LAYER_LIMIT Linear layers, max_tokens=$DIAGNOSTIC_MAX_TOKENS"

if [[ "$GROUPSIZE" != "$GPTQ_BLOCKSIZE" ]]; then
  echo "post_block mode currently requires GROUPSIZE == GPTQ_BLOCKSIZE" >&2
  exit 1
fi

if [[ "$RUN_SETUP" == "1" ]]; then
  python3 -m pip install -q -r requirements.txt
fi

if [[ ! -d GPTVQ/.git ]]; then
  git clone https://github.com/Qualcomm-AI-research/gptvq.git GPTVQ
else
  git -C GPTVQ pull --ff-only
fi

if [[ ! -d NCCQuant/.git ]]; then
  git clone https://github.com/anhnda/NCCQuant.git NCCQuant
else
  git -C NCCQuant pull --ff-only
fi

python3 - <<'PY'
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
print("NCCQuant source present: NCCQuant/quantizers/ncc.py")
PY

python3 gptvq_rbvt_benchmark.py \
  --model-path "$MODEL" \
  --device "$DEVICE" \
  --output-root "$OUTPUT_ROOT" \
  --variants gptvq_ncc \
  --correction ncc \
  --ncc-placement post_block \
  --wbits 4 \
  --groupsize "$GROUPSIZE" \
  --gptq-blocksize "$GPTQ_BLOCKSIZE" \
  --percdamp 0.01 \
  --kmeans-iters "$KMEANS_ITERS" \
  --kmeans-init-method mahalanobis \
  --assignment-chunk-size "$ASSIGNMENT_CHUNK_SIZE" \
  --no-include-m-step \
  --n-calib "$N_CALIB" \
  --max-length "$MAX_LENGTH" \
  --calib-dataset wikitext2 \
  --eval-samples "$EVAL_SAMPLES" \
  --eval-max-length "$EVAL_MAX_LENGTH" \
  --eval-stride 512 \
  --include-lm-eval \
  --lm-eval-tasks $LM_EVAL_TASKS \
  --lm-eval-limit "$LM_EVAL_LIMIT" \
  --lm-eval-batch-size auto \
  --ncc-budget-p "$NCC_BUDGET_P" \
  --ncc-sweeps "$NCC_SWEEPS" \
  --ncc-stop-eps "$NCC_STOP_EPS" \
  --diagnostic-layer-limit "$DIAGNOSTIC_LAYER_LIMIT" \
  --diagnostic-max-tokens "$DIAGNOSTIC_MAX_TOKENS" \
  --cleanup-model-artifacts
