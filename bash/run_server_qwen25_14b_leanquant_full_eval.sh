#!/usr/bin/env bash
set -euo pipefail

# Disk-conscious Qwen2.5-14B LeanQuant full evaluation runner.
# Runs 4/3-bit x RTN/RBVT, with WikiText-2/C4 perplexity and lm-eval tasks
# including MMLU and GSM8K. It reuses precomputed LeanQuant codebooks when
# available, builds missing codebooks, then removes temporary artifacts/caches.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.env"
  set +a
fi

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-server}"
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif [ -x "$VENV_DIR/bin/python" ]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
  else
    PYTHON_BIN="$(command -v python || command -v python3 || true)"
  fi
fi

MODEL="${MODEL:-Qwen/Qwen2.5-14B}"
MODEL_SLUG="${MODEL_SLUG:-Qwen2p5-14B}"
DEVICE="${DEVICE:-cuda:0}"
BITS="${BITS:-4 3}"
METHODS="${METHODS:-rtn rbvt}"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/leanquant_server}"
STATISTICS_CACHE_DIR="${STATISTICS_CACHE_DIR:-$OUTPUT_ROOT/_statistics}"
CODEBOOK_IMPORT_ROOT="${CODEBOOK_IMPORT_ROOT:-/root/leanquant_qwen25_14b_codebooks}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"

CACHE_ROOT="${CACHE_ROOT:-$WORKSPACE_ROOT/rbvtquant_runtime_cache}"
HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-$CACHE_ROOT/evaluation}"
LM_EVAL_OUTPUT_DIR="${LM_EVAL_OUTPUT_DIR:-$OUTPUT_ROOT/lm_eval}"

N_CALIB="${N_CALIB:-128}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
SEED="${SEED:-42}"
ROW_CHUNK="${ROW_CHUNK:-1024}"
LEANQUANT_EXPONENT="${LEANQUANT_EXPONENT:-4.0}"
LEANQUANT_PERCDAMP="${LEANQUANT_PERCDAMP:-0.15}"
KMEANS_SEED="${KMEANS_SEED:-0}"
RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
RBVT_TOPK="${RBVT_TOPK:-0}"
GAP_FLOOR="${GAP_FLOOR:-1e-8}"

EVAL_STRIDE="${EVAL_STRIDE:-512}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-2048}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-arc_challenge arc_easy boolq hellaswag lambada_openai openbookqa piqa rte winogrande mmlu gsm8k}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
LM_EVAL_NUM_FEWSHOT="${LM_EVAL_NUM_FEWSHOT:-}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"

RUN_SETUP="${RUN_SETUP:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-0}"
FORCE_EVAL="${FORCE_EVAL:-1}"
BUILD_MISSING_CODEBOOKS="${BUILD_MISSING_CODEBOOKS:-1}"
REBUILD_INVALID_CODEBOOKS="${REBUILD_INVALID_CODEBOOKS:-1}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-rbvtquant}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
MIN_FREE_GIB="${MIN_FREE_GIB:-30}"
CLEAN_PPL_CACHE_BETWEEN_RUNS="${CLEAN_PPL_CACHE_BETWEEN_RUNS:-1}"
CLEAN_ROOT_CACHE="${CLEAN_ROOT_CACHE:-1}"

export HF_HOME
export HF_DATASETS_CACHE
export HF_MODULES_CACHE
export TRANSFORMERS_CACHE
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export RBVT_LEANQUANT_CACHE_HESSIAN="${RBVT_LEANQUANT_CACHE_HESSIAN:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"

if [ "$RUN_SETUP" = "1" ]; then
  VENV_DIR="$VENV_DIR" CACHE_ROOT="$CACHE_ROOT" bash bash/setup_server_leanquant.sh
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Error: Python environment not found. Set PYTHON_BIN or run setup first." >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}}" ]; then
  echo "Error: HF_TOKEN is required for $MODEL." >&2
  exit 1
fi

mkdir -p \
  "$OUTPUT_ROOT" \
  "$STATISTICS_CACHE_DIR/codebooks/$MODEL_SLUG" \
  "$LOG_DIR" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$HF_MODULES_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$EVAL_CACHE_DIR" \
  "$LM_EVAL_OUTPUT_DIR"

read -r -a BITS_ARRAY <<< "$BITS"
read -r -a METHOD_ARRAY <<< "$METHODS"
read -r -a LM_EVAL_TASK_ARRAY <<< "$LM_EVAL_TASKS"

free_gib() {
  df -BG "$WORKSPACE_ROOT" 2>/dev/null | awk 'NR==2 {gsub("G", "", $4); print $4}'
}

print_space() {
  echo "== disk =="
  df -h / "$WORKSPACE_ROOT" 2>/dev/null | awk 'NR==1 || NR==2 || NR==3'
  echo "== large dirs =="
  du -sh "$OUTPUT_ROOT" "$CACHE_ROOT" /root/.cache 2>/dev/null || true
}

cleanup_ppl_cache() {
  echo "Cleaning PPL/eval caches; preserving LeanQuant codebooks and HF model cache ..."
  rm -rf "$EVAL_CACHE_DIR"
  mkdir -p "$EVAL_CACHE_DIR"

  find "$HF_DATASETS_CACHE" -maxdepth 4 -type d \( \
    -iname '*wikitext*' -o \
    -iname '*c4*' -o \
    -iname '*allenai___c4*' -o \
    -iname '*Salesforce___wikitext*' \
  \) -prune -exec rm -rf {} + 2>/dev/null || true

  rm -rf "$HF_MODULES_CACHE/datasets_modules" 2>/dev/null || true

  if [ "$CLEAN_ROOT_CACHE" = "1" ]; then
    rm -rf /root/.cache/huggingface /root/.cache/pip /root/.cache/torch 2>/dev/null || true
  fi
}

cleanup_weight_outputs() {
  echo "Removing saved model weights from output runs ..."
  find "$OUTPUT_ROOT" -maxdepth 2 -type f \( \
    -name 'model.safetensors' -o \
    -name '*.safetensors' -o \
    -name 'pytorch_model*.bin' \
  \) -not -path "$STATISTICS_CACHE_DIR/codebooks/*" -delete 2>/dev/null || true
}

codebook_manifest() {
  local bits="$1"
  echo "$STATISTICS_CACHE_DIR/codebooks/$MODEL_SLUG/leanquant_${bits}bit_direct_upstream/manifest.json"
}

validate_codebook_manifest() {
  local manifest="$1"
  local bits="$2"
  "$PYTHON_BIN" - "$manifest" "$bits" "$LEANQUANT_PERCDAMP" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
bits = sys.argv[2]
expected_percdamp = float(sys.argv[3])
if not manifest.exists():
    raise SystemExit(f"Missing {bits}-bit manifest: {manifest}")
data = json.loads(manifest.read_text(encoding="utf-8"))
if not data.get("complete"):
    raise SystemExit(f"Incomplete {bits}-bit codebook manifest: {manifest}")
actual_percdamp = data.get("metadata", {}).get("percdamp", data.get("percdamp"))
if actual_percdamp is not None and abs(float(actual_percdamp) - expected_percdamp) > 1e-12:
    raise SystemExit(
        f"{bits}-bit codebook percdamp mismatch: manifest={actual_percdamp}, "
        f"runner={expected_percdamp}"
    )
print(f"Codebook OK: {bits}-bit -> {manifest.parent}")
PY
}

codebook_complete() {
  local bits="$1"
  local manifest
  manifest="$(codebook_manifest "$bits")"
  [ -f "$manifest" ] || return 1
  validate_codebook_manifest "$manifest" "$bits" >/dev/null
}

cleanup_auxiliary_statistics() {
  local bits="$1"
  local hessian_dir="$STATISTICS_CACHE_DIR/hessian/$MODEL_SLUG/leanquant_${bits}bit_direct_upstream"

  if codebook_complete "$bits"; then
    rm -rf "$hessian_dir" 2>/dev/null || true
  fi
  rm -rf "$STATISTICS_CACHE_DIR/calibration" 2>/dev/null || true
  find "$STATISTICS_CACHE_DIR" -type d -empty -delete 2>/dev/null || true
}

copy_codebook_if_needed() {
  local bits="$1"
  local name="leanquant_${bits}bit_direct_upstream"
  local dst="$STATISTICS_CACHE_DIR/codebooks/$MODEL_SLUG/$name"
  local manifest="$dst/manifest.json"

  if [ -f "$manifest" ]; then
    return 0
  fi

  local src=""
  for candidate in \
    "$CODEBOOK_IMPORT_ROOT/$MODEL_SLUG/$name" \
    "$CODEBOOK_IMPORT_ROOT/$name" \
    "$WORKSPACE_ROOT/leanquant_qwen25_14b_codebooks/$MODEL_SLUG/$name" \
    "$WORKSPACE_ROOT/leanquant_qwen25_14b_codebooks/$name"
  do
    if [ -f "$candidate/manifest.json" ]; then
      src="$candidate"
      break
    fi
  done

  if [ -z "$src" ]; then
    return 1
  fi

  mkdir -p "$dst"
  rsync -a --ignore-existing "$src/" "$dst/"
}

prepare_codebooks() {
  for bits in "${BITS_ARRAY[@]}"; do
    local manifest
    manifest="$(codebook_manifest "$bits")"
    if [ ! -f "$manifest" ]; then
      copy_codebook_if_needed "$bits" || true
    fi

    if [ -f "$manifest" ]; then
      if validate_codebook_manifest "$manifest" "$bits"; then
        cleanup_auxiliary_statistics "$bits"
        continue
      fi

      if [ "$BUILD_MISSING_CODEBOOKS" = "1" ] && [ "$REBUILD_INVALID_CODEBOOKS" = "1" ]; then
        echo "Invalid/incomplete ${bits}-bit codebook; removing it so the next job rebuilds cleanly."
        rm -rf "$(dirname "$manifest")"
        rm -rf "$STATISTICS_CACHE_DIR/hessian/$MODEL_SLUG/leanquant_${bits}bit_direct_upstream"
        rm -rf "$STATISTICS_CACHE_DIR/calibration"
        continue
      fi

      exit 1
    fi

    if [ "$BUILD_MISSING_CODEBOOKS" = "1" ]; then
      echo "Codebook missing for ${bits}-bit; it will be built during the first ${bits}-bit job."
      continue
    fi

    echo "Error: missing ${bits}-bit codebook and BUILD_MISSING_CODEBOOKS=0." >&2
    echo "Expected: $manifest" >&2
    exit 1
  done
}

summary_has_full_metrics() {
  local summary_path="$1"
  "$PYTHON_BIN" - "$summary_path" <<'PY'
import json
import sys
from pathlib import Path

from runtime_utils import pick_lm_eval_metric

summary_path = Path(sys.argv[1])
if not summary_path.exists():
    raise SystemExit(1)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
perplexity = summary.get("evaluation", {}).get("perplexity", {})
for dataset in ("WikiText-2", "C4"):
    value = perplexity.get(dataset, {}).get("perplexity")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SystemExit(1)
run_label = summary.get("run_label")
tasks = summary.get("args", {}).get("lm_eval_tasks", [])
task_summary = (
    summary.get("evaluation", {})
    .get("lm_eval", {})
    .get(run_label, {})
    .get("summary", {})
)
for task in tasks:
    metrics = task_summary.get(task, {})
    if task == "gsm8k":
        strict = metrics.get("exact_match,strict-match")
        flex = metrics.get("exact_match,flexible-extract")
        if not any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (strict, flex)):
            raise SystemExit(1)
    _, score = pick_lm_eval_metric(metrics)
    if score is None:
        raise SystemExit(1)
PY
}

run_one() {
  local bits="$1"
  local method="$2"
  local run_dir="$OUTPUT_ROOT/leanquant_${bits}bit_${method}"
  local free
  free="$(free_gib || echo 0)"
  if [ "${free:-0}" -lt "$MIN_FREE_GIB" ]; then
    echo "Error: only ${free}GiB free in $WORKSPACE_ROOT; need ${MIN_FREE_GIB}GiB." >&2
    print_space
    exit 1
  fi

  echo
  echo "=== LeanQuant Qwen2.5-14B | ${bits}-bit ${method^^} | PPL + lm-eval ==="
  print_space

  local args=(
    --model-path "$MODEL"
    --device "$DEVICE"
    --output-root "$OUTPUT_ROOT"
    --resume
    --codebooks leanquant
    --bits "$bits"
    --methods "$method"
    --calib-dataset c4
    --n-calib "$N_CALIB"
    --max-length "$MAX_LENGTH"
    --seed "$SEED"
    --group-size -1
    --row-chunk "$ROW_CHUNK"
    --leanquant-exponent "$LEANQUANT_EXPONENT"
    --leanquant-percdamp "$LEANQUANT_PERCDAMP"
    --kmeans-seed "$KMEANS_SEED"
    --statistics-cache-dir "$STATISTICS_CACHE_DIR"
    --rbvt-lambda "$RBVT_LAMBDA"
    --rbvt-topk "$RBVT_TOPK"
    --gap-floor "$GAP_FLOOR"
    --eval-stride "$EVAL_STRIDE"
    --eval-max-length "$EVAL_MAX_LENGTH"
    --eval-samples "$EVAL_SAMPLES"
    --eval-cache-dir "$EVAL_CACHE_DIR"
    --include-lm-eval
    --lm-eval-tasks "${LM_EVAL_TASK_ARRAY[@]}"
    --lm-eval-batch-size "$LM_EVAL_BATCH_SIZE"
    --lm-eval-output-dir "$LM_EVAL_OUTPUT_DIR"
  )

  if [ "$FORCE_EVAL" = "1" ]; then
    args+=(--force-eval)
  fi
  if [ -n "$LM_EVAL_NUM_FEWSHOT" ]; then
    args+=(--lm-eval-num-fewshot "$LM_EVAL_NUM_FEWSHOT")
  fi
  if [ -n "$LM_EVAL_LIMIT" ]; then
    args+=(--lm-eval-limit "$LM_EVAL_LIMIT")
  fi
  if [ "$USE_WANDB" = "1" ]; then
    args+=(--use-wandb --wandb-project "$WANDB_PROJECT")
    if [ -n "$WANDB_ENTITY" ]; then
      args+=(--wandb-entity "$WANDB_ENTITY")
    fi
  else
    args+=(--no-wandb)
  fi

  set +e
  "$PYTHON_BIN" codebook_benchmark.py "${args[@]}"
  local status=$?
  set -e

  cleanup_weight_outputs
  cleanup_auxiliary_statistics "$bits"
  if [ "$CLEAN_PPL_CACHE_BETWEEN_RUNS" = "1" ]; then
    cleanup_ppl_cache
  fi
  print_space

  if [ "$status" -ne 0 ]; then
    return "$status"
  fi
  summary_has_full_metrics "$run_dir/run_summary.json"
}

build_report() {
  local args=(
    --model-path "$MODEL"
    --device "$DEVICE"
    --output-root "$OUTPUT_ROOT"
    --resume
    --codebooks leanquant
    --bits "${BITS_ARRAY[@]}"
    --methods "${METHOD_ARRAY[@]}"
    --calib-dataset c4
    --n-calib "$N_CALIB"
    --max-length "$MAX_LENGTH"
    --seed "$SEED"
    --group-size -1
    --row-chunk "$ROW_CHUNK"
    --leanquant-exponent "$LEANQUANT_EXPONENT"
    --leanquant-percdamp "$LEANQUANT_PERCDAMP"
    --kmeans-seed "$KMEANS_SEED"
    --statistics-cache-dir "$STATISTICS_CACHE_DIR"
    --rbvt-lambda "$RBVT_LAMBDA"
    --rbvt-topk "$RBVT_TOPK"
    --gap-floor "$GAP_FLOOR"
    --eval-stride "$EVAL_STRIDE"
    --eval-max-length "$EVAL_MAX_LENGTH"
    --eval-samples "$EVAL_SAMPLES"
    --eval-cache-dir "$EVAL_CACHE_DIR"
    --include-lm-eval
    --lm-eval-tasks "${LM_EVAL_TASK_ARRAY[@]}"
    --lm-eval-batch-size "$LM_EVAL_BATCH_SIZE"
    --lm-eval-output-dir "$LM_EVAL_OUTPUT_DIR"
    --no-wandb
  )
  "$PYTHON_BIN" codebook_benchmark.py "${args[@]}"
}

if [ "$RUN_PREFLIGHT" = "1" ]; then
  "$PYTHON_BIN" - "$MODEL" <<'PY'
import sys
import torch
from transformers import AutoConfig

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print("GPU:", torch.cuda.get_device_name(0))
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")
config = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
print("Model config:", config.model_type)
PY
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/qwen25_14b_full_eval_${TIMESTAMP}.log"

{
  echo "=== Qwen2.5-14B LeanQuant full eval ==="
  echo "Repository: $ROOT_DIR"
  echo "Model: $MODEL"
  echo "Bits: $BITS"
  echo "Methods: $METHODS"
  echo "LeanQuant: exponent=$LEANQUANT_EXPONENT percdamp=$LEANQUANT_PERCDAMP"
  echo "PPL: WikiText-2 + C4 | samples=$EVAL_SAMPLES stride=$EVAL_STRIDE max_length=$EVAL_MAX_LENGTH"
  echo "lm-eval tasks: $LM_EVAL_TASKS"
  echo "Output: $OUTPUT_ROOT"
  echo "Codebooks: $STATISTICS_CACHE_DIR/codebooks/$MODEL_SLUG"
  echo "Build missing codebooks: $BUILD_MISSING_CODEBOOKS"
  echo "Rebuild invalid codebooks: $REBUILD_INVALID_CODEBOOKS"
  echo "LeanQuant Hessian cache: $RBVT_LEANQUANT_CACHE_HESSIAN"
  echo "Runtime cache: $CACHE_ROOT"
  echo "Log: $LOG_FILE"
  nvidia-smi || true
} 2>&1 | tee -a "$LOG_FILE"

cleanup_ppl_cache 2>&1 | tee -a "$LOG_FILE"
cleanup_weight_outputs 2>&1 | tee -a "$LOG_FILE"
prepare_codebooks 2>&1 | tee -a "$LOG_FILE"

run_index=0
total_runs=$((${#BITS_ARRAY[@]} * ${#METHOD_ARRAY[@]}))
for bits in "${BITS_ARRAY[@]}"; do
  for method in "${METHOD_ARRAY[@]}"; do
    run_index=$((run_index + 1))
    {
      echo
      echo "=== Job ${run_index}/${total_runs}: ${bits}-bit ${method^^} ==="
      run_one "$bits" "$method"
    } 2>&1 | tee -a "$LOG_FILE"
  done
done

{
  echo
  echo "=== Building combined report ==="
  build_report
  cleanup_weight_outputs
  for bits in "${BITS_ARRAY[@]}"; do
    cleanup_auxiliary_statistics "$bits"
  done
  print_space
} 2>&1 | tee -a "$LOG_FILE"

echo "Full eval complete."
echo "Summary CSV: $OUTPUT_ROOT/benchmark_results.csv"
echo "Summary JSON: $OUTPUT_ROOT/benchmark_results.json"
echo "Log: $LOG_FILE"
