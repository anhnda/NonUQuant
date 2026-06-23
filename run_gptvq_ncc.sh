#!/usr/bin/env bash
# =============================================================================
# run_gptvq_ncc.sh
# -----------------------------------------------------------------------------
# Quantize a model with upstream GPTVQ-1D (scalar VQ, vq_dim=1) and produce full
# quantized models for:
#   1. GPTVQ-1D base                (NCC sweep with budget 0 -> effectively none)
#   2. GPTVQ-1D + NCC, post_module  (recommended; NCC after each Linear module)
#   3. GPTVQ-1D + NCC, post_block   (NCC inside GPTVQ after each GPTQ block)
#
# NOTE vs run_gptq_ncc.sh: the gptvq path does NOT use --quantizer nf*, nor the
# NCC scoring knobs --ncc-score / --ncc-baseline / --ncc-cov-eps / --ncc-mse-guard
# (those belong to the GPTQ+NCC path). GPTVQ-1D's NCC reads only --ncc-budget-p,
# --ncc-sweeps, --ncc-stop-eps, --ncc-placement, --ncc-james-stein. Bit-width is
# set by --wbits {3,4} and the codebook group by --groupsize. The k-means knobs
# (--kmeans-iters, --kmeans-init-method, --include-m-step,
# --hessian-weighted-lookups, --true-sequential) control the VQ codebook.
#
# Each run writes a full model via model.save_pretrained() to its own dir, then
# runs perplexity + lm-eval exactly like the other methods. Pick which variants
# to run with the RUN_* toggles.
#
# This script does NOT run anything on import; you invoke it. It assumes the
# patched main.py (with the gptvq branch) + gptvq_rbvt_benchmark.py are in place,
# and that ./GPTVQ and ./NCCQuant are cloned.
# =============================================================================
set -euo pipefail

# ---- paths / model ----------------------------------------------------------
REPO_DIR="${REPO_DIR:-$(pwd)}"               # dir containing main.py
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
DEVICE="${DEVICE:-cuda:0}"
OUT_ROOT="${OUT_ROOT:-./runs_gptvq_ncc}"

# ---- GPTVQ-1D codebook knobs ------------------------------------------------
WBITS="${WBITS:-3}"                           # 3|4
GROUPSIZE="${GROUPSIZE:-128}"
KMEANS_ITERS="${KMEANS_ITERS:-20}"
KMEANS_INIT="${KMEANS_INIT:-mahalanobis}"     # cdf|kpp|mahalanobis
INCLUDE_M_STEP="${INCLUDE_M_STEP:-1}"         # 1 -> M-step on, 0 -> --no-include-m-step
HESSIAN_LOOKUPS="${HESSIAN_LOOKUPS:-1}"       # 1 -> on, 0 -> --no-hessian-weighted-lookups
TRUE_SEQUENTIAL="${TRUE_SEQUENTIAL:-1}"       # 1 -> on, 0 -> --no-true-sequential
KEEP_ON_DEVICE="${KEEP_ON_DEVICE:-0}"         # 1 -> --keep-model-on-device (more VRAM, less CPU RAM/IO)

# ---- calibration / NCC knobs ------------------------------------------------
N_CALIB="${N_CALIB:-128}"
MAX_LEN="${MAX_LEN:-2048}"
CALIB_DS="${CALIB_DS:-c4}"                    # c4|wikitext2
BUDGET_P="${BUDGET_P:-0.02}"
NCC_SWEEPS="${NCC_SWEEPS:-1}"
NCC_STOP_EPS="${NCC_STOP_EPS:-0.0}"
NCC_SCORE="${NCC_SCORE:-cov}"                 # cov|lite (mse-guard needs cov)
COV_EPS="${COV_EPS:-1e-6}"
MSE_GUARD="${MSE_GUARD:-0}"          # 1 -> add --ncc-mse-guard (gap<2|e| Cor-2 filter)
JAMES_STEIN="${JAMES_STEIN:-0}"               # 1 -> --ncc-james-stein
GPTQ_BLOCKSIZE="${GPTQ_BLOCKSIZE:-128}"
GPTQ_PERCDAMP="${GPTQ_PERCDAMP:-0.01}"

# ---- eval toggles (set to 0 to skip the heavy lm-eval during debugging) -----
LM_EVAL="${LM_EVAL:-1}"                       # 1 -> include lm-eval, 0 -> skip
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"

# ---- which variants to run --------------------------------------------------
RUN_BASE="${RUN_BASE:-1}"                     # GPTVQ-1D, no correction (budget 0)
RUN_NCC_POST_MODULE="${RUN_NCC_POST_MODULE:-1}"
RUN_NCC_POST_BLOCK="${RUN_NCC_POST_BLOCK:-0}"

# -----------------------------------------------------------------------------
cd "$REPO_DIR"
mkdir -p "$OUT_ROOT"
# fresh comparison table
printf "variant\tperplexity...\tquant_stats\n" > "$OUT_ROOT/perplexity_table.tsv"

# GPTVQ upstream is required for every variant.
if [[ ! -d "GPTVQ" ]]; then
  echo "[setup] GPTVQ not found -> cloning ..."
  git clone https://github.com/Qualcomm-AI-research/gptvq.git GPTVQ
fi
# Clone NCCQuant if any NCC variant is requested and it's missing.
if [[ "$RUN_NCC_POST_MODULE$RUN_NCC_POST_BLOCK" == *1* ]]; then
  if [[ ! -f "NCCQuant/quantizers/ncc.py" ]]; then
    echo "[setup] NCCQuant not found -> cloning ..."
    git clone https://github.com/anhnda/NCCQuant.git NCCQuant
  fi
fi

# slug for output dirs / table rows
SLUG="gptvq${WBITS}b_g${GROUPSIZE}"

# common args shared by every run
common_args=(
  --model-path "$MODEL"
  --method gptvq
  --device "$DEVICE"
  --wbits "$WBITS"
  --groupsize "$GROUPSIZE"
  --kmeans-iters "$KMEANS_ITERS"
  --kmeans-init-method "$KMEANS_INIT"
  --gptq-blocksize "$GPTQ_BLOCKSIZE"
  --gptq-percdamp "$GPTQ_PERCDAMP"
  --n-calib "$N_CALIB"
  --max-length "$MAX_LEN"
  --calib-dataset "$CALIB_DS"
  --eval-samples "$EVAL_SAMPLES"
)
[[ "$INCLUDE_M_STEP"   == "0" ]] && common_args+=(--no-include-m-step)
[[ "$HESSIAN_LOOKUPS"  == "0" ]] && common_args+=(--no-hessian-weighted-lookups)
[[ "$TRUE_SEQUENTIAL"  == "0" ]] && common_args+=(--no-true-sequential)
[[ "$KEEP_ON_DEVICE"   == "1" ]] && common_args+=(--keep-model-on-device)
[[ "$LM_EVAL"          == "0" ]] && common_args+=(--no-lm-eval)

run_variant () {
  local tag="$1"; shift
  local outdir="$OUT_ROOT/${SLUG}_${tag}"
  echo ""
  echo "================================================================"
  echo ">>> VARIANT: $tag  ->  $outdir"
  echo "================================================================"
  # main.py: quantize -> save_pretrained -> perplexity eval -> (lm-eval) ->
  # save run_summary.json -> cleanup_output_dir (deletes model, keeps summary).
  set +e
  python main.py "${common_args[@]}" --output-dir "$outdir" "$@" \
    2>&1 | tee "$OUT_ROOT/log_${SLUG}_${tag}.txt"
  local rc=${PIPESTATUS[0]}
  set -e

  # Safety net: if main.py crashed before its own cleanup, delete model shards
  # ourselves so repeated variants don't fill the disk. Keep run_summary.json.
  if [[ -d "$outdir" ]]; then
    find "$outdir" -type f \
      ! -name "run_summary.json" \
      \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" \
         -o -name "*.json" -o -name "*.model" -o -name "*.txt" \) \
      ! -name "run_summary.json" -delete 2>/dev/null || true
  fi

  # Pull perplexity out of the summary into the comparison table.
  if [[ -f "$outdir/run_summary.json" ]]; then
    python - "$tag" "$outdir/run_summary.json" >> "$OUT_ROOT/perplexity_table.tsv" <<'PYEOF'
import json, sys
tag, path = sys.argv[1], sys.argv[2]
try:
    s = json.load(open(path))
    q = (s.get("evaluation", {}).get("quantized_model", {})
         or s.get("results", {}).get("quantized_model", {})
         or s.get("quantized_model", {}))
    # results layout: {dataset: {"perplexity": x, ...}}
    cols = []
    for ds, m in (q.items() if isinstance(q, dict) else []):
        if isinstance(m, dict) and "perplexity" in m:
            cols.append(f"{ds}={m['perplexity']:.4f}")
    qs = s.get("quantization", {})
    extra = []
    for k in ("method", "bits", "vq_dim", "flips", "bias_before", "bias_after"):
        if k in qs:
            extra.append(f"{k}={qs[k]}")
    print(tag + "\t" + "\t".join(cols) + "\t" + " ".join(extra))
except Exception as e:
    print(f"{tag}\t<parse-error: {e}>")
PYEOF
  fi

  if [[ "$rc" != "0" ]]; then
    echo "!! variant $tag exited with code $rc (model cleaned up; summary kept if produced)."
  fi
}

# NCC scoring + safety flags shared by the NCC variants.
ncc_flags=(--ncc-score "$NCC_SCORE" --ncc-cov-eps "$COV_EPS")
[[ "$MSE_GUARD"   == "1" ]] && ncc_flags+=(--ncc-mse-guard)
[[ "$JAMES_STEIN" == "1" ]] && ncc_flags+=(--ncc-james-stein)

# 1) GPTVQ-1D base ------------------------------------------------------------
# The gptvq branch always runs NCC sweeps; budget_p=0 admits no flips, so this is
# the uncorrected GPTVQ-1D baseline.
if [[ "$RUN_BASE" == "1" ]]; then
  run_variant "base" \
    --ncc-budget-p 0.0 --ncc-sweeps 1 --ncc-placement post_module
fi

# 2) GPTVQ-1D + NCC, post_module (recommended) --------------------------------
if [[ "$RUN_NCC_POST_MODULE" == "1" ]]; then
  run_variant "ncc_post_module" \
    --ncc-budget-p "$BUDGET_P" --ncc-sweeps "$NCC_SWEEPS" \
    --ncc-stop-eps "$NCC_STOP_EPS" --ncc-placement post_module "${ncc_flags[@]}"
fi

# 3) GPTVQ-1D + NCC, post_block -----------------------------------------------
# post_block requires --groupsize == --gptq-blocksize and --no-include-m-step.
if [[ "$RUN_NCC_POST_BLOCK" == "1" ]]; then
  run_variant "ncc_post_block" \
    --ncc-budget-p "$BUDGET_P" --ncc-sweeps "$NCC_SWEEPS" \
    --ncc-stop-eps "$NCC_STOP_EPS" --ncc-placement post_block \
    --no-include-m-step "${ncc_flags[@]}"
fi

echo ""
echo "================================================================"
echo "All requested variants done. Models DELETED after eval; kept:"
echo "  $OUT_ROOT/${SLUG}_<tag>/run_summary.json   (has perplexity)"
echo "  $OUT_ROOT/log_${SLUG}_*.txt                (full logs)"
echo ""
echo "Perplexity comparison:"
echo "----------------------------------------------------------------"
column -t -s$'\t' "$OUT_ROOT/perplexity_table.tsv" 2>/dev/null || cat "$OUT_ROOT/perplexity_table.tsv"
echo "================================================================"