#!/usr/bin/env bash
# =============================================================================
# run_gptq_ncc.sh
# -----------------------------------------------------------------------------
# Quantize a model with non-uniform GPTQ and produce full quantized models for:
#   1. GPTQ base                 (no correction)
#   2. GPTQ + NCC-Cov, original  (recommended NCC; baseline = original fp)
#   3. GPTQ + NCC-Cov, adjusted  (baseline = error-feedback-adjusted weights)
#   4. GPTQ + NCC-Lite, original (covariance-free score, for ablation)
#
# Each run writes a full model via model.save_pretrained() to its own dir, then
# runs perplexity + lm-eval exactly like the other methods. Pick which variants
# to run with the RUN_* toggles.
#
# This script does NOT run anything on import; you invoke it. It also assumes
# the patched main.py + nonuniform_gptq.py are in place and NCCQuant is cloned.
# =============================================================================
set -euo pipefail

# ---- paths / model ----------------------------------------------------------
REPO_DIR="${REPO_DIR:-$(pwd)}"               # dir containing main.py
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
QUANTIZER="${QUANTIZER:-nf4}"                 # nf3|nf4|nvfp4|codebook3|codebook4
DEVICE="${DEVICE:-cuda:0}"
OUT_ROOT="${OUT_ROOT:-./runs_gptq_ncc}"

# ---- calibration / NCC knobs ------------------------------------------------
N_CALIB="${N_CALIB:-128}"
MAX_LEN="${MAX_LEN:-2048}"
CALIB_DS="${CALIB_DS:-c4}"                    # c4|wikitext2
BUDGET_P="${BUDGET_P:-0.02}"
COV_EPS="${COV_EPS:-1e-6}"
GPTQ_BLOCKSIZE="${GPTQ_BLOCKSIZE:-128}"
GPTQ_PERCDAMP="${GPTQ_PERCDAMP:-0.01}"

# ---- eval toggles (set to 0 to skip the heavy lm-eval during debugging) -----
LM_EVAL="${LM_EVAL:-1}"                       # 1 -> include lm-eval, 0 -> skip
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"

# ---- which variants to run --------------------------------------------------
RUN_BASE="${RUN_BASE:-1}"
RUN_NCC_COV_ORIG="${RUN_NCC_COV_ORIG:-1}"
RUN_NCC_COV_ADJ="${RUN_NCC_COV_ADJ:-1}"
RUN_NCC_LITE_ORIG="${RUN_NCC_LITE_ORIG:-0}"

# -----------------------------------------------------------------------------
cd "$REPO_DIR"
mkdir -p "$OUT_ROOT"
# fresh comparison table
printf "variant\tperplexity...\tquant_stats\n" > "$OUT_ROOT/perplexity_table.tsv"

# Clone NCCQuant if NCC variants are requested and it's missing.
if [[ "$RUN_NCC_COV_ORIG$RUN_NCC_COV_ADJ$RUN_NCC_LITE_ORIG" == *1* ]]; then
  if [[ ! -f "NCCQuant/quantizers/ncc.py" ]]; then
    echo "[setup] NCCQuant not found -> cloning ..."
    git clone https://github.com/anhnda/NCCQuant.git NCCQuant
  fi
fi

# common args shared by every run
common_args=(
  --model-path "$MODEL"
  --method gptq
  --quantizer "$QUANTIZER"
  --device "$DEVICE"
  --n-calib "$N_CALIB"
  --max-length "$MAX_LEN"
  --calib-dataset "$CALIB_DS"
  --gptq-blocksize "$GPTQ_BLOCKSIZE"
  --gptq-percdamp "$GPTQ_PERCDAMP"
  --eval-samples "$EVAL_SAMPLES"
)
if [[ "$LM_EVAL" == "0" ]]; then
  common_args+=(--no-lm-eval)
fi

run_variant () {
  local tag="$1"; shift
  local outdir="$OUT_ROOT/${QUANTIZER}_${tag}"
  echo ""
  echo "================================================================"
  echo ">>> VARIANT: $tag  ->  $outdir"
  echo "================================================================"
  # main.py: quantize -> save_pretrained -> perplexity eval -> (lm-eval) ->
  # save run_summary.json -> cleanup_output_dir (deletes model, keeps summary).
  set +e
  python main.py "${common_args[@]}" --output-dir "$outdir" "$@" \
    2>&1 | tee "$OUT_ROOT/log_${QUANTIZER}_${tag}.txt"
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
    q = s.get("results", {}).get("quantized_model", {}) or s.get("quantized_model", {})
    # results layout: {dataset: {"perplexity": x, ...}}
    cols = []
    for ds, m in (q.items() if isinstance(q, dict) else []):
        if isinstance(m, dict) and "perplexity" in m:
            cols.append(f"{ds}={m['perplexity']:.4f}")
    qs = s.get("quantization", {})
    extra = []
    for k in ("method", "flips", "bias_before", "bias_after", "total_error"):
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

# 1) GPTQ base ----------------------------------------------------------------
if [[ "$RUN_BASE" == "1" ]]; then
  run_variant "gptq_base"
fi

# 2) GPTQ + NCC-Cov, baseline original (recommended) --------------------------
if [[ "$RUN_NCC_COV_ORIG" == "1" ]]; then
  run_variant "gptq_ncc_cov_orig" \
    --gptq-ncc --ncc-score cov --ncc-baseline original \
    --ncc-budget-p "$BUDGET_P" --ncc-cov-eps "$COV_EPS"
fi

# 3) GPTQ + NCC-Cov, baseline adjusted ----------------------------------------
if [[ "$RUN_NCC_COV_ADJ" == "1" ]]; then
  run_variant "gptq_ncc_cov_adj" \
    --gptq-ncc --ncc-score cov --ncc-baseline adjusted \
    --ncc-budget-p "$BUDGET_P" --ncc-cov-eps "$COV_EPS"
fi

# 4) GPTQ + NCC-Lite, baseline original (ablation) ----------------------------
if [[ "$RUN_NCC_LITE_ORIG" == "1" ]]; then
  run_variant "gptq_ncc_lite_orig" \
    --gptq-ncc --ncc-score lite --ncc-baseline original \
    --ncc-budget-p "$BUDGET_P"
fi

echo ""
echo "================================================================"
echo "All requested variants done. Models DELETED after eval; kept:"
echo "  $OUT_ROOT/<quantizer>_<tag>/run_summary.json   (has perplexity)"
echo "  $OUT_ROOT/log_${QUANTIZER}_*.txt               (full logs)"
echo ""
echo "Perplexity comparison:"
echo "----------------------------------------------------------------"
column -t -s$'\t' "$OUT_ROOT/perplexity_table.tsv" 2>/dev/null || cat "$OUT_ROOT/perplexity_table.tsv"
echo "================================================================"

MODEL=/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b \
QUANTIZER=nf4 \
DEVICE=cuda:0 \
bash run_gptq_ncc.sh