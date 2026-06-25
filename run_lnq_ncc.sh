#!/usr/bin/env bash
# =============================================================================
# run_lnq_ncc.sh
# -----------------------------------------------------------------------------
# Quantize a model with the LNQ backbone (GuidedQuant layer-wise non-uniform
# quantization, real train_least_squares) + NCC first-moment correction, and
# produce full quantized models for the relevant correction modes:
#   1. LNQ base                    (NCC budget 0 -> no flips; pure LNQ)
#   2. LNQ + NCC, full-greedy      (DEFAULT; exact full-Sigma gate, awMSE never up)
#   3. LNQ + NCC, full-screen      (cheap first-order full-Sigma screen)
#   4. LNQ + NCC, diag             (faithful published apply_ncc; optional Cor-2 guard)
#
# This mirrors run_gptvq_ncc.sh. NOTE vs run_gptvq_ncc.sh: the lnq path does NOT
# use --quantizer nf*, --groupsize, --kmeans-init-method, --include-m-step,
# --hessian-weighted-lookups, --true-sequential, --ncc-placement, --ncc-sweeps,
# --ncc-baseline (those belong to the gptvq / gptq paths). LNQ reads --wbits
# {3,4} for the codebook bit-width, --lnq-iters / --cd-cycles for the alternating
# minimisation, --guided for the GuidedQuant saliency Hessian, --score {cov,lite}
# + --cov-eps for the NCC ordering, --ncc-budget-p for the per-channel budget,
# and --mse-guard-mode {diag,full-screen,full-greedy} (+ --mse-guard in diag mode)
# for the corrector. --kmeans-iters seeds the SqueezeLLM-style LNQ init.
#
# Equivalent to the standalone debug command:
#   python debug_ncc_mse_lnq.py --model-path ... --wbits 3 --lnq-iters 3 \
#       --cd-cycles 4 --guided --ncc-budget-p 0.005 --score cov \
#       --mse-guard-mode full-greedy
# but run over the WHOLE model with save_pretrained + perplexity + lm-eval.
#
# Each run writes a full model via model.save_pretrained() to its own dir, then
# runs perplexity + lm-eval exactly like the other methods. Pick which variants
# to run with the RUN_* toggles.
#
# This script does NOT run anything on import; you invoke it. It assumes the
# patched main.py (with the lnq branch) + lnq_ncc_quant.py + debug_ncc_mse_lnq.py
# are in place, and that ./GuidedQuant and ./NCCQuant are cloned.
# =============================================================================
set -euo pipefail

# ---- paths / model ----------------------------------------------------------
REPO_DIR="${REPO_DIR:-$(pwd)}"               # dir containing main.py
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
DEVICE="${DEVICE:-cuda:0}"
OUT_ROOT="${OUT_ROOT:-./runs_lnq_ncc}"

# ---- LNQ codebook knobs -----------------------------------------------------
WBITS="${WBITS:-3}"                          # 3|4  (K = 2**WBITS levels per channel)
LNQ_ITERS="${LNQ_ITERS:-3}"                  # alternating-minimisation iters T
CD_CYCLES="${CD_CYCLES:-4}"                  # coordinate-descent cycles per assignment update
KMEANS_ITERS="${KMEANS_ITERS:-50}"           # Lloyd iters for the SqueezeLLM-style LNQ init
GUIDED="${GUIDED:-1}"                         # 1 -> --guided (GuidedQuant saliency Hessian), 0 -> plain X^T X
PERCDAMP="${PERCDAMP:-0.01}"                  # relative diag(H) dampening (PD-safety)
MAX_LAYERS="${MAX_LAYERS:-0}"                 # 0 = all decoder blocks; N = first N (debug/cheap)

# ---- calibration / NCC knobs ------------------------------------------------
N_CALIB="${N_CALIB:-128}"
MAX_LEN="${MAX_LEN:-2048}"
CALIB_DS="${CALIB_DS:-c4}"                    # c4|wikitext2
BUDGET_P="${BUDGET_P:-0.005}"                 # per-channel NCC budget fraction p
NCC_SCORE="${NCC_SCORE:-cov}"                 # cov|lite  (cov = |mu|/((sigma_ii+eps) g))
COV_EPS="${COV_EPS:-1e-6}"
JAMES_STEIN="${JAMES_STEIN:-0}"               # 1 -> --ncc-james-stein
ROW_CHUNK="${ROW_CHUNK:-1024}"
DIAG_MAX_TOKENS="${DIAG_MAX_TOKENS:-4096}"    # tokens kept per layer for awMSE diagnostics
# diag-mode-only Corollary-2 gate (gap < 2|e|); ignored by the full-* modes.
MSE_GUARD="${MSE_GUARD:-0}"                   # 1 -> add --mse-guard (diag mode only)

# ---- eval toggles (set to 0 to skip the heavy lm-eval during debugging) -----
LM_EVAL="${LM_EVAL:-1}"                       # 1 -> include lm-eval, 0 -> skip
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"

# ---- which variants to run --------------------------------------------------
RUN_BASE="${RUN_BASE:-1}"                     # LNQ, no correction (budget 0)
RUN_NCC_FULL_GREEDY="${RUN_NCC_FULL_GREEDY:-1}"   # DEFAULT mse-guard-mode
RUN_NCC_FULL_SCREEN="${RUN_NCC_FULL_SCREEN:-0}"
RUN_NCC_DIAG="${RUN_NCC_DIAG:-0}"             # faithful published apply_ncc

# -----------------------------------------------------------------------------
cd "$REPO_DIR"
mkdir -p "$OUT_ROOT"
# fresh comparison table
printf "variant\tperplexity...\tquant_stats\n" > "$OUT_ROOT/perplexity_table.tsv"

# GuidedQuant (real LNQ optimiser) is required for every variant.
if [[ ! -d "GuidedQuant" ]]; then
  echo "[setup] GuidedQuant not found -> cloning ..."
  git clone https://github.com/snu-mllab/GuidedQuant.git GuidedQuant
fi
# Clone NCCQuant if any variant is requested (base also bridges to QuantResult).
if [[ ! -f "NCCQuant/quantizers/ncc.py" ]]; then
  echo "[setup] NCCQuant not found -> cloning ..."
  git clone https://github.com/anhnda/NCCQuant.git NCCQuant
fi

# slug for output dirs / table rows
GUIDED_TAG=$([[ "$GUIDED" == "1" ]] && echo "g" || echo "p")   # g=guided, p=plain
SLUG="lnq${WBITS}b_${GUIDED_TAG}"

# common args shared by every run
common_args=(
  --model-path "$MODEL"
  --method lnq
  --device "$DEVICE"
  --wbits "$WBITS"
  --lnq-iters "$LNQ_ITERS"
  --cd-cycles "$CD_CYCLES"
  --kmeans-iters "$KMEANS_ITERS"
  --percdamp "$PERCDAMP"
  --max-layers "$MAX_LAYERS"
  --row-chunk "$ROW_CHUNK"
  --diag-max-tokens "$DIAG_MAX_TOKENS"
  --score "$NCC_SCORE"
  --cov-eps "$COV_EPS"
  --n-calib "$N_CALIB"
  --max-length "$MAX_LEN"
  --calib-dataset "$CALIB_DS"
  --eval-samples "$EVAL_SAMPLES"
)
[[ "$GUIDED"      == "1" ]] && common_args+=(--guided)
[[ "$JAMES_STEIN" == "1" ]] && common_args+=(--ncc-james-stein)
[[ "$LM_EVAL"     == "0" ]] && common_args+=(--no-lm-eval)

run_variant () {
  local tag="$1"; shift
  local outdir="$OUT_ROOT/${SLUG}_${tag}"
  echo ""
  echo "================================================================"
  echo ">>> VARIANT: $tag  ->  $outdir"
  echo "================================================================"
  # main.py: quantize -> save_pretrained -> perplexity eval -> (lm-eval) ->
  # save run_summary.json. Models are deleted after eval to save disk.
  set +e
  python main.py "${common_args[@]}" --output-dir "$outdir" "$@" \
    2>&1 | tee "$OUT_ROOT/log_${SLUG}_${tag}.txt"
  local rc=${PIPESTATUS[0]}
  set -e

  # Safety net: if main.py crashed before cleanup, delete model shards ourselves
  # so repeated variants don't fill the disk. Keep run_summary.json.
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
    for k in ("method", "bits", "guided", "mse_guard_mode", "flips",
              "bias_before", "bias_after"):
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

# optional diag-mode Corollary-2 gate
guard_flag=()
[[ "$MSE_GUARD" == "1" ]] && guard_flag+=(--mse-guard)

# 1) LNQ base -----------------------------------------------------------------
# budget_p = 0 admits no flips, so this is the uncorrected LNQ baseline. We route
# it through full-greedy (the default corrector) with an empty budget so the
# pipeline is identical to the corrected runs minus the flips.
if [[ "$RUN_BASE" == "1" ]]; then
  run_variant "base" \
    --mse-guard-mode full-greedy --ncc-budget-p 0.0
fi

# 2) LNQ + NCC, full-greedy (DEFAULT, exact full-Sigma gate) -------------------
if [[ "$RUN_NCC_FULL_GREEDY" == "1" ]]; then
  run_variant "ncc_full_greedy" \
    --mse-guard-mode full-greedy --ncc-budget-p "$BUDGET_P"
fi

# 3) LNQ + NCC, full-screen (cheap first-order full-Sigma screen) --------------
if [[ "$RUN_NCC_FULL_SCREEN" == "1" ]]; then
  run_variant "ncc_full_screen" \
    --mse-guard-mode full-screen --ncc-budget-p "$BUDGET_P"
fi

# 4) LNQ + NCC, diag (faithful published apply_ncc; optional Cor-2 guard) ------
if [[ "$RUN_NCC_DIAG" == "1" ]]; then
  run_variant "ncc_diag" \
    --mse-guard-mode diag --ncc-budget-p "$BUDGET_P" "${guard_flag[@]}"
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