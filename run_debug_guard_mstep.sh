#!/usr/bin/env bash
# =============================================================================
# run_debug_guard_mstep.sh
# -----------------------------------------------------------------------------
# A/B test for the question: "why does mse_guard (gap < 2|e|) still admit flips
# under GPTVQ, when nearest assignment should force gap >= 2|e| (zero flips)?"
#
# Hypothesis: GPTVQ's M-step shifts the centroids AFTER the per-weight index is
# fixed, so the final assignment Wq = grid_final[idx] is NOT the nearest level
# of the final grid. That breaks |e| <= gap/2, so gap < 2|e| becomes satisfiable
# and the guard lets flips through.
#
# Prediction:
#   M-step ON  + guard  -> MANY flips   (index != nearest of final grid)
#   M-step OFF + guard  -> ~ZERO flips  (index == nearest of final grid)
#
# Both runs use the SAME everything else. Reads the first --max-layers blocks,
# so it's cheap. This script does NOT run on import; you invoke it.
#
# Requires: debug_ncc_bias_mae_mse.py, gptvq_rbvt_benchmark.py, ./GPTVQ, ./NCCQuant.
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(pwd)}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B}"
DEVICE="${DEVICE:-cuda:0}"

MAX_LAYERS="${MAX_LAYERS:-2}"
N_CALIB="${N_CALIB:-16}"
MAX_LEN="${MAX_LEN:-512}"
WBITS="${WBITS:-3}"
GROUPSIZE="${GROUPSIZE:-128}"
KMEANS_ITERS="${KMEANS_ITERS:-100}"
BUDGET_P="${BUDGET_P:-0.005}"
SCORE="${SCORE:-cov}"
COV_EPS="${COV_EPS:-1e-2}"
LOGDIR="${LOGDIR:-./debug_guard_mstep}"

cd "$REPO_DIR"
mkdir -p "$LOGDIR"

if [[ ! -f "NCCQuant/quantizers/ncc.py" ]]; then
  echo "[setup] cloning NCCQuant ..."
  git clone https://github.com/anhnda/NCCQuant.git NCCQuant
fi
if [[ ! -d "GPTVQ" ]]; then
  echo "[setup] cloning GPTVQ ..."
  git clone https://github.com/Qualcomm-AI-research/gptvq.git GPTVQ
fi

common=(
  --model-path "$MODEL"
  --device "$DEVICE"
  --backend gptvq
  --max-layers "$MAX_LAYERS"
  --n-calib "$N_CALIB"
  --max-length "$MAX_LEN"
  --wbits "$WBITS"
  --groupsize "$GROUPSIZE"
  --kmeans-iters "$KMEANS_ITERS"
  --ncc-budget-p "$BUDGET_P"
  --ncc-sweeps 1
  --score "$SCORE"
  --cov-eps "$COV_EPS"
  --hessian-weighted-lookups
  --mse-guard
  --baseline original
)

run () {
  local tag="$1"; shift
  echo ""
  echo "================================================================"
  echo ">>> $tag"
  echo "================================================================"
  python debug_ncc_bias_mae_mse.py "${common[@]}" "$@" \
    2>&1 | tee "$LOGDIR/${tag}.log"
}

# A) M-step ON  + guard  -> expect MANY flips
run "guard_mstep_ON"  --include-m-step

# B) M-step OFF + guard  -> expect ~ZERO flips
run "guard_mstep_OFF"

echo ""
echo "================================================================"
echo "FLIP TOTALS (the load-bearing comparison)"
echo "----------------------------------------------------------------"
for tag in guard_mstep_ON guard_mstep_OFF; do
  total=$(grep -oE 'flips=[0-9]+' "$LOGDIR/${tag}.log" | grep -oE '[0-9]+' \
          | paste -sd+ - | bc 2>/dev/null || echo "n/a")
  echo "  ${tag}: total flips = ${total}"
done
echo ""
echo "If ON >> OFF (OFF ~ 0): confirms M-step breaks nearest, so gap<2|e| is"
echo "satisfiable only because the final index is not the nearest final level."
echo "If both large: assignment is non-nearest for another reason — inspect"
echo "_gptvq_quant_result (q_levels vs block_codebooks) and idx provenance."
echo "================================================================"