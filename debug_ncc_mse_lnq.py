#!/usr/bin/env python3
"""
debug_ncc_mse_lnq.py
====================

Standalone debug harness — the LNQ / GuidedQuant analogue of
`debug_ncc_bias_mae_mse.py`.

Instead of sitting NCC on top of GPTVQ-1D (the `gptvq` backend) or
NonUniformGPTQ + NF (the `nf` backend), this script sits NCC on top of **LNQ**
(the Layer-wise Non-uniform Quantization algorithm of GuidedQuant: closed-form
codebook update + cyclic-CD assignment update, Algorithm 2 in the paper),
optionally with the **GuidedQuant** end-loss saliency Hessian
H = X^T diag(s) X  (Eq. 7, g = 1) instead of the plain layer-wise H = X^T X.

It runs LNQ on the FIRST FEW Linear layers of a Llama-like model, applies NCC on
top, and measures whether

    bias  (B_hat = sum_j (mu^T e_j)^2)         -- the thing NCC targets
    MAE   (mean |W_q - W_fp|)
    MSE   (mean (W_q - W_fp)^2)
    awMSE (activation-weighted layer error, both vs the LNQ baseline and vs the
           ORIGINAL fp weights = true inference error)

go DOWN (or at least: bias must NOT go up; awMSE must NOT go up) after the NCC
correction.

It REUSES the *real* LNQ optimiser (`train_least_squares` from GuidedQuant's
`any_precision.quantization.layerwise_quantize`) and the *real* NCC corrector
(`apply_ncc` from NCCQuant), so the LNQ-codebook / QuantResult / apply_ncc
interfaces are identical to the production code — no re-implementation, no drift.
The bridge from LNQ's per-output-channel (centroids C [out, K], labels [out, in])
to NCC's block-wise QuantResult is the LNQ single-block analogue of
`_gptvq_quant_result`: one block spanning the whole input dim, block_codebooks =
sorted C[:, None, :], indices = labels remapped after the sort.

It only touches the first --max-layers decoder blocks and stops, so it is cheap.

This script never runs anything on its own; you invoke it, e.g.:

    cd RBVTQuant-main
    # NCCQuant + GuidedQuant must be importable (clone / on PYTHONPATH):
    #   git clone https://github.com/anhnda/NCCQuant.git NCCQuant
    python debug_ncc_mse_lnq.py \
        --model-path meta-llama/Llama-3.1-8B \
        --device cuda:0 \
        --max-layers 2 \
        --n-calib 16 \
        --max-length 512 \
        --wbits 3 \
        --lnq-iters 3 \
        --cd-cycles 4 \
        --guided \
        --ncc-budget-p 0.02 \
        --score cov
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Paths. NCCQuant supplies apply_ncc / QuantResult; GuidedQuant supplies the
# real LNQ optimiser. Both are imported from their on-disk checkouts so the
# interfaces never drift from production.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
NCC_ROOT = ROOT / "NCCQuant"
GUIDED_ROOT = ROOT / "GuidedQuant"
for p in (ROOT, GUIDED_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# transformers.Conv1D shim (same as the GPTVQ debug script) -------------------
import transformers  # noqa: E402

if not hasattr(transformers, "Conv1D"):
    from transformers.pytorch_utils import Conv1D

    transformers.Conv1D = Conv1D

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


# ---------------------------------------------------------------------------
# Lazy loaders for the two external pieces (mirrors gptvq benchmark's
# _load_ncc_apply so the same "clone first" error message surfaces).
# ---------------------------------------------------------------------------
_NCC_APPLY = None
_QUANT_RESULT = None


def _load_ncc():
    """Return (apply_ncc, QuantResult) from the NCCQuant checkout."""
    global _NCC_APPLY, _QUANT_RESULT
    if _NCC_APPLY is not None:
        return _NCC_APPLY, _QUANT_RESULT
    ncc_file = NCC_ROOT / "quantizers" / "ncc.py"
    base_file = NCC_ROOT / "quantizers" / "base_quantizer.py"
    if not ncc_file.exists():
        raise RuntimeError(
            "Missing ./NCCQuant. Clone upstream first: "
            "git clone https://github.com/anhnda/NCCQuant.git NCCQuant"
        )
    package_name = "_lnq_external_nccquant_quantizers"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(NCC_ROOT / "quantizers")]
        sys.modules[package_name] = package

    def _load(modname, path):
        spec = importlib.util.spec_from_file_location(f"{package_name}.{modname}", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load NCCQuant module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{package_name}.{modname}"] = module
        spec.loader.exec_module(module)
        return module

    base_mod = _load("base_quantizer", base_file)
    ncc_mod = _load("ncc", ncc_file)
    _NCC_APPLY = ncc_mod.apply_ncc
    _QUANT_RESULT = base_mod.QuantResult
    return _NCC_APPLY, _QUANT_RESULT


_LNQ_TLS = None


def _load_lnq():
    """Return the real LNQ optimiser train_least_squares from GuidedQuant.

    `layerwise_quantize.py` is loaded BY FILE PATH so that GuidedQuant's package
    __init__ chain (any_precision.modules -> AnyPrecisionForCausalLM -> analyzer
    -> splitted_models.qwen3 -> transformers.cache_utils.SlidingWindowCache) is
    never triggered. That chain breaks on some transformers versions and is dead
    weight for `train_least_squares`, which only needs `get_progress_bar`.

    We pre-seed sys.modules with light stubs for the two heavy module-level
    imports `layerwise_quantize.py` makes:
      - any_precision.analyzer.analyzer.ModelAnalyzer  (imported, unused here)
      - any_precision.quantization.utils.get_progress_bar (real, via tqdm; the
        upstream utils.py also imports numba, which we avoid)
    """
    global _LNQ_TLS
    if _LNQ_TLS is not None:
        return _LNQ_TLS

    lq_file = GUIDED_ROOT / "any_precision" / "quantization" / "layerwise_quantize.py"
    if not lq_file.exists():
        raise RuntimeError(
            f"Missing GuidedQuant LNQ source at {lq_file}. Expected the GuidedQuant "
            f"checkout under {GUIDED_ROOT}."
        )

    # --- stub package tree so the by-path module's `from any_precision...` lines
    #     resolve to our lightweight shims instead of the real (heavy) modules. ---
    def _ensure_pkg(qualname: str):
        if qualname in sys.modules:
            return sys.modules[qualname]
        mod = types.ModuleType(qualname)
        mod.__path__ = []  # mark as package so submodule imports are allowed
        sys.modules[qualname] = mod
        return mod

    # Only install stubs if the real ones are not already importable cleanly.
    if "any_precision.analyzer.analyzer" not in sys.modules:
        _ensure_pkg("any_precision")
        _ensure_pkg("any_precision.analyzer")
        analyzer_mod = types.ModuleType("any_precision.analyzer.analyzer")

        class _ModelAnalyzerStub:  # noqa: D401 - placeholder, never instantiated here
            """Stub: train_least_squares does not use ModelAnalyzer."""

        analyzer_mod.ModelAnalyzer = _ModelAnalyzerStub
        sys.modules["any_precision.analyzer.analyzer"] = analyzer_mod

    if "any_precision.quantization.utils" not in sys.modules:
        _ensure_pkg("any_precision")
        _ensure_pkg("any_precision.quantization")
        utils_mod = types.ModuleType("any_precision.quantization.utils")
        from tqdm import tqdm as _tqdm

        def _get_progress_bar(total, desc):
            return _tqdm(total=total, desc=desc,
                         bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        utils_mod.get_progress_bar = _get_progress_bar
        sys.modules["any_precision.quantization.utils"] = utils_mod

    spec = importlib.util.spec_from_file_location(
        "any_precision.quantization.layerwise_quantize", lq_file
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load LNQ module from {lq_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["any_precision.quantization.layerwise_quantize"] = module
    spec.loader.exec_module(module)

    _LNQ_TLS = module.train_least_squares
    return _LNQ_TLS


# ---------------------------------------------------------------------------
# Calibration text (tiny wikitext2 slice) — same fallback as the GPTVQ script.
# ---------------------------------------------------------------------------
def load_calib_texts(n: int) -> list[str]:
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts, i = [], 0
        while len(texts) < n and i < len(ds):
            t = ds[i]["text"].strip()
            if len(t) > 200:
                texts.append(t)
            i += 1
        if texts:
            return texts
    except Exception as exc:  # noqa: BLE001
        print(f"[calib] datasets unavailable ({exc}); using synthetic text.")
    base = (
        "The quick brown fox jumps over the lazy dog. "
        "Post-training quantization adapts the codebook to the weight density. "
    ) * 40
    return [base for _ in range(n)]


# ---------------------------------------------------------------------------
# Metrics (identical definitions to debug_ncc_bias_mae_mse.py).
# ---------------------------------------------------------------------------
@torch.no_grad()
def weight_metrics(W_fp: torch.Tensor, W_q: torch.Tensor) -> dict:
    e = (W_q - W_fp).float()
    return {
        "mae": e.abs().mean().item(),
        "mse": (e * e).mean().item(),
        "max_abs": e.abs().max().item(),
    }


@torch.no_grad()
def bias_metric(W_fp: torch.Tensor, W_q: torch.Tensor, mu: torch.Tensor) -> float:
    """B_hat = sum_j (mu^T e_j)^2 ; e_j is column j (= output row) of (W_q - W_fp)."""
    e = (W_q - W_fp).float()              # [out, in]
    b = e @ mu.float()                    # [out]
    return (b * b).sum().item()


@torch.no_grad()
def activation_weighted_mse(W_fp, W_q, X) -> float:
    """R_l proxy: mean over tokens & channels of (x^T e)^2. X [tokens, in]."""
    e = (W_q - W_fp).float()              # [out, in]
    yerr = X.float() @ e.t()              # [tokens, out]
    return (yerr * yerr).mean().item()


# ---------------------------------------------------------------------------
# LNQ -> QuantResult bridge (the single-block LNQ analogue of
# gptvq_rbvt_benchmark._gptvq_quant_result).
# ---------------------------------------------------------------------------
@torch.no_grad()
def _lnq_quant_result(
    *,
    QuantResult,
    W_dequant: torch.Tensor,   # [out, in]  dequantized LNQ weights
    labels: torch.Tensor,      # [out, in]  per-weight codeword index
    centroids: torch.Tensor,   # [out, K]   per-output-channel codebook
    bits: int,
):
    """Build a single-block (whole-row) QuantResult from LNQ output.

    LNQ assigns each output channel its own scalar codebook of size K = 2**bits,
    spanning the entire input dimension -> exactly ONE block of width in_features.
    Centroids are sorted ascending (NCC's neighbour query assumes a sorted grid)
    and the labels are remapped through the sort permutation so indices keep
    pointing at the same realised value.
    """
    device = W_dequant.device
    rows, cols = W_dequant.shape
    K = 2 ** bits
    centers = centroids.to(device=device, dtype=torch.float32)
    if centers.shape != (rows, K):
        raise RuntimeError(
            f"LNQ centroids have shape {tuple(centers.shape)}, expected {(rows, K)}"
        )

    sorted_centers, old_from_new = torch.sort(centers, dim=1)
    new_from_old = torch.empty_like(old_from_new)
    new_from_old.scatter_(
        dim=1,
        index=old_from_new,
        src=torch.arange(K, device=device).view(1, K).expand(rows, K),
    )
    idx = labels.to(device=device, dtype=torch.long).reshape(rows, cols)
    idx = torch.gather(new_from_old, dim=1, index=idx)

    block_codebooks = sorted_centers.unsqueeze(1)              # [out, 1, K]
    return QuantResult(
        W_dequant=W_dequant,
        indices=idx,
        q_levels=torch.linspace(-1.0, 1.0, K, device=device),
        block_scales=block_codebooks.abs().amax(dim=-1).clamp_min(1e-12),
        block_size=cols,                                       # one block = whole row
        block_codebooks=block_codebooks,
        block_zeros=None,
    )


# ---------------------------------------------------------------------------
# SqueezeLLM-style 1-D k-means initialisation for LNQ (per output channel).
# This is the "initial assignment / codebook" LNQ's alternating minimisation is
# seeded with (paper: LNQ initialises with SqueezeLLM assignments). Vectorised
# Lloyd over rows; centers sorted ascending.
# ---------------------------------------------------------------------------
@torch.no_grad()
def kmeans_init_per_channel(W: torch.Tensor, bits: int, n_iters: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (labels [out, in] uint8-range long, centroids [out, K] float32)."""
    device = W.device
    rows, cols = W.shape
    K = 2 ** bits
    qs = torch.linspace(0.0, 1.0, K, device=device, dtype=torch.float32)
    centers = torch.quantile(W.float(), qs, dim=1).t().contiguous()   # [out, K]
    # snap nearest-to-zero center to exactly 0 (matches learned_codebook init)
    zc = centers.abs().argmin(dim=1)
    centers[torch.arange(rows, device=device), zc] = 0.0
    centers, _ = torch.sort(centers, dim=1)

    Wb = W.float()
    assign = torch.zeros(rows, cols, dtype=torch.long, device=device)
    for _ in range(n_iters):
        d = (Wb.unsqueeze(-1) - centers.unsqueeze(1)).abs()          # [out, in, K]
        assign = d.argmin(dim=-1)
        del d
        new_centers = centers.clone()
        for k in range(K):
            mask = assign == k
            cnt = mask.sum(dim=1).clamp(min=1)
            summ = (Wb * mask).sum(dim=1)
            upd = summ / cnt
            has = mask.any(dim=1)
            new_centers[:, k] = torch.where(has, upd, centers[:, k])
        new_centers, _ = torch.sort(new_centers, dim=1)
        if torch.allclose(new_centers, centers, atol=1e-7):
            centers = new_centers
            break
        centers = new_centers
    d = (Wb.unsqueeze(-1) - centers.unsqueeze(1)).abs()
    assign = d.argmin(dim=-1)
    del d
    return assign, centers


# ---------------------------------------------------------------------------
# Summary printing (mirrors debug_ncc_bias_mae_mse.py).
# ---------------------------------------------------------------------------
def print_summary(rows_report):
    print("\n================ SUMMARY ================")
    n = len(rows_report)
    if n == 0:
        print("no layers checked")
        print("=========================================")
        return
    n_bias_down = sum(r["bias_after"] <= r["bias_before"] + 1e-12 for r in rows_report)
    n_awmse_up = sum(
        (r["awmse_after"] == r["awmse_after"])
        and r["awmse_after"] > r["awmse_before"] * (1 + 1e-6)
        for r in rows_report
    )
    n_awmse_orig_up = sum(
        (r["awmse_orig_after"] == r["awmse_orig_after"])
        and r["awmse_orig_after"] > r["awmse_orig_before"] * (1 + 1e-6)
        for r in rows_report
    )
    print(f"layers checked          : {n}")
    print(f"bias  not increased     : {n_bias_down}/{n}   (Thm 3 -> expect {n}/{n})")
    print(f"awMSE[base] up          : {n_awmse_up}/{n}   (vs LNQ baseline; Thm 4 target 0)")
    print(f"awMSE[orig] up          : {n_awmse_orig_up}/{n}   (vs ORIGINAL fp = true inference error)")
    if n_bias_down < n:
        print("  !! BIAS WENT UP on some layer -> Theorem 3 / no-overshoot VIOLATED. Bug.")
    if n_awmse_orig_up > 0:
        print("  ?? awMSE[orig] rose somewhere -> flips hurt true error. Try --mse-guard,")
        print("     lower --ncc-budget-p, or larger --cov-eps.")
    print("=========================================")

    up = [r for r in rows_report
          if (r["awmse_orig_after"] == r["awmse_orig_after"])
          and r["awmse_orig_after"] > r["awmse_orig_before"] * (1 + 1e-6)]
    if up:
        print("\n--- layers where awMSE[orig] increased ---")
        print(f"{'layer':<30}{'flips':>7}{'bias_d%':>10}{'awMSEo_d%':>11}{'awMSEo_abs_d':>15}")
        for r in sorted(up, key=lambda x: -x["awmse_orig_d%"]):
            abs_d = r["awmse_orig_after"] - r["awmse_orig_before"]
            print(f"{r['layer']:<30}{r['flips']:>7}{r['bias_d%']:>+10.2f}"
                  f"{r['awmse_orig_d%']:>+11.2f}{abs_d:>15.3e}")

    tot_bias_b = sum(r["bias_before"] for r in rows_report)
    tot_bias_a = sum(r["bias_after"] for r in rows_report)
    tot_flips = sum(r["flips"] for r in rows_report)
    valid = [r for r in rows_report if r["awmse_before"] == r["awmse_before"]]
    tot_aw_b = sum(r["awmse_before"] for r in valid)
    tot_aw_a = sum(r["awmse_after"] for r in valid)
    valid_o = [r for r in rows_report if r["awmse_orig_before"] == r["awmse_orig_before"]]
    tot_awo_b = sum(r["awmse_orig_before"] for r in valid_o)
    tot_awo_a = sum(r["awmse_orig_after"] for r in valid_o)
    print("\n--- NET (summed over checked layers) ---")
    print(f"total flips      : {tot_flips}")
    if tot_bias_b > 0:
        print(f"total bias       : {tot_bias_b:.4e} -> {tot_bias_a:.4e} "
              f"({(tot_bias_a-tot_bias_b)/tot_bias_b*100:+.2f}%)")
    if tot_aw_b > 0:
        net = (tot_aw_a - tot_aw_b) / tot_aw_b * 100
        v = "NET WIN" if tot_aw_a <= tot_aw_b else "NET REGRESSION"
        print(f"total act-wMSE[base] : {tot_aw_b:.4e} -> {tot_aw_a:.4e} ({net:+.2f}%)  [{v}]")
    if tot_awo_b > 0:
        neto = (tot_awo_a - tot_awo_b) / tot_awo_b * 100
        vo = "NET WIN" if tot_awo_a <= tot_awo_b else "NET REGRESSION"
        print(f"total act-wMSE[orig] : {tot_awo_b:.4e} -> {tot_awo_a:.4e} ({neto:+.2f}%)  [{vo}]  <- TRUE inference error")
    print("=========================================")


# ---------------------------------------------------------------------------
# Per-layer LNQ run.
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_lnq_layer(
    *,
    name: str,
    module: nn.Linear,
    train_least_squares,
    QuantResult,
    apply_ncc,
    H: torch.Tensor,            # [in, in] layer-wise (X^T X) or GuidedQuant Hessian
    mu: torch.Tensor,           # [in] activation mean
    sigma: torch.Tensor,        # [in] activation variance (diag of Sigma)
    X: torch.Tensor | None,     # [tokens, in] activation sample for awMSE
    cnt: int,
    args,
    device,
) -> dict:
    weight = module.weight.data.detach().clone().float()        # [out, in]
    out_features, in_features = weight.shape

    # ---- LNQ init (SqueezeLLM-style k-means per output channel) ----
    init_labels, init_centroids = kmeans_init_per_channel(
        weight.to(device), args.wbits, args.kmeans_iters
    )
    init_labels_np = init_labels.detach().cpu().numpy()
    init_centroids_np = init_centroids.detach().cpu().numpy()
    weight_np = weight.detach().cpu().numpy()

    # LNQ wants H shaped (num_groups, in, in). Single group (g would partition
    # OUTPUT channels in GuidedQuant; here H already encodes the chosen objective).
    H_lnq = H.detach().to(device).float().unsqueeze(0).cpu().numpy()   # [1, in, in]

    t0 = time.time()
    labels, C, _log = train_least_squares(
        weight_np,
        init_labels_np,
        init_centroids_np,
        H_lnq,
        num_iterations=args.lnq_iters,
        cd_cycles=args.cd_cycles,
    )
    dt_lnq = time.time() - t0

    labels = torch.from_numpy(labels).long().to(device)             # [out, in]
    C = torch.from_numpy(C).float().to(device)                      # [out, K]

    # Dequantize LNQ weights: W_lnq[i,j] = C[i, labels[i,j]]
    W_lnq = torch.gather(C, 1, labels).float()                     # [out, in]

    # ---- QuantResult bridge ----
    qres = _lnq_quant_result(
        QuantResult=QuantResult,
        W_dequant=W_lnq,
        labels=labels,
        centroids=C,
        bits=args.wbits,
    )

    # sanity: qres.W_dequant must equal the LNQ weights we measured
    dq_diff = (qres.W_dequant.float() - W_lnq).abs().max().item()
    assert dq_diff < 1e-4, (
        f"[{name}] qres.W_dequant disagrees with W_lnq (max|diff|={dq_diff:.3e}). "
        f"_lnq_quant_result sort/remap altered the realised values."
    )

    W_fp = weight.to(device)        # original fp (true inference reference)
    W_base = W_fp                   # NCC's first-moment target = original weights

    # ---- metrics BEFORE NCC (LNQ only) ----
    m_before = weight_metrics(W_base, W_lnq)
    bias_before = bias_metric(W_base, W_lnq, mu)
    awmse_before = activation_weighted_mse(W_base, W_lnq, X) if X is not None else float("nan")
    awmse_orig_before = activation_weighted_mse(W_fp, W_lnq, X) if X is not None else float("nan")

    # ---- apply NCC (same call the real pipeline uses) ----
    mu_var_js = (sigma / cnt) if args.ncc_use_james_stein else None
    W_corr, stats = apply_ncc(
        W_fp=W_base,
        qres=qres,
        mu=mu,
        budget_p=args.ncc_budget_p,
        use_james_stein=args.ncc_use_james_stein,
        mu_var=mu_var_js,
        row_chunk=args.row_chunk,
        score=args.score,
        sigma_ii=sigma if args.score == "cov" else None,
        cov_eps=args.cov_eps,
        mse_guard=args.mse_guard,
    )
    W_corr = W_corr.float()

    # ---- metrics AFTER NCC ----
    m_after = weight_metrics(W_base, W_corr)
    bias_after = bias_metric(W_base, W_corr, mu)
    awmse_after = activation_weighted_mse(W_base, W_corr, X) if X is not None else float("nan")
    awmse_orig_after = activation_weighted_mse(W_fp, W_corr, X) if X is not None else float("nan")
    flips = int(getattr(stats, "flips", -1))

    # ---- cross-check NCC-internal bias vs external B_hat (same baseline) ----
    s_bb = getattr(stats, "bias_before", None)
    s_ba = getattr(stats, "bias_after", None)

    def _close(a, b, rtol=1e-2, atol=1e-8):
        if a is None:
            return True
        a = float(a)
        return abs(a - b) <= atol + rtol * max(abs(a), abs(b))

    if s_bb is not None:
        assert _close(s_bb, bias_before), (
            f"[{name}] NCC stats.bias_before={float(s_bb):.6e} != external "
            f"B_hat={bias_before:.6e}. NCC bias definition differs from "
            f"sum_j (mu^T e_j)^2 on the same baseline."
        )
    if s_ba is not None:
        assert _close(s_ba, bias_after), (
            f"[{name}] NCC stats.bias_after={float(s_ba):.6e} != external "
            f"B_hat(W_corr)={bias_after:.6e}. Reported post-correction bias does "
            f"not match the corrected weights returned."
        )

    # ---- Thm 3 invariant on OUR measurement: bias must not increase ----
    assert bias_after <= bias_before * (1 + 1e-6) + 1e-12, (
        f"[{name}] BIAS INCREASED {bias_before:.6e} -> {bias_after:.6e}. "
        f"Theorem 3 / no-overshoot VIOLATED on the realised tensor."
    )

    def pct(a, b):
        return (b - a) / a * 100.0 if a not in (0.0, float("nan")) else float("nan")

    # write back so downstream blocks see corrected weights
    module.weight.data = W_corr.reshape(module.weight.shape).to(module.weight.dtype)

    print(
        f"  {name:<30} flips={flips:>7} | "
        f"bias {bias_before:.4e}->{bias_after:.4e} ({pct(bias_before,bias_after):+.2f}%) | "
        f"MSE {m_before['mse']:.4e}->{m_after['mse']:.4e} ({pct(m_before['mse'],m_after['mse']):+.2f}%) | "
        f"awMSE[base] {awmse_before:.4e}->{awmse_after:.4e} ({pct(awmse_before,awmse_after):+.2f}%) | "
        f"awMSE[orig] {awmse_orig_before:.4e}->{awmse_orig_after:.4e} "
        f"({pct(awmse_orig_before,awmse_orig_after):+.2f}%) | lnq={dt_lnq:.1f}s"
    )

    return {
        "layer": name, "flips": flips,
        "bias_before": bias_before, "bias_after": bias_after, "bias_d%": pct(bias_before, bias_after),
        "mae_before": m_before["mae"], "mae_after": m_after["mae"], "mae_d%": pct(m_before["mae"], m_after["mae"]),
        "mse_before": m_before["mse"], "mse_after": m_after["mse"], "mse_d%": pct(m_before["mse"], m_after["mse"]),
        "awmse_before": awmse_before, "awmse_after": awmse_after, "awmse_d%": pct(awmse_before, awmse_after),
        "awmse_orig_before": awmse_orig_before, "awmse_orig_after": awmse_orig_after,
        "awmse_orig_d%": pct(awmse_orig_before, awmse_orig_after),
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-layers", type=int, default=2,
                    help="only quantize the first N decoder blocks")
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--wbits", type=int, default=3,
                    help="LNQ codebook bit-width (K = 2**wbits levels per channel)")
    ap.add_argument("--kmeans-iters", type=int, default=50,
                    help="Lloyd iterations for the LNQ k-means initialisation")
    # LNQ knobs (mirror GuidedQuant Algorithm 2 / layerwise_nuq)
    ap.add_argument("--lnq-iters", type=int, default=3,
                    help="LNQ alternating-minimisation iterations T")
    ap.add_argument("--cd-cycles", type=int, default=4,
                    help="cyclic coordinate-descent cycles K per assignment update")
    ap.add_argument("--guided", action="store_true",
                    help="use the GuidedQuant Hessian H = X^T diag(s) X with "
                         "s = mean over output channels of (d loss / d z_j)^2 "
                         "(Eq. 7, g=1). Without this flag, H = X^T X (plain "
                         "layer-wise LNQ objective, Eq. 1).")
    ap.add_argument("--percdamp", type=float, default=0.01,
                    help="relative dampening added to diag(H) for PD-safety")
    # NCC knobs (identical to debug_ncc_bias_mae_mse.py)
    ap.add_argument("--ncc-budget-p", type=float, default=0.02)
    ap.add_argument("--ncc-use-james-stein", action="store_true")
    ap.add_argument("--row-chunk", type=int, default=1024)
    ap.add_argument("--score", choices=["lite", "cov"], default="lite",
                    help="NCC scoring rule. 'lite' = |mu|/g. "
                         "'cov' = |mu|/((sigma_ii+eps)*g) (NCC-Cov).")
    ap.add_argument("--cov-eps", type=float, default=1e-6)
    ap.add_argument("--mse-guard", action="store_true",
                    help="only admit flips with gap<2|e| (Cor-2 diagonal safety).")
    # diagnostics
    ap.add_argument("--diag-max-tokens", type=int, default=4096,
                    help="cap tokens kept for activation-weighted MSE")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        # train_least_squares / update_C / update_P hardcode cuda; warn loudly.
        print("[warn] LNQ (train_least_squares) uses CUDA internally; a non-CUDA "
              "device will fail. Provide a CUDA device.")
    print(f"[setup] backend=lnq device={device} model={args.model_path} "
          f"max_layers={args.max_layers} wbits={args.wbits} "
          f"guided={args.guided} lnq_iters={args.lnq_iters} cd_cycles={args.cd_cycles} "
          f"score={args.score} budget_p={args.ncc_budget_p} mse_guard={args.mse_guard}")

    apply_ncc, QuantResult = _load_ncc()
    train_least_squares = _load_lnq()

    # ---- load model & tokenizer ----
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    model.config.use_cache = False

    if not (hasattr(model, "model") and hasattr(model.model, "layers")):
        raise RuntimeError("expected a Llama-like model with model.model.layers")

    calib_texts = load_calib_texts(args.n_calib)

    blocks = model.model.layers
    n_blocks = min(args.max_layers, len(blocks))
    targets = []  # (name, module)
    for bi in range(n_blocks):
        for nm, m in blocks[bi].named_modules():
            if isinstance(m, nn.Linear):
                targets.append((f"layers.{bi}.{nm}", m))

    rows_report = []
    for name, module in targets:
        # ---- collect activation Hessian H = X^T X, mean mu, var sigma, sample X ----
        in_features = module.weight.shape[1]
        Hs = {"H": None, "s": None, "sq": None, "n": 0}
        xcache = []
        # GuidedQuant saliency: per-output-channel squared end-loss grad averaged
        # over channels (g=1). Captured via a backward hook on this layer's output.
        sal = {"acc": None, "n": 0}

        def fwd_hook(_m, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            xf = x.reshape(-1, x.shape[-1]).detach().float()        # [tok, in]
            HxtX = xf.t() @ xf                                      # [in, in]
            Hs["H"] = HxtX if Hs["H"] is None else Hs["H"] + HxtX
            s = xf.sum(0)
            Hs["s"] = s if Hs["s"] is None else Hs["s"] + s
            sq = (xf * xf).sum(0)
            Hs["sq"] = sq if Hs["sq"] is None else Hs["sq"] + sq
            Hs["n"] += xf.shape[0]
            kept = sum(t.shape[0] for t in xcache)
            if kept < args.diag_max_tokens:
                xcache.append(xf[: args.diag_max_tokens - kept].detach().cpu().clone())

        h_fwd = module.register_forward_hook(fwd_hook)

        # GuidedQuant saliency hook on the OUTPUT activation z = x @ W^T (Eq. 7).
        h_bwd = None
        if args.guided:
            def out_fwd_hook(_m, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                if out.requires_grad:
                    out.retain_grad()
                    _m._lnq_dbg_out = out
            def out_bwd_hook(_m, grad_in, grad_out):
                g = grad_out[0] if isinstance(grad_out, tuple) else grad_out
                if g is None:
                    return
                gf = g.reshape(-1, g.shape[-1]).detach().float()    # [tok, out]
                # s_j (one group) = mean over output channels of (dL/dz_j)^2,
                # broadcast to all rows. Here we accumulate the per-token squared
                # grad and reduce over channels at the end.
                acc = (gf * gf).sum(0)                              # [out]
                sal["acc"] = acc if sal["acc"] is None else sal["acc"] + acc
                sal["n"] += gf.shape[0]
            h_bwd = module.register_full_backward_hook(out_bwd_hook)

        try:
            for text in calib_texts[: args.n_calib]:
                enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length)
                enc = {k: v.to(device) for k, v in enc.items()}
                if args.guided:
                    labels_in = enc["input_ids"]
                    out = model(**enc, labels=labels_in, use_cache=False)
                    model.zero_grad(set_to_none=True)
                    out.loss.backward()
                else:
                    with torch.no_grad():
                        model(**enc, use_cache=False)
        finally:
            h_fwd.remove()
            if h_bwd is not None:
                h_bwd.remove()

        cnt = max(1, Hs["n"])
        mu = (Hs["s"] / cnt).to(device).float()
        sigma = ((Hs["sq"] / cnt).to(device).float() - mu * mu).clamp(min=0.0)
        H = Hs["H"].to(device).float()

        if args.guided and sal["acc"] is not None and sal["n"] > 0:
            # one-group GuidedQuant Hessian: H_k = X^T diag(s_bar) X with s_bar a
            # scalar mean saliency (g=1, averaged over all output channels).
            s_bar = float((sal["acc"].sum() / (sal["n"] * sal["acc"].numel())).item())
            # scale gradients by 1e3 like GuidedQuant to avoid underflow then it
            # cancels in the relative objective; here we just fold s_bar into H.
            H = H * max(s_bar, 1e-30)

        # PD-safety dampening (LNQ also dampens internally, this is belt+braces)
        diag_mean = torch.diag(H).mean().clamp(min=1e-12)
        H = H + (args.percdamp * diag_mean) * torch.eye(in_features, device=device)

        X = torch.cat(xcache, 0).to(device) if xcache else None

        row = run_lnq_layer(
            name=name, module=module,
            train_least_squares=train_least_squares,
            QuantResult=QuantResult, apply_ncc=apply_ncc,
            H=H, mu=mu, sigma=sigma, X=X, cnt=cnt, args=args, device=device,
        )
        rows_report.append(row)

        del H, Hs, xcache, sal
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print_summary(rows_report)


if __name__ == "__main__":
    main()