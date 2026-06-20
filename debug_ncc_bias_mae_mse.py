#!/usr/bin/env python3
"""
debug_ncc_bias_mae_mse.py
=========================

Standalone debug harness: run GPTVQ-1D on the FIRST FEW Linear layers of
Llama-3.1-8B, apply NCC on top, and measure whether

    bias  (B_hat = sum_j (mu^T e_j)^2)   -- the thing NCC targets
    MAE   (mean |W_q - W_fp|)
    MSE   (layer-MSE proxy: mean ((W_q-W_fp) scaled)^2 , and the
           activation-weighted R_l = mean over tokens of (x^T e)^2 )

go DOWN (or at least: MSE must NOT go up) after the NCC correction.

This mirrors `gptvq_rbvt_benchmark.py` exactly by REUSING its helpers
(_make_vq_quantizer, _gptvq_quant_result, _load_ncc_apply), so the codebook /
QuantResult / apply_ncc interfaces are identical to the real benchmark — no
re-implementation, no drift.

It only touches the first --max-layers decoder blocks and stops, so it is cheap.

Run it yourself, e.g.:

    cd RBVTQuant-main
    # NCCQuant must be cloned first (the benchmark does this for you):
    #   git clone https://github.com/anhnda/NCCQuant.git NCCQuant
    python debug_ncc_bias_mae_mse.py \
        --model-path meta-llama/Llama-3.1-8B \
        --device cuda:0 \
        --max-layers 2 \
        --n-calib 16 \
        --max-length 512 \
        --groupsize 128 \
        --wbits 4 \
        --ncc-budget-p 0.02 \
        --ncc-sweeps 1

NOTE: this script never runs anything on its own; you invoke it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Reuse the *real* benchmark helpers so interfaces never drift.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "GPTVQ"))  # gptq / modelutils / vq_quant live here

# transformers.Conv1D shim (same as the bash script does) ---------------------
import transformers  # noqa: E402

if not hasattr(transformers, "Conv1D"):
    from transformers.pytorch_utils import Conv1D

    transformers.Conv1D = Conv1D

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# GPTVQ upstream --------------------------------------------------------------
from gptq import GPTQ  # noqa: E402
from modelutils import find_layers  # noqa: E402

# Benchmark helpers (the ones we want to mirror) ------------------------------
import gptvq_rbvt_benchmark as B  # noqa: E402


# ---------------------------------------------------------------------------
# Calibration text (tiny wikitext2 slice) — kept minimal & dependency-light.
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
    # fallback: repeat a deterministic paragraph so the script still runs.
    base = (
        "The quick brown fox jumps over the lazy dog. "
        "Post-training quantization adapts the codebook to the weight density. "
    ) * 40
    return [base for _ in range(n)]


# ---------------------------------------------------------------------------
# Metrics
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
    """B_hat = sum_j (mu^T e_j)^2 ; e_j is column j of (W_q - W_fp).

    W is [out, in]; channel j = output row; input dim = mu dim.
    e_j = (W_q - W_fp)[j, :]  (length = in_features = len(mu)).
    """
    e = (W_q - W_fp).float()              # [out, in]
    b = e @ mu.float()                    # [out]  == per-output-channel mu^T e_j
    return (b * b).sum().item()


@torch.no_grad()
def activation_weighted_mse(W_fp, W_q, X) -> float:
    """R_l proxy: mean over tokens of (x^T e)^2 summed over output channels.

    X is [tokens, in], W is [out, in]. y_err = X @ e^T -> [tokens, out].
    Returns mean over tokens & channels of squared output error.
    """
    e = (W_q - W_fp).float()              # [out, in]
    yerr = X.float() @ e.t()              # [tokens, out]
    return (yerr * yerr).mean().item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-layers", type=int, default=2,
                    help="only quantize the first N decoder blocks")
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--wbits", type=int, default=4)
    ap.add_argument("--groupsize", type=int, default=128)
    ap.add_argument("--gptq-blocksize", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--kmeans-iters", type=int, default=20)
    ap.add_argument("--kmeans-init-method", default="mahalanobis")
    ap.add_argument("--assignment-chunk-size", type=int, default=4096)
    ap.add_argument("--kpp-n-subsample", type=int, default=-1)
    ap.add_argument("--sym", action="store_true")
    ap.add_argument("--include-m-step", action="store_true")
    ap.add_argument("--hessian-weighted-lookups", action="store_true")
    ap.add_argument("--true-sequential", action="store_true")
    # NCC knobs (mirror benchmark)
    ap.add_argument("--ncc-budget-p", type=float, default=0.02)
    ap.add_argument("--ncc-sweeps", type=int, default=1)
    ap.add_argument("--ncc-stop-eps", type=float, default=0.0)
    ap.add_argument("--ncc-use-james-stein", action="store_true")
    ap.add_argument("--row-chunk", type=int, default=1024)
    # diagnostics
    ap.add_argument("--diag-max-tokens", type=int, default=4096,
                    help="cap tokens kept for activation-weighted MSE")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device} model={args.model_path} "
          f"max_layers={args.max_layers} bits={args.wbits} gs={args.groupsize}")

    # ---- load model & tokenizer ----
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    model.eval()
    model.seqlen = args.max_length

    if not (hasattr(model, "model") and hasattr(model.model, "layers")):
        raise RuntimeError("expected a Llama-like model with model.model.layers")

    # ---- calibration: capture first decoder-block inputs (reuse helper) ----
    calib_texts = load_calib_texts(args.n_calib)
    batches = B._make_calibration_batches(tok, calib_texts, args.max_length)
    n_calib = min(args.n_calib, len(batches))
    print(f"[calib] using {n_calib} calibration batches")

    inps, outs, cache = B._capture_first_layer_inputs(
        model=model, batches=batches, device=device,
        nsamples=n_calib, seqlen=args.max_length,
    )

    layers = model.model.layers
    model.config.use_cache = False
    apply_ncc = B._load_ncc_apply()   # raises if NCCQuant not cloned

    n_layers = min(args.max_layers, len(layers))
    rows_report = []

    for layer_idx in range(n_layers):
        print(f"\n=== decoder block {layer_idx + 1}/{n_layers} ===")
        layer = layers[layer_idx].to(device)
        full = find_layers(layer)

        for names in B._sequential_groups(full, args.true_sequential):
            subset = {nm: full[nm] for nm in names}
            gptq = {}
            stat_sum, stat_sumsq, stat_count = {}, {}, {}
            xcache = {}  # key -> list of activation chunks for act-weighted MSE

            for nm, module in subset.items():
                gptq[nm] = GPTQ(module)
                gptq[nm].quantizer = B._make_vq_quantizer(args)

            def add_batch(nm):
                key = B._linear_key(layer_idx, nm)

                def hook(_m, inp, out):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    gptq[nm].add_batch(x.data, out.data)
                    xf = x.reshape(-1, x.shape[-1]).detach().float().cpu()
                    stat_sum[key] = stat_sum.get(
                        key, torch.zeros(xf.shape[-1])) + xf.sum(0)
                    stat_sumsq[key] = stat_sumsq.get(
                        key, torch.zeros(xf.shape[-1])) + (xf * xf).sum(0)
                    stat_count[key] = stat_count.get(key, 0) + xf.shape[0]
                    # keep a bounded sample of tokens for act-weighted MSE
                    cur = xcache.setdefault(key, [])
                    kept = sum(t.shape[0] for t in cur)
                    if kept < args.diag_max_tokens:
                        room = args.diag_max_tokens - kept
                        cur.append(xf[:room].clone())
                return hook

            handles = [m.register_forward_hook(add_batch(nm))
                       for nm, m in subset.items()]
            try:
                for s in range(n_calib):
                    outs[s] = B._layer_call(
                        layer, inps[s].unsqueeze(0), cache)
            finally:
                for h in handles:
                    h.remove()

            for nm, module in subset.items():
                key = B._linear_key(layer_idx, nm)
                W_fp = module.weight.data.detach().clone().float()

                # ---- GPTVQ-1D quantize (writes dequant weights into module) --
                t0 = time.time()
                gptq[nm].fasterquant(
                    blocksize=args.gptq_blocksize,
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=False,
                    static_groups=False,
                    include_m_step=args.include_m_step,
                    use_vq=True,
                    svd_rank=None,
                    hessian_weighted_lookups=args.hessian_weighted_lookups,
                    only_init_kmeans=False,
                )
                W_gptvq = module.weight.data.detach().float().clone()

                # ---- rebuild QuantResult exactly like the benchmark ----------
                qres = B._gptvq_quant_result(
                    W_dequant=W_gptvq,
                    assignments=gptq[nm].assignments,
                    centroids=gptq[nm].quantizer.all_centroids,
                    bits=args.wbits,
                    block_size=args.groupsize,
                )

                cnt = max(1, stat_count[key])
                mu = (stat_sum[key] / cnt).to(device)
                ex2 = (stat_sumsq[key] / cnt).to(device)
                sigma = (ex2 - mu * mu).clamp(min=0.0)

                # activation sample for act-weighted MSE
                X = torch.cat(xcache[key], dim=0).to(device) if key in xcache else None

                # ---- metrics BEFORE NCC (GPTVQ only) -------------------------
                m_before = weight_metrics(W_fp, W_gptvq)
                bias_before = bias_metric(W_fp, W_gptvq, mu)
                awmse_before = (activation_weighted_mse(W_fp, W_gptvq, X)
                                if X is not None else float("nan"))

                # ---- apply NCC (same call the benchmark uses) ----------------
                W_corr, stats = apply_ncc(
                    W_fp=W_fp.to(device),
                    qres=qres,
                    mu=mu,
                    budget_p=args.ncc_budget_p,
                    use_james_stein=args.ncc_use_james_stein,
                    mu_var=sigma,
                    row_chunk=args.row_chunk,
                )
                W_corr = W_corr.float()

                # ---- metrics AFTER NCC ---------------------------------------
                m_after = weight_metrics(W_fp, W_corr)
                bias_after = bias_metric(W_fp, W_corr, mu)
                awmse_after = (activation_weighted_mse(W_fp, W_corr, X)
                               if X is not None else float("nan"))

                flips = int(getattr(stats, "flips", -1))
                dt = time.time() - t0

                # ---- CROSS-CHECK: external metrics vs NCC-internal stats ------
                # (1) qres.W_dequant must equal the GPTVQ weights we passed in.
                #     If sort/remap in _gptvq_quant_result changed values, the
                #     residual e NCC sees != the residual we measure -> bug.
                qres_dq = qres.W_dequant.float()
                dq_diff = (qres_dq - W_gptvq).abs().max().item()
                assert dq_diff < 1e-4, (
                    f"[{key}] qres.W_dequant disagrees with W_gptvq "
                    f"(max|diff|={dq_diff:.3e}). _gptvq_quant_result may have "
                    f"altered the codebook projection (sort/remap)."
                )

                # (2) NCC-internal bias_before/after must match what we compute
                #     externally with the SAME mu and W_fp. A mismatch means NCC
                #     defines bias on a different residual/normalisation than
                #     B_hat = sum_j (mu^T e_j)^2 -> the printed descent would not
                #     correspond to the tensor that landed in module.weight.
                s_bb = getattr(stats, "bias_before", None)
                s_ba = getattr(stats, "bias_after", None)

                def _close(a, b, rtol=1e-2, atol=1e-8):
                    if a is None:
                        return True  # field absent -> skip silently
                    a = float(a)
                    return abs(a - b) <= atol + rtol * max(abs(a), abs(b))

                if s_bb is not None:
                    assert _close(s_bb, bias_before), (
                        f"[{key}] NCC stats.bias_before={float(s_bb):.6e} != "
                        f"external B_hat(W_gptvq)={bias_before:.6e}. NCC's bias "
                        f"definition differs from sum_j (mu^T e_j)^2 "
                        f"(check mu normalisation / per-channel vs summed)."
                    )
                if s_ba is not None:
                    assert _close(s_ba, bias_after), (
                        f"[{key}] NCC stats.bias_after={float(s_ba):.6e} != "
                        f"external B_hat(W_corr)={bias_after:.6e}. The reported "
                        f"post-correction bias does not match the corrected "
                        f"weights actually returned."
                    )

                # (3) Thm 3 invariant, checked on OUR measurement (not NCC's):
                #     bias must not increase. This is the load-bearing assert.
                assert bias_after <= bias_before * (1 + 1e-6) + 1e-12, (
                    f"[{key}] BIAS INCREASED {bias_before:.6e} -> {bias_after:.6e}. "
                    f"Theorem 3 / no-overshoot VIOLATED on the realised tensor."
                )

                def pct(a, b):
                    return (b - a) / a * 100.0 if a not in (0.0, float("nan")) else float("nan")

                row = {
                    "layer": key,
                    "flips": flips,
                    "bias_before": bias_before, "bias_after": bias_after,
                    "bias_d%": pct(bias_before, bias_after),
                    "mae_before": m_before["mae"], "mae_after": m_after["mae"],
                    "mae_d%": pct(m_before["mae"], m_after["mae"]),
                    "mse_before": m_before["mse"], "mse_after": m_after["mse"],
                    "mse_d%": pct(m_before["mse"], m_after["mse"]),
                    "awmse_before": awmse_before, "awmse_after": awmse_after,
                    "awmse_d%": pct(awmse_before, awmse_after),
                    "time_s": dt,
                }
                rows_report.append(row)

                # write back (so subsequent layers see corrected weights)
                module.weight.data = W_corr.to(module.weight.data.dtype)

                print(
                    f"  {key:<28} flips={flips:>6} | "
                    f"bias {bias_before:.4e}->{bias_after:.4e} ({row['bias_d%']:+.2f}%) | "
                    f"MAE {m_before['mae']:.4e}->{m_after['mae']:.4e} ({row['mae_d%']:+.2f}%) | "
                    f"MSE {m_before['mse']:.4e}->{m_after['mse']:.4e} ({row['mse_d%']:+.2f}%) | "
                    f"awMSE {awmse_before:.4e}->{awmse_after:.4e} ({row['awmse_d%']:+.2f}%)"
                )

        # propagate this block's output to next block's input
        layer = layers[layer_idx]
        for s in range(n_calib):
            inps[s] = outs[s]
        layers[layer_idx] = layer.cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- summary + sanity checks ----
    print("\n================ SUMMARY ================")
    n_bias_down = sum(r["bias_after"] <= r["bias_before"] + 1e-12 for r in rows_report)
    n_mse_up = sum(r["mse_after"] > r["mse_before"] * (1 + 1e-6) for r in rows_report)
    n_awmse_up = sum(
        (r["awmse_after"] == r["awmse_after"])  # not NaN
        and r["awmse_after"] > r["awmse_before"] * (1 + 1e-6)
        for r in rows_report
    )
    n = len(rows_report)
    print(f"layers checked          : {n}")
    print(f"bias  not increased     : {n_bias_down}/{n}   (Thm 3 -> expect {n}/{n})")
    print(f"weight-MSE INCREASED    : {n_mse_up}/{n}   (expect 0; NCC may raise raw weight-MSE)")
    print(f"act-weighted MSE up     : {n_awmse_up}/{n}   (Thm 4 target: expect 0)")
    if n_bias_down < n:
        print("  !! BIAS WENT UP on some layer -> Theorem 3 / no-overshoot VIOLATED. Bug.")
    if n_awmse_up > 0:
        print("  ?? act-weighted MSE rose somewhere -> check budget_p / gap / Sigma cost (Thm 4).")
    print("=========================================")
    print("\nNote on the two MSE columns:")
    print("  * weight-MSE (mean (W_q-W)^2) is NOT what NCC protects; a complementary")
    print("    flip moves a weight by one full gap, so raw weight-MSE CAN rise. That is")
    print("    expected and not a violation by itself.")
    print("  * act-weighted MSE (mean (x^T e)^2) is the R_l proxy Theorem 4 bounds.")
    print("    THIS is the one that should not increase. Watch the 'act-weighted MSE up'")
    print("    counter above.")


if __name__ == "__main__":
    main()