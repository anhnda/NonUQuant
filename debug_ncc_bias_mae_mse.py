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
def print_summary(rows_report):
    print("\n================ SUMMARY ================")
    n = len(rows_report)
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
    print(f"awMSE[base] up          : {n_awmse_up}/{n}   (vs chosen baseline; Thm 4 target 0)")
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


def run_nf_backend(args, device):
    """NonUniformGPTQ + NF/codebook quantizer path (the main.py pipeline).

    Quantizes only the first --max-layers DECODER BLOCKS' Linear layers (the rest
    stay FP), applies NCC on top per layer, and reports the same bias / awMSE
    metrics as the GPTVQ path. Cheap 2-layer debug of the NF3 path.
    """
    from nonuniform_gptq import NonUniformGPTQ, _load_ncc_apply
    from quantizers import get_quantizer

    apply_ncc = _load_ncc_apply()
    quantizer = get_quantizer(
        args.quantizer, nf_block_size=args.nf_block_size,
        nvfp4_block_size=16, cb_block_size=args.nf_block_size,
        n_iters=args.kmeans_iters, seed=42,
    )
    print(f"[nf] quantizer = {quantizer}")

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    model.config.use_cache = False

    calib_texts = load_calib_texts(args.n_calib)

    # collect the first N decoder blocks' Linear modules, in order
    blocks = model.model.layers
    n_blocks = min(args.max_layers, len(blocks))
    targets = []  # (name, module)
    for bi in range(n_blocks):
        for nm, m in blocks[bi].named_modules():
            if isinstance(m, nn.Linear):
                targets.append((f"layers.{bi}.{nm}", m))

    rows_report = []
    for name, module in targets:
        weight = module.weight.data.detach().clone().float()
        qres = quantizer.quantize(weight, row_chunk=args.row_chunk)
        capture = (args.baseline == "adjusted")
        gptq = NonUniformGPTQ(layer=module, qres=qres, reference_weight=weight)

        stat = {"s": None, "sq": None, "n": 0}
        xcache = []

        def hook(_m, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            gptq.add_batch(x.data, out.data)
            xf = x.reshape(-1, x.shape[-1]).detach().float()
            s = xf.sum(0)
            stat["s"] = s if stat["s"] is None else stat["s"] + s
            stat["sq"] = (xf * xf).sum(0) if stat["sq"] is None else stat["sq"] + (xf * xf).sum(0)
            stat["n"] += xf.shape[0]
            kept = sum(t.shape[0] for t in xcache)
            if kept < args.diag_max_tokens:
                xcache.append(xf[: args.diag_max_tokens - kept].detach().cpu().clone())

        h = module.register_forward_hook(hook)
        try:
            for i, text in enumerate(calib_texts[: args.n_calib]):
                enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length)
                enc = {k: v.to(device) for k, v in enc.items()}
                model(**enc, use_cache=False)
        finally:
            h.remove()

        t0 = time.time()
        err, _ = gptq.fasterquant(
            blocksize=args.gptq_blocksize, percdamp=args.percdamp,
            actorder=False, capture_w_assigned=capture,
        )
        W_gptq = module.weight.data.detach().float().clone()

        cnt = max(1, stat["n"])
        mu = (stat["s"] / cnt).to(device).float()
        sigma = ((stat["sq"] / cnt).to(device).float() - mu * mu).clamp(min=0.0)
        X = torch.cat(xcache, 0).to(device) if xcache else None

        if args.baseline == "adjusted":
            W_base = gptq.W_assigned
            if W_base is None:
                raise RuntimeError(f"{name}: W_assigned missing (capture failed).")
            W_base = W_base.to(device).float()
        else:
            W_base = weight.to(device)

        m_before = weight_metrics(W_base, W_gptq)
        bias_before = bias_metric(W_base, W_gptq, mu)
        awmse_before = activation_weighted_mse(W_base, W_gptq, X) if X is not None else float("nan")
        awmse_orig_before = activation_weighted_mse(weight.to(device), W_gptq, X) if X is not None else float("nan")

        qres_post = gptq.gptq_quant_result()
        mu_var_js = (sigma / cnt) if args.ncc_use_james_stein else None
        W_corr, stats = apply_ncc(
            W_fp=W_base, qres=qres_post, mu=mu,
            budget_p=args.ncc_budget_p, use_james_stein=args.ncc_use_james_stein,
            mu_var=mu_var_js, row_chunk=args.row_chunk,
            score=args.score, sigma_ii=sigma if args.score == "cov" else None,
            cov_eps=args.cov_eps, mse_guard=args.mse_guard,
        )
        W_corr = W_corr.float()

        m_after = weight_metrics(W_base, W_corr)
        bias_after = bias_metric(W_base, W_corr, mu)
        awmse_after = activation_weighted_mse(W_base, W_corr, X) if X is not None else float("nan")
        awmse_orig_after = activation_weighted_mse(weight.to(device), W_corr, X) if X is not None else float("nan")
        flips = int(getattr(stats, "flips", -1))

        assert bias_after <= bias_before * (1 + 1e-6) + 1e-12, (
            f"[{name}] BIAS INCREASED {bias_before:.6e}->{bias_after:.6e}. Thm 3 violated."
        )

        def pct(a, b):
            return (b - a) / a * 100.0 if a not in (0.0, float("nan")) else float("nan")

        rows_report.append({
            "layer": name, "flips": flips,
            "bias_before": bias_before, "bias_after": bias_after, "bias_d%": pct(bias_before, bias_after),
            "mse_before": m_before["mse"], "mse_after": m_after["mse"], "mse_d%": pct(m_before["mse"], m_after["mse"]),
            "awmse_before": awmse_before, "awmse_after": awmse_after, "awmse_d%": pct(awmse_before, awmse_after),
            "awmse_orig_before": awmse_orig_before, "awmse_orig_after": awmse_orig_after,
            "awmse_orig_d%": pct(awmse_orig_before, awmse_orig_after),
        })
        module.weight.data = W_corr.reshape(module.weight.shape).to(module.weight.dtype)
        gptq.free()
        print(
            f"  {name:<30} flips={flips:>7} | "
            f"bias {bias_before:.4e}->{bias_after:.4e} ({pct(bias_before,bias_after):+.2f}%) | "
            f"awMSE[base] {awmse_before:.4e}->{awmse_after:.4e} ({pct(awmse_before,awmse_after):+.2f}%) | "
            f"awMSE[orig] {awmse_orig_before:.4e}->{awmse_orig_after:.4e} ({pct(awmse_orig_before,awmse_orig_after):+.2f}%)"
        )

    print_summary(rows_report)


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
    ap.add_argument("--kmeans-iters", type=int, default=100)
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
    ap.add_argument("--score", choices=["lite", "cov"], default="lite",
                    help="NCC scoring rule. 'lite' = |mu|/g (covariance-free). "
                         "'cov' = |mu|/((sigma_ii+eps)*g) (NCC-Cov, the derived "
                         "diagonal bias-variance rule). 'cov' passes sigma_ii.")
    ap.add_argument("--cov-eps", type=float, default=1e-6)
    ap.add_argument("--mse-guard", action="store_true",
                    help="Only admit flips with gap<2|e| (Cor-2 diagonal safety): "
                         "each flip reduces both bias and diagonal awMSE. Expect "
                         "fewer flips, higher bias_after, but awMSE not increasing.")
    ap.add_argument("--baseline", choices=["original", "adjusted"],
                    default="original",
                    help="Which full-precision reference NCC corrects against. "
                         "'original' = layer.weight before quantization (end-to-end "
                         "first-moment target; matches the paper's stated target). "
                         "'adjusted' = error-feedback-adjusted weights at the moment "
                         "of nearest assignment (restores |e|<=g/2; on-top-of-GPTVQ "
                         "view). Bias is measured vs the chosen baseline; MAE/MSE/"
                         "awMSE are ALWAYS measured vs the original (true inference "
                         "error).")
    ap.add_argument("--backend", choices=["gptvq", "nf"], default="gptvq",
                    help="gptvq = GPTVQ-1D learned scalar codebook (k-means). "
                         "nf = NonUniformGPTQ + a NormalFloat/codebook quantizer "
                         "(the main.py path). Both apply NCC on top identically.")
    ap.add_argument("--quantizer", default="nf3",
                    help="(backend=nf only) quantizer name passed to get_quantizer, "
                         "e.g. nf3, nf4, nvfp4, codebook3, codebook4.")
    ap.add_argument("--nf-block-size", type=int, default=64)
    # diagnostics
    ap.add_argument("--diag-max-tokens", type=int, default=4096,
                    help="cap tokens kept for activation-weighted MSE")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[setup] backend={args.backend} device={device} model={args.model_path} "
          f"max_layers={args.max_layers} "
          f"{'quant='+args.quantizer if args.backend=='nf' else 'bits='+str(args.wbits)} "
          f"score={args.score} budget_p={args.ncc_budget_p} "
          f"baseline={args.baseline} mse_guard={args.mse_guard}")

    if args.backend == "nf":
        return run_nf_backend(args, device)
    # else: fall through to the existing GPTVQ path below

    # ---- load model & tokenizer ----
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
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
                    capture_w_assigned=(args.baseline == "adjusted"),
                )
                W_gptvq = module.weight.data.detach().float().clone()

                # Choose the baseline NCC corrects against.
                if args.baseline == "adjusted":
                    W_adj = getattr(gptq[nm], "W_assigned", None)
                    if W_adj is None:
                        raise RuntimeError(
                            "baseline=adjusted but gptq.W_assigned is None. "
                            "Is the patched GPTVQ/gptq.py on the path? "
                            "(capture_w_assigned support required)."
                        )
                    W_base = W_adj.detach().float().clone()
                else:
                    W_base = W_fp

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
                # bias + the *_self metrics use the chosen baseline W_base
                # (baseline N -> error vs N). awMSE_orig additionally tracks the
                # true inference error vs original W_fp so the gap between the
                # two columns exposes how much error-feedback NCC is undoing.
                # When baseline=original the two awMSE columns coincide.
                m_before = weight_metrics(W_base, W_gptvq)
                bias_before = bias_metric(W_base, W_gptvq, mu)
                awmse_before = (activation_weighted_mse(W_base, W_gptvq, X)
                                if X is not None else float("nan"))
                awmse_orig_before = (activation_weighted_mse(W_fp, W_gptvq, X)
                                     if X is not None else float("nan"))

                # ---- apply NCC (same call the benchmark uses) ----------------
                # mu_var for James-Stein must be variance OF THE MEAN estimate
                # (= activation variance / m), NOT raw activation variance, per
                # ncc.py docstring. sigma_ii (for NCC-Cov scoring) is the RAW
                # per-input-channel activation variance.
                mu_var_js = (sigma / cnt) if args.ncc_use_james_stein else None
                W_corr, stats = apply_ncc(
                    W_fp=W_base.to(device),
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

                # ---- metrics AFTER NCC ---------------------------------------
                m_after = weight_metrics(W_base, W_corr)
                bias_after = bias_metric(W_base, W_corr, mu)
                awmse_after = (activation_weighted_mse(W_base, W_corr, X)
                               if X is not None else float("nan"))
                awmse_orig_after = (activation_weighted_mse(W_fp, W_corr, X)
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
                        f"external B_hat(baseline)={bias_before:.6e}. NCC's bias "
                        f"definition differs from sum_j (mu^T e_j)^2 on the SAME "
                        f"baseline (check mu norm / per-channel vs summed)."
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
                    "awmse_orig_before": awmse_orig_before,
                    "awmse_orig_after": awmse_orig_after,
                    "awmse_orig_d%": pct(awmse_orig_before, awmse_orig_after),
                    "time_s": dt,
                }
                rows_report.append(row)

                # write back (so subsequent layers see corrected weights)
                module.weight.data = W_corr.to(module.weight.data.dtype)

                print(
                    f"  {key:<28} flips={flips:>6} | "
                    f"bias {bias_before:.4e}->{bias_after:.4e} ({row['bias_d%']:+.2f}%) | "
                    f"MSE {m_before['mse']:.4e}->{m_after['mse']:.4e} ({row['mse_d%']:+.2f}%) | "
                    f"awMSE[base] {awmse_before:.4e}->{awmse_after:.4e} ({row['awmse_d%']:+.2f}%) | "
                    f"awMSE[orig] {awmse_orig_before:.4e}->{awmse_orig_after:.4e} "
                    f"({row['awmse_orig_d%']:+.2f}%)"
                )

        # propagate this block's output to next block's input
        layer = layers[layer_idx]
        for s in range(n_calib):
            inps[s] = outs[s]
        layers[layer_idx] = layer.cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- summary + sanity checks ----
    print_summary(rows_report)


if __name__ == "__main__":
    main()