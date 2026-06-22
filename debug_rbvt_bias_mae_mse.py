#!/usr/bin/env python3
"""
Debug GPTVQ-1D + RBVT post-block on the first Llama-like decoder blocks.

The script compares a plain GPTVQ-1D layer result against the same layer
quantized with RBVT post-block correction. It reuses the real benchmark helper
so the post-block path is exactly the one used for full evaluation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "GPTVQ"))

import transformers  # noqa: E402

if not hasattr(transformers, "Conv1D"):
    from transformers.pytorch_utils import Conv1D

    transformers.Conv1D = Conv1D

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from calibration_utils import load_calibration_data  # noqa: E402
from gptq import GPTQ  # noqa: E402
from modelutils import find_layers  # noqa: E402
from runtime_utils import load_runtime_env, resolve_hf_token  # noqa: E402

import gptvq_rbvt_benchmark as B  # noqa: E402


@torch.no_grad()
def weight_metrics(W_fp: torch.Tensor, W_q: torch.Tensor) -> dict:
    e = (W_q - W_fp).float()
    return {
        "mae": float(e.abs().mean().item()),
        "mse": float(e.square().mean().item()),
        "max_abs": float(e.abs().max().item()),
    }


@torch.no_grad()
def bias_metric(W_fp: torch.Tensor, W_q: torch.Tensor, mu: torch.Tensor) -> float:
    e = (W_q - W_fp).float()
    b = e @ mu.float()
    return float(b.square().sum().item())


@torch.no_grad()
def activation_weighted_mse(W_fp: torch.Tensor, W_q: torch.Tensor, X: torch.Tensor) -> float:
    e = (W_q - W_fp).float()
    yerr = X.float() @ e.t()
    return float(yerr.square().mean().item())


def pct(before: float, after: float) -> float:
    if before != before or after != after or before == 0.0:
        return float("nan")
    return (after - before) / before * 100.0


def build_parser():
    ap = argparse.ArgumentParser(description="Debug GPTVQ-1D + RBVT post-block metrics")
    ap.add_argument("--model-path", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-layers", type=int, default=2)
    ap.add_argument("--n-calib", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--calib-dataset", choices=["c4", "wikitext2"], default="c4")
    ap.add_argument("--calibration-cache-dir", default="./calibration_cache")
    ap.add_argument("--wbits", type=int, default=4, choices=[3, 4])
    ap.add_argument("--groupsize", type=int, default=128)
    ap.add_argument("--gptq-blocksize", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=0.01)
    ap.add_argument("--kmeans-iters", type=int, default=20)
    ap.add_argument("--kmeans-init-method", choices=["cdf", "kpp", "mahalanobis"], default="mahalanobis")
    ap.add_argument("--assignment-chunk-size", type=int, default=4096)
    ap.add_argument("--kpp-n-subsample", type=int, default=10000)
    ap.add_argument("--sym", action="store_true", default=False)
    ap.add_argument("--include-m-step", action="store_true", default=False)
    ap.add_argument("--hessian-weighted-lookups", action="store_true", default=True)
    ap.add_argument("--no-hessian-weighted-lookups", dest="hessian_weighted_lookups", action="store_false")
    ap.add_argument("--true-sequential", action="store_true", default=True)
    ap.add_argument("--no-true-sequential", dest="true_sequential", action="store_false")
    ap.add_argument("--row-chunk", type=int, default=1024)
    ap.add_argument("--rbvt-lambda", type=float, default=1.0)
    ap.add_argument("--rbvt-topk", type=int, default=0)
    ap.add_argument("--gap-floor", type=float, default=1e-8)
    ap.add_argument("--strict-descent", action="store_true", default=True)
    ap.add_argument("--allow-overshoot", dest="strict_descent", action="store_false")
    ap.add_argument("--diag-max-tokens", type=int, default=4096)
    return ap


def main():
    load_runtime_env()
    args = build_parser().parse_args()
    if args.groupsize != args.gptq_blocksize:
        raise ValueError("RBVT post-block debug currently requires --groupsize == --gptq-blocksize.")
    if args.include_m_step:
        raise ValueError("RBVT post-block debug requires no M-step; omit --include-m-step.")

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    hf_token = resolve_hf_token()
    print(
        f"[setup] model={args.model_path} device={device} bits={args.wbits} "
        f"groupsize={args.groupsize} block={args.gptq_blocksize}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32,
        "trust_remote_code": True,
        "token": hf_token,
        "low_cpu_mem_usage": True,
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    model.eval()
    model.seqlen = args.max_length

    calib_texts = load_calibration_data(
        dataset_name=args.calib_dataset,
        tokenizer=tokenizer,
        n_samples=args.n_calib,
        seqlen=args.max_length,
        seed=42,
        cache_dir=args.calibration_cache_dir,
    )
    batches = B._make_calibration_batches(tokenizer, calib_texts, args.max_length)
    n_calib = min(args.n_calib, len(batches))
    print(f"[calib] using {n_calib} batches from {args.calib_dataset}")

    inps, outs, cache = B._capture_first_layer_inputs(
        model=model,
        batches=batches,
        device=device,
        nsamples=n_calib,
        seqlen=args.max_length,
    )

    layers = model.model.layers
    use_cache = model.config.use_cache
    model.config.use_cache = False
    rows = []

    for layer_idx in range(min(args.max_layers, len(layers))):
        print(f"\n=== decoder block {layer_idx + 1}/{min(args.max_layers, len(layers))} ===")
        layer = layers[layer_idx].to(device)
        full = find_layers(layer)

        for names in B._sequential_groups(full, args.true_sequential):
            subset = {name: full[name] for name in names}
            gptq = {}
            stat_sum: dict[str, torch.Tensor] = {}
            stat_sumsq: dict[str, torch.Tensor] = {}
            stat_count: dict[str, int] = {}
            xcache: dict[str, list[torch.Tensor]] = {}

            for name, module in subset.items():
                gptq[name] = GPTQ(module)
                gptq[name].quantizer = B._make_vq_quantizer(args)

            def add_batch(name):
                key = B._linear_key(layer_idx, name)

                def hook(_module, inp, out):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    gptq[name].add_batch(x.data, out.data)
                    xf = x.reshape(-1, x.shape[-1]).detach().float().cpu()
                    stat_sum[key] = stat_sum.get(key, torch.zeros(xf.shape[-1])) + xf.sum(dim=0)
                    stat_sumsq[key] = stat_sumsq.get(key, torch.zeros(xf.shape[-1])) + xf.square().sum(dim=0)
                    stat_count[key] = stat_count.get(key, 0) + xf.shape[0]
                    chunks = xcache.setdefault(key, [])
                    kept = sum(chunk.shape[0] for chunk in chunks)
                    if kept < args.diag_max_tokens:
                        chunks.append(xf[: args.diag_max_tokens - kept].clone())

                return hook

            handles = [module.register_forward_hook(add_batch(name)) for name, module in subset.items()]
            try:
                for sample_idx in range(n_calib):
                    outs[sample_idx] = B._layer_call(layer, inps[sample_idx].unsqueeze(0), cache)
            finally:
                for handle in handles:
                    handle.remove()

            h_snapshots = {name: gptq[name].H.detach().clone() for name in subset}

            for name, module in subset.items():
                key = B._linear_key(layer_idx, name)
                W_fp = module.weight.data.detach().clone().float()
                count = max(1, stat_count[key])
                mu = (stat_sum[key] / count).to(device)
                ex2 = (stat_sumsq[key] / count).to(device)
                sigma = (ex2 - mu * mu).clamp(min=0.0)
                X = torch.cat(xcache[key], dim=0).to(device) if key in xcache else None

                tick = time.time()
                gptq[name].fasterquant(
                    blocksize=args.gptq_blocksize,
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=False,
                    static_groups=False,
                    include_m_step=False,
                    use_vq=True,
                    svd_rank=None,
                    hessian_weighted_lookups=args.hessian_weighted_lookups,
                    only_init_kmeans=False,
                )
                W_gptvq = module.weight.data.detach().float().clone()

                module.weight.data = W_fp.to(module.weight.data.dtype)
                rbvt_gptq = GPTQ(module)
                rbvt_gptq.H = h_snapshots[name].to(device).clone()
                rbvt_gptq.quantizer = B._make_vq_quantizer(args)
                stats = B._gptvq_fasterquant_rbvt_post_block(
                    rbvt_gptq,
                    args=args,
                    mu=mu,
                    sigma=sigma,
                )
                W_rbvt = module.weight.data.detach().float().clone()

                m_before = weight_metrics(W_fp, W_gptvq)
                m_after = weight_metrics(W_fp, W_rbvt)
                bias_before = bias_metric(W_fp.to(device), W_gptvq.to(device), mu)
                bias_after = bias_metric(W_fp.to(device), W_rbvt.to(device), mu)
                awmse_before = activation_weighted_mse(W_fp.to(device), W_gptvq.to(device), X) if X is not None else float("nan")
                awmse_after = activation_weighted_mse(W_fp.to(device), W_rbvt.to(device), X) if X is not None else float("nan")
                dt = time.time() - tick

                row = {
                    "layer": key,
                    "flips": stats["flips"],
                    "candidates": stats["candidates"],
                    "objective_before": stats["objective_before"],
                    "objective_after": stats["objective_after"],
                    "variance_increase": stats["variance_increase"],
                    "bias_before": bias_before,
                    "bias_after": bias_after,
                    "mae_before": m_before["mae"],
                    "mae_after": m_after["mae"],
                    "mse_before": m_before["mse"],
                    "mse_after": m_after["mse"],
                    "awmse_before": awmse_before,
                    "awmse_after": awmse_after,
                    "time_s": dt,
                }
                rows.append(row)
                print(
                    f"  {key:<28} flips={row['flips']:>7} cand={row['candidates']:>9} | "
                    f"obj {row['objective_before']:.4e}->{row['objective_after']:.4e} | "
                    f"bias {bias_before:.4e}->{bias_after:.4e} ({pct(bias_before, bias_after):+.2f}%) | "
                    f"MAE {m_before['mae']:.4e}->{m_after['mae']:.4e} ({pct(m_before['mae'], m_after['mae']):+.2f}%) | "
                    f"MSE {m_before['mse']:.4e}->{m_after['mse']:.4e} ({pct(m_before['mse'], m_after['mse']):+.2f}%) | "
                    f"awMSE {awmse_before:.4e}->{awmse_after:.4e} ({pct(awmse_before, awmse_after):+.2f}%)"
                )

                rbvt_gptq.free()
                del W_fp, W_gptvq, W_rbvt, mu, sigma
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        for sample_idx in range(n_calib):
            outs[sample_idx] = B._layer_call(layer, inps[sample_idx].unsqueeze(0), cache)
        layers[layer_idx] = layer.cpu()
        inps, outs = outs, inps
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    model.config.use_cache = use_cache

    print("\n================ RBVT DEBUG SUMMARY ================")
    n = len(rows)
    obj_ok = sum(row["objective_after"] <= row["objective_before"] * (1 + 1e-6) + 1e-12 for row in rows)
    bias_down = sum(row["bias_after"] <= row["bias_before"] * (1 + 1e-6) + 1e-12 for row in rows)
    awmse_down = sum(
        row["awmse_after"] == row["awmse_after"]
        and row["awmse_after"] <= row["awmse_before"] * (1 + 1e-6) + 1e-12
        for row in rows
    )
    print(f"layers checked          : {n}")
    print(f"objective not increased : {obj_ok}/{n}")
    print(f"final bias not increased: {bias_down}/{n}")
    print(f"final awMSE not increased: {awmse_down}/{n}")
    print("====================================================")


if __name__ == "__main__":
    main()
