"""Run upstream GPTVQ-1D and GPTVQ-1D+RBVT on a Llama-like HF model.

This file intentionally imports Qualcomm-AI-research/gptvq as an external
checkout from ./GPTVQ. The quantization loop mirrors GPTVQ's llama.py, with the
minimum extra bookkeeping needed to convert 1D VQ centroids/assignments into
RBVT's scalar QuantResult format.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import random
import shutil
import sys
import time
import types
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
GPTVQ_ROOT = ROOT / "GPTVQ"
NCC_ROOT = ROOT / "NCCQuant"
if not GPTVQ_ROOT.exists():
    raise RuntimeError(
        "Missing ./GPTVQ. Clone upstream first: "
        "git clone https://github.com/Qualcomm-AI-research/gptvq.git GPTVQ"
    )
sys.path.insert(0, str(GPTVQ_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transformers  # noqa: E402

if not hasattr(transformers, "Conv1D"):
    from transformers.pytorch_utils import Conv1D  # noqa: E402

    transformers.Conv1D = Conv1D

from gptq import GPTQ  # type: ignore  # noqa: E402
from modelutils import find_layers  # type: ignore  # noqa: E402
from vq_quant import VQQuantizer, vq_quantize  # type: ignore  # noqa: E402

from calibration_utils import load_calibration_data  # noqa: E402
from eval_perplexity import RBVTSlidingWindowEvaluator  # noqa: E402
from lm_eval_runner import LMEvalHarnessRunner  # noqa: E402
from quantizers import apply_rbvt  # noqa: E402
from quantizers.base_quantizer import QuantResult  # noqa: E402
from runtime_utils import build_model_slug, load_runtime_env, resolve_hf_token  # noqa: E402


_NCC_APPLY = None


def _hf_device(device: str) -> torch.device:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"{device=} requested but CUDA is not available")
    return torch.device(device)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _linear_key(layer_idx: int, name: str) -> str:
    return f"model.layers.{layer_idx}.{name}"


def _layer_call(layer: nn.Module, hidden: torch.Tensor, cache: dict) -> torch.Tensor:
    return layer(hidden, **cache.get("layer_kwargs", {}))[0]


def _make_calibration_batches(tokenizer, texts: Iterable[str], seqlen: int) -> list[tuple[torch.Tensor]]:
    batches = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=seqlen,
        )
        batches.append((encoded["input_ids"],))
    return batches


def _capture_first_layer_inputs(model, batches, device: torch.device, nsamples: int, seqlen: int):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if getattr(model.model, "norm", None) is not None:
        model.model.norm = model.model.norm.to(device)
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    hidden_size = model.config.hidden_size
    inps = torch.zeros((nsamples, seqlen, hidden_size), dtype=dtype, device=device)
    cache = {"i": 0, "layer_kwargs": {}}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name):
            # nn.Module.__getattr__ handles registered submodules/params/buffers.
            # For anything else (e.g. Qwen2 reads decoder_layer.attention_type
            # before calling forward), delegate to the wrapped module so the
            # Catcher is transparent to the surrounding model code.
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.__dict__["_modules"]["module"], name)

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["layer_kwargs"] = dict(kwargs)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in batches[:nsamples]:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if getattr(model.model, "norm", None) is not None:
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return inps, torch.zeros_like(inps), cache


def _sequential_groups(full: dict[str, nn.Module], true_sequential: bool) -> list[list[str]]:
    if not true_sequential:
        return [[k for k in list(full.keys()) if "block_sparse_moe.gate" not in k]]
    groups = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
    return [[name for name in group if name in full] for group in groups if any(name in full for name in group)]


def _make_vq_quantizer(args) -> VQQuantizer:
    quantizer = VQQuantizer(
        vq_dim=1,
        columns_per_group=None,
        vq_scaling_blocksize=-1,
        vq_scaling_norm="max",
        vq_scaling_n_bits=4,
        vq_scaling_domain="log",
        kmeans_init_method=args.kmeans_init_method,
        assignment_chunk_size=args.assignment_chunk_size,
        kmeans_iters=args.kmeans_iters,
        codebook_bitwidth=None,
        quantize_per_codebook=True,
        quantize_during_kmeans=False,
        n_subsample=args.kpp_n_subsample,
    )
    quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False)
    return quantizer


def _gptvq_quant_result(
    *,
    W_dequant: torch.Tensor,
    assignments: list[list[torch.Tensor]],
    centroids: list[torch.Tensor],
    bits: int,
    block_size: int,
) -> QuantResult:
    if len(assignments) != len(centroids):
        raise RuntimeError(
            f"GPTVQ assignments/centroids mismatch: {len(assignments)} vs {len(centroids)}"
        )
    device = W_dequant.device
    rows, cols = W_dequant.shape
    n_blocks = len(centroids)
    K = 2**bits

    all_indices = []
    block_codebooks = torch.empty((rows, n_blocks, K), dtype=torch.float32, device=device)

    for block_idx, (block_assignments, block_centroids) in enumerate(zip(assignments, centroids)):
        centers = block_centroids.to(device=device, dtype=torch.float32).squeeze(-1)
        if centers.shape != (rows, K):
            raise RuntimeError(
                f"GPTVQ 1D centers for block {block_idx} have shape {tuple(centers.shape)}, "
                f"expected {(rows, K)}"
            )

        sorted_centers, old_from_new = torch.sort(centers, dim=1)
        new_from_old = torch.empty_like(old_from_new)
        new_from_old.scatter_(
            dim=1,
            index=old_from_new,
            src=torch.arange(K, device=device).view(1, K).expand(rows, K),
        )

        idx = torch.cat(
            [assignment.to(device=device, dtype=torch.long).reshape(rows, -1) for assignment in block_assignments],
            dim=1,
        )
        idx = torch.gather(new_from_old, dim=1, index=idx)
        all_indices.append(idx)
        block_codebooks[:, block_idx, :] = sorted_centers

    indices = torch.cat(all_indices, dim=1)
    if indices.shape != (rows, cols):
        raise RuntimeError(f"GPTVQ indices have shape {tuple(indices.shape)}, expected {(rows, cols)}")

    return QuantResult(
        W_dequant=W_dequant,
        indices=indices,
        q_levels=torch.linspace(-1.0, 1.0, K, device=device),
        block_scales=block_codebooks.abs().amax(dim=-1).clamp_min(1e-12),
        block_size=block_size,
        block_codebooks=block_codebooks,
        block_zeros=None,
    )


def _append_diagnostic_inputs(
    *,
    key: str,
    x: torch.Tensor,
    diagnostic_inputs: dict[str, list[torch.Tensor]],
    diagnostic_order: list[str],
    args,
):
    if args.diagnostic_layer_limit <= 0 or args.diagnostic_max_tokens <= 0:
        return
    if key not in diagnostic_inputs:
        if len(diagnostic_order) >= args.diagnostic_layer_limit:
            return
        diagnostic_inputs[key] = []
        diagnostic_order.append(key)

    current = sum(chunk.shape[0] for chunk in diagnostic_inputs[key])
    remaining = args.diagnostic_max_tokens - current
    if remaining <= 0:
        return
    x_flat = x.reshape(-1, x.shape[-1]).detach()
    diagnostic_inputs[key].append(x_flat[:remaining].to(device="cpu", dtype=torch.float16))


@torch.no_grad()
def _activation_error_metrics(
    *,
    key: str,
    X_cpu: torch.Tensor,
    W_fp: torch.Tensor,
    W_quant: torch.Tensor,
    variant: str,
) -> dict:
    device = W_quant.device
    X = X_cpu.to(device=device, dtype=torch.float32)
    diff = W_fp.to(device=device, dtype=torch.float32) - W_quant.to(device=device, dtype=torch.float32)
    Yerr = X.matmul(diff.t())
    metrics = {
        "layer": key,
        "variant": variant,
        "tokens": int(X.shape[0]),
        "mae": float(Yerr.abs().mean().item()),
        "mse": float(Yerr.square().mean().item()),
        "max_abs": float(Yerr.abs().max().item()),
    }
    del X, diff, Yerr
    return metrics


def _load_ncc_apply():
    global _NCC_APPLY
    if _NCC_APPLY is not None:
        return _NCC_APPLY
    ncc_file = NCC_ROOT / "quantizers" / "ncc.py"
    if not ncc_file.exists():
        raise RuntimeError(
            "Missing ./NCCQuant. Clone upstream first: "
            "git clone https://github.com/anhnda/NCCQuant.git NCCQuant"
        )
    package_name = "_rbvt_external_nccquant_quantizers"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(NCC_ROOT / "quantizers")]
        sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(f"{package_name}.ncc", ncc_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load NCCQuant module from {ncc_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{package_name}.ncc"] = module
    spec.loader.exec_module(module)
    _NCC_APPLY = module.apply_ncc
    return _NCC_APPLY


@torch.no_grad()
def _refresh_quant_indices_from_dequant(qres: QuantResult, W_dequant: torch.Tensor) -> QuantResult:
    if qres.block_codebooks is None:
        raise RuntimeError("GPTVQ-1D NCC sweeps require materialized block_codebooks.")
    device = W_dequant.device
    out_features, in_features = W_dequant.shape
    bs = qres.block_size
    n_blocks = qres.block_codebooks.shape[1]
    indices = torch.empty(out_features, in_features, dtype=torch.long, device=device)
    levels = qres.block_codebooks.to(device).float()
    W = W_dequant.float()
    for block_idx in range(n_blocks):
        c0 = block_idx * bs
        c1 = min(c0 + bs, in_features)
        if c0 >= c1:
            continue
        grid = levels[:, block_idx, :]
        diff = (W[:, c0:c1].unsqueeze(-1) - grid.unsqueeze(1)).abs()
        indices[:, c0:c1] = diff.argmin(dim=-1)
        del diff
    return QuantResult(
        W_dequant=W_dequant,
        indices=indices,
        q_levels=qres.q_levels,
        block_scales=qres.block_scales,
        block_size=qres.block_size,
        block_codebooks=qres.block_codebooks,
    )


@torch.no_grad()
def _apply_ncc_sweeps(
    *,
    W_fp: torch.Tensor,
    qres: QuantResult,
    mu: torch.Tensor,
    mu_var: torch.Tensor | None,
    args,
) -> tuple[torch.Tensor, dict]:
    apply_ncc = _load_ncc_apply()
    current = qres
    W_corr = qres.W_dequant
    history = []
    total_flips = 0
    first_bias_before = None
    final_bias_after = None

    # NCC scoring / safety knobs. mse_guard (Cor-2 diagonal safety) only reduces
    # awMSE when the diagonal activation variance is supplied, so it pairs with
    # score="cov" and sigma_ii=mu_var. These default off / "lite" when the args
    # are absent so older callers keep the previous behaviour.
    ncc_score = getattr(args, "ncc_score", "lite")
    ncc_mse_guard = getattr(args, "ncc_mse_guard", False)
    ncc_cov_eps = getattr(args, "ncc_cov_eps", 1e-6)
    sigma_ii = mu_var if (ncc_score == "cov" or ncc_mse_guard) else None

    for sweep_idx in range(args.ncc_sweeps):
        W_corr, stats = apply_ncc(
            W_fp=W_fp,
            qres=current,
            mu=mu,
            budget_p=args.ncc_budget_p,
            use_james_stein=args.ncc_use_james_stein,
            mu_var=mu_var,
            row_chunk=args.row_chunk,
            score=ncc_score,
            sigma_ii=sigma_ii,
            cov_eps=ncc_cov_eps,
            mse_guard=ncc_mse_guard,
        )
        bias_before = float(stats.bias_before)
        bias_after = float(stats.bias_after)
        improvement = bias_before - bias_after
        row = {
            "sweep": sweep_idx + 1,
            "flips": int(stats.flips),
            "bias_before": bias_before,
            "bias_after": bias_after,
            "improvement": improvement,
        }
        history.append(row)
        total_flips += int(stats.flips)
        first_bias_before = bias_before if first_bias_before is None else first_bias_before
        final_bias_after = bias_after
        if stats.flips == 0 or improvement <= args.ncc_stop_eps:
            break
        if sweep_idx + 1 < args.ncc_sweeps:
            current = _refresh_quant_indices_from_dequant(current, W_corr)

    return W_corr, {
        "flips": total_flips,
        "bias_before": float(first_bias_before if first_bias_before is not None else 0.0),
        "bias_after": float(final_bias_after if final_bias_after is not None else 0.0),
        "objective_before": float(first_bias_before if first_bias_before is not None else 0.0),
        "objective_after": float(final_bias_after if final_bias_after is not None else 0.0),
        "sweep_history": history,
    }


@torch.no_grad()
def _gptvq_fasterquant_ncc_post_block(gptq: GPTQ, args, mu: torch.Tensor) -> dict:
    if args.wbits not in (3, 4):
        raise ValueError("Post-block NCC path expects 3-bit or 4-bit GPTVQ.")
    if args.groupsize != args.gptq_blocksize:
        raise ValueError("--ncc-placement post_block currently requires --groupsize == --gptq-blocksize.")
    if args.include_m_step:
        raise ValueError("--ncc-placement post_block must be used with --no-include-m-step.")

    layer = gptq.layer
    W = layer.weight.data.clone()
    if isinstance(layer, nn.Conv2d):
        W = W.flatten(1)
    if isinstance(layer, transformers.Conv1D):
        W = W.t()
    W = W.float()
    W_ref = W.clone()

    gptq.tick = time.time()
    H = gptq.H
    gptq.G = gptq.H.clone()
    del gptq.H

    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    W_ref[:, dead] = 0

    quantizer = gptq.quantizer
    vq_dim = quantizer.vq_dim
    if vq_dim != 1:
        raise ValueError("--ncc-placement post_block currently supports GPTVQ vq_dim=1 only.")
    groupsize = quantizer.get_groupsize(W, args.groupsize)
    gptq.assignments = []

    vq_scaling_blocksize = quantizer.vq_scaling_blocksize
    vq_scaling_n_bits = quantizer.vq_scaling_n_bits
    if vq_scaling_blocksize > 0:
        raise ValueError("--ncc-placement post_block currently expects vq_scaling_blocksize <= 0.")

    print(W.shape)
    print(
        f"VQ scaling BS {vq_scaling_blocksize} @ {vq_scaling_n_bits}b "
        f"({quantizer.vq_scaling_domain} domain)"
    )
    print(f"Using Hessian-aware K-means {args.hessian_weighted_lookups}")
    print("NCC placement: post_block")

    Losses = torch.zeros_like(W)
    Q = torch.zeros_like(W)

    damp = args.percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(gptq.columns, device=gptq.dev)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    totals = {
        "flips": 0,
        "bias_before": 0.0,
        "bias_after": 0.0,
        "objective_before": 0.0,
        "objective_after": 0.0,
        "sweep_history": [],
    }
    mu = mu.to(W.device).float()

    for i1 in range(0, gptq.columns, args.gptq_blocksize):
        i2 = min(i1 + args.gptq_blocksize, gptq.columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        W1_start = W1.clone()
        W1_scaled = W1
        S1 = torch.ones_like(W1)
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        for i in range(count):
            if (i1 + i) % groupsize == 0:
                extra_args = {}
                W_group = W[:, (i1 + i) : (i1 + i + groupsize)]
                gptq.assignments.append([])
                quantizer.find_params(W_group, weight=True, **extra_args)

            w = W1[:, i : i + vq_dim]
            d = torch.diag(Hinv1)[i : i + vq_dim].unsqueeze(0)
            w_scaled = W1_scaled[:, i : i + vq_dim]
            s = S1[:, i : i + vq_dim]

            q, assmt = vq_quantize(w_scaled, quantizer, H_inv_diag=None)
            q = torch.mul(q, s)
            gptq.assignments[-1].append(assmt)

            Q1[:, i : i + vq_dim] = q
            Losses1[:, i : i + vq_dim] = (w - q) ** 2 / d**2
            err1 = (w - q) / d
            if i + vq_dim < count:
                update = torch.bmm(
                    err1.transpose(0, 1).unsqueeze(-1),
                    Hinv1[i : i + vq_dim, i + vq_dim :].unsqueeze(1),
                ).sum(0)
                W1[:, i + vq_dim :] -= update
                Err1[:, i : i + vq_dim] = err1

        qres = _gptvq_quant_result(
            W_dequant=Q1,
            assignments=[gptq.assignments[-1]],
            centroids=[quantizer.all_centroids[-1]],
            bits=args.wbits,
            block_size=count,
        )
        W_corr, ncc_stats = _apply_ncc_sweeps(
            W_fp=W_ref[:, i1:i2].to(W.device),
            qres=qres,
            mu=mu[i1:i2],
            mu_var=None,
            args=args,
        )
        Q1 = W_corr.to(Q1.dtype)
        d_all = torch.diag(Hinv1).unsqueeze(0).clamp(min=1e-12)
        Err1 = (W1_start - Q1) / d_all
        Losses1 = (W1_start - Q1) ** 2 / (d_all**2)

        for row in ncc_stats["sweep_history"]:
            totals["sweep_history"].append({"block_start": i1, "block_end": i2, **row})
        totals["flips"] += ncc_stats["flips"]
        totals["bias_before"] += ncc_stats["bias_before"]
        totals["bias_after"] += ncc_stats["bias_after"]
        totals["objective_before"] += ncc_stats["objective_before"]
        totals["objective_after"] += ncc_stats["objective_after"]

        Q[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    torch.cuda.synchronize() if W.device.type == "cuda" else None
    print("time %.2f" % (time.time() - gptq.tick))
    print("error", torch.sum(Losses).item())

    if isinstance(layer, transformers.Conv1D):
        Q = Q.t()
    layer.weight.data = Q.reshape(layer.weight.shape).to(layer.weight.data.dtype)
    return totals


@torch.no_grad()
def quantize_model_gptvq_1d(
    model,
    tokenizer,
    calib_texts: list[str],
    args,
    correction: str | None,
    gptvq_state: dict[str, torch.Tensor | str] | None = None,
    gptvq_snapshot_dir: Path | None = None,
) -> dict:
    device = _hf_device(args.device)
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise RuntimeError("GPTVQ runner expects a Llama-like model with model.layers")

    model.seqlen = args.max_length
    batches = _make_calibration_batches(tokenizer, calib_texts, args.max_length)
    actual_n_calib = min(args.n_calib, len(batches))
    if actual_n_calib <= 0:
        raise RuntimeError("No calibration batches were produced.")
    if actual_n_calib != args.n_calib:
        print(f"Using {actual_n_calib} calibration samples; requested {args.n_calib}.")

    inps, outs, cache = _capture_first_layer_inputs(
        model=model,
        batches=batches,
        device=device,
        nsamples=actual_n_calib,
        seqlen=args.max_length,
    )

    layers = model.model.layers
    use_cache = model.config.use_cache
    model.config.use_cache = False

    totals = {
        "flips": 0,
        "candidates": 0,
        "boundary_kept": 0,
        "bias_before": 0.0,
        "bias_after": 0.0,
        "objective_before": 0.0,
        "objective_after": 0.0,
        "variance_increase": 0.0,
    }
    ncc_sweep_history: list[dict] = []
    diagnostics: list[dict] = []
    diagnostic_inputs: dict[str, list[torch.Tensor]] = {}
    diagnostic_order: list[str] = []
    quantized_layers = 0
    tick = time.time()

    for layer_idx in range(len(layers)):
        print(f"\n=== GPTVQ layer {layer_idx + 1}/{len(layers)} ===")
        layer = layers[layer_idx].to(device)
        full = find_layers(layer)

        for names in _sequential_groups(full, args.true_sequential):
            subset = {name: full[name] for name in names}
            gptq = {}
            stat_sum: Dict[str, torch.Tensor] = {}
            stat_sumsq: Dict[str, torch.Tensor] = {}
            stat_count: Dict[str, int] = {}

            for name, module in subset.items():
                gptq[name] = GPTQ(module)
                gptq[name].quantizer = _make_vq_quantizer(args)

            def add_batch(name):
                key = _linear_key(layer_idx, name)

                def hook(_module, inp, out):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    gptq[name].add_batch(x.data, out.data)
                    if correction is not None:
                        _append_diagnostic_inputs(
                            key=key,
                            x=x,
                            diagnostic_inputs=diagnostic_inputs,
                            diagnostic_order=diagnostic_order,
                            args=args,
                        )
                        x_float = x.reshape(-1, x.shape[-1]).detach().float()
                        stat_sum[key] = stat_sum.get(key, torch.zeros(x_float.shape[-1])).to(x_float.device)
                        stat_sum[key] += x_float.sum(dim=0)
                        stat_sumsq[key] = stat_sumsq.get(key, torch.zeros(x_float.shape[-1])).to(x_float.device)
                        stat_sumsq[key] += (x_float * x_float).sum(dim=0)
                        stat_count[key] = stat_count.get(key, 0) + x_float.shape[0]

                return hook

            handles = [module.register_forward_hook(add_batch(name)) for name, module in subset.items()]
            try:
                for sample_idx in range(actual_n_calib):
                    outs[sample_idx] = _layer_call(layer, inps[sample_idx].unsqueeze(0), cache)
            finally:
                for handle in handles:
                    handle.remove()

            for name, module in subset.items():
                key = _linear_key(layer_idx, name)
                W_fp = module.weight.data.detach().clone().float()
                print(f"Quantizing {key} with upstream GPTVQ-1D ...")
                post_block_ncc = correction == "ncc" and args.ncc_placement == "post_block"
                if post_block_ncc:
                    if key not in stat_sum:
                        raise RuntimeError(f"Missing activation stats for NCC layer {key}")
                    count = max(1, stat_count[key])
                    mu = stat_sum[key].to(device) / count
                    ncc_stats = _gptvq_fasterquant_ncc_post_block(gptq[name], args=args, mu=mu)
                    totals["flips"] += ncc_stats["flips"]
                    totals["bias_before"] += ncc_stats["bias_before"]
                    totals["bias_after"] += ncc_stats["bias_after"]
                    totals["objective_before"] += ncc_stats["objective_before"]
                    totals["objective_after"] += ncc_stats["objective_after"]
                    for row in ncc_stats["sweep_history"]:
                        ncc_sweep_history.append({"layer": key, **row})
                    del mu
                else:
                    gptq[name].fasterquant(
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
                quantized_layers += 1
                if gptvq_state is not None:
                    snapshot = module.weight.data.detach().cpu().clone()
                    if gptvq_snapshot_dir is not None:
                        gptvq_snapshot_dir.mkdir(parents=True, exist_ok=True)
                        filename = key.replace("/", "__").replace(".", "_") + ".pt"
                        path = gptvq_snapshot_dir / filename
                        torch.save(snapshot, path)
                        gptvq_state[key] = str(path)
                        del snapshot
                    else:
                        gptvq_state[key] = snapshot
                if key in diagnostic_inputs:
                    X_diag = torch.cat(diagnostic_inputs[key], dim=0)
                    diagnostics.append(
                        _activation_error_metrics(
                            key=key,
                            X_cpu=X_diag,
                            W_fp=W_fp,
                            W_quant=module.weight.data.detach(),
                            variant="gptvq_ncc" if post_block_ncc else "gptvq",
                        )
                    )

                if correction is not None and not post_block_ncc:
                    if key not in stat_sum:
                        raise RuntimeError(f"Missing activation stats for {correction.upper()} layer {key}")
                    W_gptvq = module.weight.data.detach().float()
                    qres = _gptvq_quant_result(
                        W_dequant=W_gptvq,
                        assignments=gptq[name].assignments,
                        centroids=gptq[name].quantizer.all_centroids,
                        bits=args.wbits,
                        block_size=args.groupsize,
                    )
                    count = max(1, stat_count[key])
                    mu = stat_sum[key].to(device) / count
                    ex2 = stat_sumsq[key].to(device) / count
                    sigma = (ex2 - mu * mu).clamp(min=0.0)
                    if correction == "rbvt":
                        W_corr, stats = apply_rbvt(
                            W_fp=W_fp.to(device),
                            qres=qres,
                            mu=mu,
                            sigma_ii=sigma if args.rbvt_lambda > 0.0 else None,
                            rbvt_lambda=args.rbvt_lambda,
                            rbvt_topk=args.rbvt_topk if args.rbvt_topk > 0 else None,
                            row_chunk=args.row_chunk,
                            gap_floor=args.gap_floor,
                            strict_descent=args.strict_descent,
                        )
                    elif correction == "ncc":
                        W_corr, ncc_stats = _apply_ncc_sweeps(
                            W_fp=W_fp.to(device),
                            qres=qres,
                            mu=mu,
                            mu_var=sigma,
                            args=args,
                        )
                    else:
                        raise ValueError(f"Unknown correction: {correction}")
                    if key in diagnostic_inputs:
                        X_diag = torch.cat(diagnostic_inputs[key], dim=0)
                        if correction == "ncc":
                            diagnostics.append(
                                _activation_error_metrics(
                                    key=key,
                                    X_cpu=X_diag,
                                    W_fp=W_fp,
                                    W_quant=W_corr,
                                    variant=f"gptvq_ncc_sweep{len(ncc_stats['sweep_history'])}",
                                )
                            )
                        diagnostics.append(
                            _activation_error_metrics(
                                key=key,
                                X_cpu=X_diag,
                                W_fp=W_fp,
                                W_quant=W_corr,
                                variant=f"gptvq_{correction}",
                            )
                        )
                    module.weight.data = W_corr.to(module.weight.data.dtype)
                    if correction == "rbvt":
                        for total_key in totals:
                            totals[total_key] += getattr(stats, total_key)
                    else:
                        totals["flips"] += ncc_stats["flips"]
                        totals["bias_before"] += ncc_stats["bias_before"]
                        totals["bias_after"] += ncc_stats["bias_after"]
                        totals["objective_before"] += ncc_stats["objective_before"]
                        totals["objective_after"] += ncc_stats["objective_after"]
                        for row in ncc_stats["sweep_history"]:
                            ncc_sweep_history.append({"layer": key, **row})
                    del qres, W_corr, sigma, mu

                gptq[name].free()
                del W_fp
                torch.cuda.empty_cache()

        for sample_idx in range(actual_n_calib):
            outs[sample_idx] = _layer_call(layer, inps[sample_idx].unsqueeze(0), cache)

        if args.keep_model_on_device:
            layers[layer_idx] = layer
        else:
            layers[layer_idx] = layer.cpu()
        del layer, gptq
        torch.cuda.empty_cache()
        gc.collect()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    elapsed = time.time() - tick
    stats = {
        "method": f"gptvq_{correction}" if correction is not None else "gptvq",
        "bits": args.wbits,
        "vq_dim": 1,
        "num_linear_layers": quantized_layers,
        "groupsize": args.groupsize,
        "kmeans_iters": args.kmeans_iters,
        "kmeans_init_method": args.kmeans_init_method,
        "include_m_step": args.include_m_step,
        "hessian_weighted_lookups": args.hessian_weighted_lookups,
        "time_sec": elapsed,
    }
    if correction is not None:
        stats.update(totals)
        if correction == "rbvt":
            stats["rbvt_lambda"] = args.rbvt_lambda
            stats["rbvt_topk"] = args.rbvt_topk
        if correction == "ncc":
            stats["ncc_budget_p"] = args.ncc_budget_p
            stats["ncc_placement"] = args.ncc_placement
            stats["ncc_sweeps"] = args.ncc_sweeps
            stats["ncc_stop_eps"] = args.ncc_stop_eps
            stats["ncc_use_james_stein"] = args.ncc_use_james_stein
            stats["ncc_sweep_history"] = ncc_sweep_history
        stats["activation_error_diagnostics"] = diagnostics
        print(
            f"{correction.upper()} summary | "
            f"flips={totals['flips']} candidates={totals['candidates']} "
            f"bias={totals['bias_before']:.6e}->{totals['bias_after']:.6e}"
        )
        if diagnostics:
            print("Activation output error diagnostics | X @ (W_fp16 - W_quant).T")
            for row in diagnostics:
                print(
                    f"  {row['layer']} {row['variant']} tokens={row['tokens']} "
                    f"mae={row['mae']:.6e} mse={row['mse']:.6e} max_abs={row['max_abs']:.6e}"
                )
    print(f"GPTVQ variant done in {elapsed:.2f}s")
    return stats


@torch.no_grad()
def _restore_linear_weights(model, state: dict[str, torch.Tensor | str]):
    modules = dict(model.named_modules())
    missing = []
    for name, weight_ref in state.items():
        module = modules.get(name)
        if module is None or not hasattr(module, "weight"):
            missing.append(name)
            continue
        if isinstance(weight_ref, str):
            weight = torch.load(weight_ref, map_location="cpu")
        else:
            weight = weight_ref
        module.weight.data.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
    if missing:
        raise RuntimeError(f"Missing modules while restoring GPTVQ baseline: {missing[:5]}")


def evaluate_model(model_path: str, label: str, args, hf_token: str | None) -> tuple[dict, dict]:
    evaluator = RBVTSlidingWindowEvaluator(
        device=args.device,
        seed=args.seed,
        stride=args.eval_stride,
        max_length=args.eval_max_length,
        cache_dir=args.eval_cache_dir,
        hf_token=hf_token,
    )
    perplexity = {}
    for dataset_name, texts in {
        "WikiText-2": evaluator.load_wikitext2_test(args.eval_samples),
        "C4": evaluator.load_c4_validation(args.eval_samples),
    }.items():
        result = evaluator.evaluate_model_on_dataset(
            model_path=model_path,
            model_name=label,
            texts=texts,
            dataset_name=dataset_name,
        )
        if result is not None:
            perplexity[dataset_name] = result

    lm_eval = {}
    if args.include_lm_eval:
        runner = LMEvalHarnessRunner(
            tasks=args.lm_eval_tasks,
            device=args.device,
            batch_size=args.lm_eval_batch_size,
            num_fewshot=args.lm_eval_num_fewshot,
            limit=args.lm_eval_limit,
            output_dir=args.lm_eval_output_dir,
            run_name=label.lower(),
            hf_token=hf_token,
        )
        lm_eval = runner.run({label: model_path})
    return perplexity, lm_eval


def _cleanup_model_artifacts(output_dir: Path):
    keep = {"run_summary.json"}
    for child in output_dir.iterdir():
        if child.name in keep:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _write_summary(output_dir: Path, summary: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_variant(variant: str, args, hf_token: str | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    correction = None
    if variant == "gptvq_rbvt":
        correction = "rbvt"
    elif variant == "gptvq_ncc":
        correction = "ncc"
    label = variant.upper() if correction is not None else "GPTVQ"
    output_dir = Path(args.output_root) / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}\nRunning {label}\n{'=' * 80}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": torch.float16 if args.device.startswith("cuda") else torch.float32,
        "trust_remote_code": True,
        "token": hf_token,
        "low_cpu_mem_usage": True,
    }
    if args.keep_model_on_device:
        load_kwargs["device_map"] = {"": args.device}
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    model.eval()

    calib_texts = load_calibration_data(
        dataset_name=args.calib_dataset,
        tokenizer=tokenizer,
        n_samples=args.n_calib,
        seqlen=args.max_length,
        seed=args.seed,
        cache_dir=args.calibration_cache_dir,
    )
    quant_stats = quantize_model_gptvq_1d(
        model=model,
        tokenizer=tokenizer,
        calib_texts=calib_texts,
        args=args,
        correction=correction,
    )

    print(f"Saving {label} model to {output_dir} ...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    del model
    torch.cuda.empty_cache()
    gc.collect()

    perplexity, lm_eval = evaluate_model(str(output_dir), label, args, hf_token=hf_token)
    summary = {
        "model_path": args.model_path,
        "variant": variant,
        "output_dir": str(output_dir),
        "quantization": quant_stats,
        "calibration": {
            "dataset": args.calib_dataset,
            "n_calib": args.n_calib,
            "max_length": args.max_length,
            "seed": args.seed,
        },
        "evaluation": {
            "perplexity": perplexity,
            "lm_eval": lm_eval,
            "lm_eval_tasks": args.lm_eval_tasks,
        },
        "args": vars(args),
    }
    _write_summary(output_dir, summary)
    if args.cleanup_model_artifacts:
        _cleanup_model_artifacts(output_dir)
        print(f"Cleaned model artifacts under {output_dir}; kept run_summary.json")
    return summary


def _make_summary(args, variant: str, output_dir: Path, quant_stats: dict, perplexity: dict, lm_eval: dict) -> dict:
    return {
        "model_path": args.model_path,
        "variant": variant,
        "output_dir": str(output_dir),
        "quantization": quant_stats,
        "calibration": {
            "dataset": args.calib_dataset,
            "n_calib": args.n_calib,
            "max_length": args.max_length,
            "seed": args.seed,
        },
        "evaluation": {
            "perplexity": perplexity,
            "lm_eval": lm_eval,
            "lm_eval_tasks": args.lm_eval_tasks,
        },
        "args": vars(args),
    }


def run_single_pass_compare(args, hf_token: str | None) -> list[dict]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    corrected_variant = f"gptvq_{args.correction}"
    corrected_label = corrected_variant.upper()
    print(f"\n{'=' * 80}\nRunning GPTVQ and {corrected_label} in one GPTVQ pass\n{'=' * 80}")
    output_root = Path(args.output_root)
    gptvq_dir = output_root / "gptvq"
    corrected_dir = output_root / corrected_variant
    gptvq_dir.mkdir(parents=True, exist_ok=True)
    corrected_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "torch_dtype": torch.float16 if args.device.startswith("cuda") else torch.float32,
        "trust_remote_code": True,
        "token": hf_token,
        "low_cpu_mem_usage": True,
    }
    if args.keep_model_on_device:
        load_kwargs["device_map"] = {"": args.device}
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    model.eval()

    calib_texts = load_calibration_data(
        dataset_name=args.calib_dataset,
        tokenizer=tokenizer,
        n_samples=args.n_calib,
        seqlen=args.max_length,
        seed=args.seed,
        cache_dir=args.calibration_cache_dir,
    )

    gptvq_state: dict[str, torch.Tensor | str] = {}
    snapshot_dir = output_root / "_gptvq_weight_snapshots"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    corrected_stats = quantize_model_gptvq_1d(
        model=model,
        tokenizer=tokenizer,
        calib_texts=calib_texts,
        args=args,
        correction=args.correction,
        gptvq_state=gptvq_state,
        gptvq_snapshot_dir=snapshot_dir,
    )
    gptvq_stats = {
        key: value
        for key, value in corrected_stats.items()
        if key
        not in {
            "flips",
            "candidates",
            "boundary_kept",
            "bias_before",
            "bias_after",
            "objective_before",
            "objective_after",
            "variance_increase",
            "rbvt_lambda",
            "rbvt_topk",
        }
    }
    gptvq_stats["method"] = "gptvq"
    gptvq_stats["shared_gptvq_pass"] = True
    corrected_stats["shared_gptvq_pass"] = True
    all_diagnostics = corrected_stats.get("activation_error_diagnostics", [])
    if isinstance(all_diagnostics, list):
        gptvq_stats["activation_error_diagnostics"] = [
            row for row in all_diagnostics if row.get("variant") == "gptvq"
        ]
        corrected_stats["activation_error_diagnostics"] = [
            row for row in all_diagnostics if row.get("variant") == corrected_variant
        ]

    print(f"Saving {corrected_label} model to {corrected_dir} ...")
    model.save_pretrained(corrected_dir)
    tokenizer.save_pretrained(corrected_dir)

    print(f"Restoring GPTVQ baseline weights from the shared pass and saving to {gptvq_dir} ...")
    _restore_linear_weights(model, gptvq_state)
    model.save_pretrained(gptvq_dir)
    tokenizer.save_pretrained(gptvq_dir)
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    del model, gptvq_state
    torch.cuda.empty_cache()
    gc.collect()

    summaries = []
    for variant, label, output_dir, quant_stats in (
        ("gptvq", "GPTVQ", gptvq_dir, gptvq_stats),
        (corrected_variant, corrected_label, corrected_dir, corrected_stats),
    ):
        perplexity, lm_eval = evaluate_model(str(output_dir), label, args, hf_token=hf_token)
        summary = _make_summary(
            args=args,
            variant=variant,
            output_dir=output_dir,
            quant_stats=quant_stats,
            perplexity=perplexity,
            lm_eval=lm_eval,
        )
        _write_summary(output_dir, summary)
        summaries.append(summary)
        if args.cleanup_model_artifacts:
            _cleanup_model_artifacts(output_dir)
            print(f"Cleaned model artifacts under {output_dir}; kept run_summary.json")

    return summaries


def print_comparison(summaries: list[dict]):
    print("\n" + "=" * 80)
    print("GPTVQ 1D COMPARISON")
    print("=" * 80)
    for summary in summaries:
        variant = summary["variant"]
        ppl = summary.get("evaluation", {}).get("perplexity", {})
        lm_eval = summary.get("evaluation", {}).get("lm_eval", {})
        print(f"\n[{variant}]")
        for dataset_name in ("WikiText-2", "C4"):
            value = ppl.get(dataset_name, {}).get("perplexity")
            print(f"  ppl/{dataset_name}: {value:.4f}" if isinstance(value, float) else f"  ppl/{dataset_name}: MISSING")
        payload = next(iter(lm_eval.values()), {}) if isinstance(lm_eval, dict) and lm_eval else {}
        task_summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        for task in summary.get("evaluation", {}).get("lm_eval_tasks", []):
            metrics = task_summary.get(task, {})
            metric_value = None
            metric_name = None
            if isinstance(metrics, dict):
                for candidate in ("acc,none", "acc_norm,none", "exact_match,none", "exact_match"):
                    if isinstance(metrics.get(candidate), (int, float)):
                        metric_name = candidate
                        metric_value = float(metrics[candidate])
                        break
            if metric_value is None:
                print(f"  lm_eval/{task}: MISSING")
            else:
                print(f"  lm_eval/{task}/{metric_name}: {metric_value:.4f}")


def build_parser():
    parser = argparse.ArgumentParser(description="Compare upstream GPTVQ-1D with post-GPTVQ scalar corrections")
    parser.add_argument("--model-path", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="./outputs/gptvq_1d_rbvt_colab")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["gptvq", "gptvq_rbvt"],
        choices=["gptvq", "gptvq_rbvt", "gptvq_ncc"],
    )
    parser.add_argument(
        "--single-pass-compare",
        action="store_true",
        help="Run GPTVQ once, snapshot GPTVQ weights, apply a correction, then save/evaluate both variants.",
    )
    parser.add_argument(
        "--correction",
        choices=["rbvt", "ncc"],
        default="rbvt",
        help="Post-GPTVQ assignment correction used by --single-pass-compare.",
    )
    parser.add_argument(
        "--keep-model-on-device",
        action="store_true",
        help="Load and keep the full model on --device. Useful on A100 to avoid CPU RAM OOM.",
    )
    parser.add_argument("--wbits", type=int, default=4, choices=[3, 4])
    parser.add_argument("--groupsize", type=int, default=128)
    parser.add_argument("--gptq-blocksize", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--kmeans-iters", type=int, default=20)
    parser.add_argument("--kmeans-init-method", choices=["cdf", "kpp", "mahalanobis"], default="mahalanobis")
    parser.add_argument("--assignment-chunk-size", type=int, default=4096)
    parser.add_argument("--kpp-n-subsample", type=int, default=10000)
    parser.add_argument("--include-m-step", action="store_true", default=True)
    parser.add_argument("--no-include-m-step", dest="include_m_step", action="store_false")
    parser.add_argument("--hessian-weighted-lookups", action="store_true", default=True)
    parser.add_argument("--no-hessian-weighted-lookups", dest="hessian_weighted_lookups", action="store_false")
    parser.add_argument("--true-sequential", action="store_true", default=True)
    parser.add_argument("--no-true-sequential", dest="true_sequential", action="store_false")
    parser.add_argument("--sym", action="store_true", default=False)
    parser.add_argument("--n-calib", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--calib-dataset", choices=["c4", "wikitext2"], default="wikitext2")
    parser.add_argument("--calibration-cache-dir", default="./calibration_cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--row-chunk", type=int, default=1024)
    parser.add_argument("--rbvt-lambda", type=float, default=1.0)
    parser.add_argument("--rbvt-topk", type=int, default=0)
    parser.add_argument("--ncc-budget-p", type=float, default=0.02)
    parser.add_argument(
        "--ncc-placement",
        choices=["post_module", "post_block"],
        default="post_module",
        help="Run NCC after each full Linear module or inside GPTVQ after each GPTQ block.",
    )
    parser.add_argument(
        "--ncc-sweeps",
        type=int,
        default=1,
        help="Number of iterative NCC correction sweeps after each GPTVQ-1D layer/module quantization.",
    )
    parser.add_argument(
        "--ncc-stop-eps",
        type=float,
        default=0.0,
        help="Stop NCC sweeps when first-moment bias improvement is at or below this value.",
    )
    parser.add_argument(
        "--ncc-use-james-stein",
        action="store_true",
        default=False,
        help="Use NCCQuant's James-Stein activation-mean shrinkage option.",
    )
    parser.add_argument("--gap-floor", type=float, default=1e-8)
    parser.add_argument("--strict-descent", action="store_true", default=True)
    parser.add_argument("--allow-overshoot", dest="strict_descent", action="store_false")
    parser.add_argument(
        "--diagnostic-layer-limit",
        type=int,
        default=0,
        help="Number of early Linear layers for X @ (W_fp16 - W_quant).T MAE/MSE diagnostics.",
    )
    parser.add_argument(
        "--diagnostic-max-tokens",
        type=int,
        default=4096,
        help="Maximum calibration tokens retained per diagnostic layer.",
    )
    parser.add_argument("--eval-stride", type=int, default=512)
    parser.add_argument("--eval-max-length", type=int, default=1024)
    parser.add_argument("--eval-samples", type=int, default=64)
    parser.add_argument("--eval-cache-dir", default="./dataset_cache")
    parser.add_argument("--include-lm-eval", action="store_true", default=True)
    parser.add_argument("--no-lm-eval", dest="include_lm_eval", action="store_false")
    parser.add_argument("--lm-eval-tasks", nargs="+", default=["arc_easy", "arc_challenge"])
    parser.add_argument("--lm-eval-num-fewshot", type=int, default=None)
    parser.add_argument("--lm-eval-batch-size", default="auto")
    parser.add_argument("--lm-eval-limit", type=float, default=100)
    parser.add_argument("--lm-eval-output-dir", default="./outputs/gptvq_1d_rbvt_colab/lm_eval")
    parser.add_argument("--cleanup-model-artifacts", action="store_true", default=True)
    parser.add_argument("--keep-model-artifacts", dest="cleanup_model_artifacts", action="store_false")
    return parser


def main():
    load_runtime_env()
    args = build_parser().parse_args()
    if args.groupsize <= 0:
        raise ValueError("--groupsize must be positive for GPTVQ-1D/RBVT index conversion.")
    if args.rbvt_lambda < 0:
        raise ValueError("--rbvt-lambda must be non-negative.")
    if not 0.0 < args.ncc_budget_p <= 1.0:
        raise ValueError("--ncc-budget-p must be in (0, 1].")
    if args.ncc_sweeps <= 0:
        raise ValueError("--ncc-sweeps must be positive.")
    if args.ncc_stop_eps < 0:
        raise ValueError("--ncc-stop-eps must be non-negative.")
    if args.single_pass_compare and args.ncc_placement == "post_block":
        raise ValueError("--ncc-placement post_block runs one corrected variant and cannot use --single-pass-compare.")
    if args.ncc_placement == "post_block" and args.groupsize != args.gptq_blocksize:
        raise ValueError("--ncc-placement post_block currently requires --groupsize == --gptq-blocksize.")
    if args.ncc_placement == "post_block" and args.include_m_step:
        raise ValueError("--ncc-placement post_block requires --no-include-m-step so final M-step does not overwrite NCC.")
    _set_seed(args.seed)
    hf_token = resolve_hf_token()
    print(
        f"Model={args.model_path} | device={args.device} | bits={args.wbits} | "
        f"variants={args.variants} | output={args.output_root}"
    )
    print(f"Model slug: {build_model_slug(args.model_path)}")

    if args.single_pass_compare:
        summaries = run_single_pass_compare(args, hf_token=hf_token)
    else:
        summaries = [run_variant(variant, args, hf_token=hf_token) for variant in args.variants]
    print_comparison(summaries)


if __name__ == "__main__":
    main()