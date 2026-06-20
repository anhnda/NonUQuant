"""
Non-uniform codebook quantizers (block-wise standard) + NCC / BC correction.

Public API
----------
    get_quantizer(name, **kwargs) -> BaseQuantizer
    apply_ncc(W_fp, qres, mu, ...) -> (W_corrected, NCCStats)
    apply_bias_correction(module, W_fp, W_q, mu, ...) -> BCStats

Names: nf3, nf4, nvfp4, codebook3, codebook4.

Standard granularity (block = contiguous run along input sharing one scale):
    NF3/NF4   block_size 64   (bitsandbytes default)
    NVFP4     block_size 16   (NVIDIA), FP8 E4M3 block scale
    codebook  block_size 64   (per-block learned levels)

Contract:
    res = quantizer.quantize(W, row_chunk=1024)   # block-wise, OOM-safe
    W_corr, stats = apply_ncc(W_fp, res, mu, budget_p=..., row_chunk=1024)
    bc_stats = apply_bias_correction(module, W_fp, res.W_dequant, mu, ...)
"""

from __future__ import annotations

from .base_quantizer import BaseQuantizer, QuantResult
from .normalfloat import NormalFloatQuantizer
from .nvfp4 import NVFP4Quantizer
from .learned_codebook import LearnedCodebookQuantizer
from .ncc import apply_ncc, NCCStats, james_stein_mean
from .bc import apply_bias_correction, compute_bias_correction, BCStats


def _nf3(**kw):
    return NormalFloatQuantizer(bits=3, block_size=kw.get("nf_block_size", 64))


def _nf4(**kw):
    return NormalFloatQuantizer(bits=4, block_size=kw.get("nf_block_size", 64))


def _nvfp4(**kw):
    return NVFP4Quantizer(bits=4, block_size=kw.get("nvfp4_block_size", 16),
                          fp8_scale=kw.get("fp8_scale", True))


def _codebook3(**kw):
    return LearnedCodebookQuantizer(bits=3, block_size=kw.get("cb_block_size", 64),
                                    n_iters=kw.get("n_iters", 20), seed=kw.get("seed", 0))


def _codebook4(**kw):
    return LearnedCodebookQuantizer(bits=4, block_size=kw.get("cb_block_size", 64),
                                    n_iters=kw.get("n_iters", 20), seed=kw.get("seed", 0))


QUANTIZER_REGISTRY = {
    "nf3": _nf3,
    "nf4": _nf4,
    "nvfp4": _nvfp4,
    "codebook3": _codebook3,
    "codebook4": _codebook4,
}


def get_quantizer(name: str, **kwargs) -> BaseQuantizer:
    key = name.lower()
    if key not in QUANTIZER_REGISTRY:
        raise ValueError(f"Unknown quantizer {name!r}. Available: {sorted(QUANTIZER_REGISTRY)}")
    return QUANTIZER_REGISTRY[key](**kwargs)


__all__ = [
    "BaseQuantizer", "QuantResult",
    "NormalFloatQuantizer", "NVFP4Quantizer", "LearnedCodebookQuantizer",
    "apply_ncc", "NCCStats", "james_stein_mean",
    "apply_bias_correction", "compute_bias_correction", "BCStats",
    "get_quantizer", "QUANTIZER_REGISTRY",
]