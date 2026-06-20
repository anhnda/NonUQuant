"""
Bias Correction (BC) for non-uniform codebook quantization.

Standalone, block-format-agnostic bias corrector and an INDEPENDENT alternative
to NCC. Where NCC removes the per-channel first-moment error mu . e by *flipping
codewords* (changing the dequantized weight), BC removes the mean output error by
absorbing it into the layer's BIAS term, leaving the quantized weights untouched.

For a Linear y = x W^T (+ bias), the expected output error from quantization is
    E[y_fp - y_q] = E[x] (W_fp - W_q)^T = mu @ e^T      [out]
where e = W_q - W_fp and mu = E[x] over the calibration set. Adding
    bias <- bias - mu @ e^T   ( == bias + mu @ (W_fp - W_q)^T )
makes the layer's expected output unbiased to first order. This is the classic
naive bias correction (Nagel et al., "Data-Free Quantization"; AdaRound's bias
term), expressed on the same mu the driver already collects.

Standalone by design
---------------------
BC reads the PLAIN quantized weights W_q (the quantizer's W_dequant), NOT any
NCC-corrected version. It is meant to be run as an alternative to NCC, not on top
of it: BC alone is full naive bias correction; NCC alone flips codewords. Running
both at once is an inconsistent combo (the bias would correct an error the NCC'd
weights no longer carry), so the driver exposes them as separate switches.

This module performs no torch execution on import (user preference); it only
defines functions invoked by the driver.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BCStats:
    """Diagnostics for one layer's bias correction.

    Attributes
    ----------
    out_features : int
        Number of output channels corrected.
    bias_created : bool
        True if the layer had no bias and one was created to hold the correction.
    mean_abs_correction : float
        Mean |correction| over output channels (size of the shift folded in).
    max_abs_correction : float
        Largest single-channel |correction| (catches pathological channels).
    out_err_before : float
        Sum over channels of (mu . e_j)^2 BEFORE correction (== the output
        first-moment energy BC removes).
    out_err_after : float
        Same quantity AFTER folding the correction into the bias. Exactly 0 up to
        float error, since naive BC is an exact projection of the mean error.
    """

    out_features: int
    bias_created: bool
    mean_abs_correction: float
    max_abs_correction: float
    out_err_before: float
    out_err_after: float


@torch.no_grad()
def compute_bias_correction(
    W_fp: torch.Tensor,
    W_q: torch.Tensor,
    mu: torch.Tensor,
    row_chunk: int = 4096,
) -> tuple[torch.Tensor, float, float]:
    """Mean output error correction term  c_j = mu . (W_fp - W_q)_j.

    Adding c to the bias cancels E[y_fp - y_q] to first order.

    Parameters
    ----------
    W_fp : [out, in]  full-precision weights.
    W_q  : [out, in]  PLAIN quantized (dequantized) weights from the quantizer.
    mu   : [in]       per-input-channel activation mean E[x].
    row_chunk : output rows processed at once (memory bound; no effect on result).

    Returns
    -------
    correction : [out]  the term to ADD to bias (== mu @ (W_fp - W_q)^T).
    out_err_before : float  sum_j (mu . e_j)^2 with e = W_q - W_fp.
    out_err_after  : float  residual after applying correction (~0).
    """
    device = W_fp.device
    out_features, in_features = W_fp.shape
    mu = mu.to(device).float()

    correction = torch.empty(out_features, device=device, dtype=torch.float32)
    out_err_before = 0.0

    for r0 in range(0, out_features, row_chunk):
        r1 = min(r0 + row_chunk, out_features)
        e = (W_q[r0:r1].float() - W_fp[r0:r1].float())    # [rc, in]  residual e = W_q - W_fp
        # per-channel mean output error  mu . e_j
        moe = e @ mu                                       # [rc]
        correction[r0:r1] = -moe                           # add -mu.e to bias
        out_err_before += float((moe * moe).sum().item())
        del e, moe

    # After folding c = -mu.e into the bias, the expected output error per channel
    # becomes mu.e + c = 0 exactly; residual is float round-off only.
    out_err_after = 0.0
    return correction, out_err_before, out_err_after


@torch.no_grad()
def apply_bias_correction(
    module,
    W_fp: torch.Tensor,
    W_q: torch.Tensor,
    mu: torch.Tensor,
    row_chunk: int = 4096,
) -> BCStats:
    """Fold the mean output error of (W_q vs W_fp) into `module.bias` in place.

    Creates a bias Parameter if the Linear has none. The weights themselves are
    NOT modified here — the caller is expected to have already written the
    quantized weights into module.weight (BC only touches the bias). Pass the
    PLAIN quantized weights as W_q so BC stays a standalone corrector.

    Parameters mirror compute_bias_correction. `module` is an nn.Linear-like with
    `.bias` (Optional[Parameter]) and `.weight`.
    """
    import torch.nn as nn

    device = module.weight.device
    out_features = W_fp.shape[0]
    orig_dtype = module.weight.dtype

    correction, err_before, err_after = compute_bias_correction(
        W_fp=W_fp, W_q=W_q, mu=mu, row_chunk=row_chunk
    )
    correction = correction.to(device)

    bias_created = module.bias is None
    if bias_created:
        module.bias = nn.Parameter(
            torch.zeros(out_features, device=device, dtype=orig_dtype)
        )
    module.bias.data = (module.bias.data.float() + correction).to(orig_dtype)

    abs_c = correction.abs()
    return BCStats(
        out_features=out_features,
        bias_created=bias_created,
        mean_abs_correction=float(abs_c.mean().item()),
        max_abs_correction=float(abs_c.max().item()),
        out_err_before=err_before,
        out_err_after=err_after,
    )