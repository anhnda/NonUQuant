"""
Non-uniform Codebook Correction (NCC) — block-aware, row-chunked.

Implements the paper's first-moment corrector (Sec. 3.2-3.5, Algorithm 1) over
the block-wise codebook standard. Each weight sits in a per-(row, block) codebook;
its complementary gap g_{i,j} is read from the neighbouring level *in its own
block's grid*. This is the realised-neighbour query the paper specifies for the
non-uniform / per-block case (the same mechanism NVFP4 needs).

For each output channel j (row): b_j = mu . e_j, e = Wq - W (mu over input dim).
Per-move first-moment change uses gap g read per block; selection is the
sign-aligned, gap-aware eta = |mu_i|/g ordering with greedy prefix under budget
B_j = ceil(p |A_j|). Empty selection is always feasible => Prop. 1 descent.

Degenerate-gap guard (Assumption 2, strict g>0). A complementary move whose gap
g_{i,j} is (near) zero corrects |v_{i,j}| = |mu_i| g ~ 0 first-moment error by the
method's own per-move equation, yet eta = |mu_i|/g -> infinity sends it to the top
of the ordering and lets it consume a budget slot for no progress. Assumption 2
already requires 0 < g_{i,j} <= g_max strictly; we make that numerically real with
an absolute floor `gap_floor`, so degenerate / coincident codewords (common in
learned tables and at NF level boundaries) are simply not feasible moves. This is a
feasibility cleanup, not a new ranking criterion: the eta ordering remains the sole
selection rule, so Prop. 1 (descent) and Thm A/B are untouched.

Memory: processed in row chunks; within a chunk we reconstruct realised levels
[rc, n_blocks, L] (or read learned block_codebooks) and the per-weight neighbour
gaps [rc, in]. No [out, in, L] tensor is ever held for the whole matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .base_quantizer import QuantResult


@dataclass
class NCCStats:
    flips: int
    bias_before: float
    bias_after: float
    channels: int
    # Diagnostics: total first-moment error actually removed by the selected
    # flips (sum |v| over chosen moves) and the smallest selected per-move
    # correction. If `min_selected_abs_v` is ~0, the budget was being spent on
    # zero-progress (degenerate-gap) flips — the bug the gap floor fixes.
    sum_selected_abs_v: float = 0.0
    min_selected_abs_v: float = float("inf")
    num_degenerate_dropped: int = 0


def james_stein_mean(x_bar: torch.Tensor, var: Optional[torch.Tensor] = None) -> torch.Tensor:
    """James-Stein shrinkage of the sample mean toward its global mean (Sec 3.5).

    Off by default in apply_ncc: with m >> d (calibration tokens >> input dim) the
    sample mean is not in the regime JS was built for, and the gap-aware ranking
    (not a threshold) does not need it. Kept for the ablation row only. NOTE: `var`
    must be the variance OF THE MEAN ESTIMATE (activation variance / m), not the raw
    activation variance, or the shrink factor is inflated by ~m and collapses mu.
    """
    mu0 = x_bar.mean()
    d = x_bar.numel()
    if d < 3:
        return x_bar.clone()
    diff = x_bar - mu0
    ss = (diff * diff).sum().clamp(min=1e-12)
    v = var.mean() if var is not None else torch.ones((), device=x_bar.device)
    shrink = (1.0 - (d - 2) * v / ss).clamp(min=0.0, max=1.0)
    return mu0 + shrink * diff


@torch.no_grad()
def _block_realised_levels(qres: QuantResult, r0: int, r1: int, device) -> torch.Tensor:
    """Realised level grid for rows [r0:r1], shape [rc, n_blocks, L].

    Factorised formats: block_scales[:, :, None] * q_levels[None, None, :].
    Learned formats: read block_codebooks directly.
    """
    if qres.block_codebooks is not None:
        return qres.block_codebooks[r0:r1].to(device).float()
    q = qres.q_levels.to(device).float()                       # [L]
    bscale = qres.block_scales[r0:r1].to(device).float()       # [rc, n_blocks]
    return bscale.unsqueeze(-1) * q.view(1, 1, -1)             # [rc, n_blocks, L]


@torch.no_grad()
def apply_ncc(
    W_fp: torch.Tensor,
    qres: QuantResult,
    mu: torch.Tensor,
    budget_p: float = 0.02,
    use_james_stein: bool = False,
    mu_var: Optional[torch.Tensor] = None,
    row_chunk: int = 1024,
    gap_floor: float = 1e-8,
    gap_floor_rel: float = 0.0,
    score: str = "lite",
    sigma_ii: Optional[torch.Tensor] = None,
    cov_eps: float = 1e-6,
) -> tuple[torch.Tensor, NCCStats]:
    """Run block-aware NCC; return corrected dequant weights + stats.

    Parameters
    ----------
    W_fp : [out, in] full-precision weights.
    qres : QuantResult from a block-wise quantizer (indices, block info).
    mu   : [in] per-input-channel activation mean.
    budget_p : budget fraction p in (0,1].
    row_chunk : rows processed at once (memory bound; no effect on result).
    gap_floor : absolute lower bound on a feasible complementary gap. Enforces the
        strict g>0 of Assumption 2 numerically and removes degenerate / coincident
        codewords (|v| ~ 0 yet eta -> inf). Keep this an *absolute* machine-scale
        guard; it is a feasibility condition, not a hyperparameter.
    gap_floor_rel : OPTIONAL relative floor as a fraction of the per-row median gap.
        Default 0 (disabled). WARNING: a meaningful relative floor would exclude
        genuine narrow *centre* cells — exactly where the certificate says
        corrections are cheapest and safest — so it contradicts the theory. Leave
        at 0 unless you specifically want that ablation.
    score : "lite" -> eta = |mu_i| / g_{i,j}  (covariance-free; NCC-Lite).
            "cov"  -> eta = |mu_i| / ((sigma_ii + cov_eps) * g_{i,j})  (NCC-Cov),
            the rule derived from the diagonal bias-variance surrogate
            (bias progress |mu_i| g over diagonal variance cost sigma_ii g^2).
    sigma_ii : [in] per-input-channel activation variance (diag of Sigma). Required
        for score="cov"; ignored for "lite".
    cov_eps : small constant added to sigma_ii to avoid division blow-up.
    """
    device = W_fp.device
    out_features, in_features = W_fp.shape
    bs = qres.block_size
    n_blocks = (in_features + bs - 1) // bs

    mu = mu.to(device).float()
    if use_james_stein:
        mu = james_stein_mean(mu, mu_var.to(device).float() if mu_var is not None else None)

    if score not in ("lite", "cov"):
        raise ValueError(f"score must be 'lite' or 'cov', got {score!r}")
    if score == "cov":
        if sigma_ii is None:
            raise ValueError("score='cov' requires sigma_ii (per-input-channel variance)")
        sigma_ii = sigma_ii.to(device).float()  # [in]

    Wq_full = qres.W_dequant.to(device).float()
    indices_full = qres.indices.to(device)

    Wq_corr = Wq_full.clone()
    total_flips = 0
    bias_before = 0.0
    bias_after = 0.0
    sum_selected_abs_v = 0.0
    min_selected_abs_v = float("inf")
    num_degenerate_dropped = 0

    # Column -> block id map (last block may be short).
    col_block = torch.arange(in_features, device=device) // bs   # [in]

    for r0 in range(0, out_features, row_chunk):
        r1 = min(r0 + row_chunk, out_features)
        rc = r1 - r0

        Wr = W_fp[r0:r1].float()                       # [rc, in]
        Wq = Wq_full[r0:r1]                             # [rc, in]
        idx = indices_full[r0:r1]                       # [rc, in] level index within block grid
        levels = _block_realised_levels(qres, r0, r1, device)   # [rc, n_blocks, L]
        L = levels.shape[-1]

        e = Wq - Wr                                     # residual
        e_sign = torch.sign(e)
        b = (e * mu.unsqueeze(0)).sum(dim=1)            # [rc]  channel first-moment

        # Gather each weight's block grid: levels_per_w [rc, in, L].
        # block id per column -> expand to [rc, in, L] index into n_blocks axis.
        blk = col_block.view(1, in_features, 1).expand(rc, in_features, L)
        levels_per_w = torch.gather(levels, 1, blk)     # [rc, in, L]

        # Current / left / right codeword values from the weight's own block grid.
        cur = torch.gather(levels_per_w, 2, idx.unsqueeze(-1)).squeeze(-1)        # [rc,in]
        left_idx = (idx - 1).clamp(min=0)
        right_idx = (idx + 1).clamp(max=L - 1)
        left = torch.gather(levels_per_w, 2, left_idx.unsqueeze(-1)).squeeze(-1)
        right = torch.gather(levels_per_w, 2, right_idx.unsqueeze(-1)).squeeze(-1)
        del levels_per_w, blk

        g_left = (cur - left).abs()
        g_right = (right - cur).abs()
        move_down = e_sign > 0                          # complementary = left neighbour
        gap = torch.where(move_down, g_left, g_right)
        target_idx = torch.where(move_down, left_idx, right_idx)
        target_val = torch.where(move_down, left, right)
        feasible = torch.where(move_down, idx > 0, idx < (L - 1))

        # Degenerate-gap guard (Assumption 2: strict g>0). Absolute floor, plus an
        # optional (default-off) relative floor for ablations only.
        floor = gap.new_full((), gap_floor)
        if gap_floor_rel > 0.0:
            # per-row median over feasible, positive gaps
            gpos = torch.where(feasible & (gap > gap_floor), gap, torch.full_like(gap, float("nan")))
            med = torch.nanmedian(gpos, dim=1, keepdim=True).values  # [rc,1]
            med = torch.nan_to_num(med, nan=gap_floor)
            rel = gap_floor_rel * med
            gap_ok = gap > torch.maximum(floor, rel)
        else:
            gap_ok = gap > floor

        # v = mu * sign(e) * g (first-moment corrected per move).
        v = mu.unsqueeze(0) * e_sign * gap              # [rc, in]
        # Sign filter (Eq.5): sign(mu) == sign(e*b).
        sign_ok = torch.sign(mu).unsqueeze(0) == torch.sign(e_sign * b.unsqueeze(1))

        # Track how many would-be candidates we drop purely for degenerate gap.
        would_admit = feasible & sign_ok
        num_degenerate_dropped += int((would_admit & ~gap_ok).sum().item())

        admissible = feasible & sign_ok & gap_ok
        # eta ordering. Floor the denominator at the *same* feasible floor so a
        # surviving candidate's eta cannot be inflated by a sub-floor gap.
        # lite: eta = |mu| / g ; cov: eta = |mu| / ((sigma_ii+eps) * g).
        if score == "cov":
            denom = (sigma_ii.unsqueeze(0) + cov_eps) * gap.clamp(min=gap_floor)
        else:
            denom = gap.clamp(min=gap_floor)
        eta = mu.abs().unsqueeze(0) / denom
        eta = torch.where(admissible, eta, torch.full_like(eta, -1.0))
        order = torch.argsort(eta, dim=1, descending=True)   # [rc, in]

        bias_before += float((b * b).sum().item())

        for rr in range(rc):
            bj = float(b[rr].item())
            o = order[rr]
            adm = admissible[rr, o]
            n_adm = int(adm.sum().item())
            if n_adm == 0 or bj == 0.0:
                bias_after += bj * bj
                continue
            cap = max(1, math.ceil(budget_p * n_adm))  # B_j = ceil(p|A_j|)
            cand = o[:n_adm][:cap]
            v_cand = v[rr, cand]
            cumv = torch.cumsum(v_cand, dim=0)
            residuals = (bj - cumv).abs()
            best_val, best_k = residuals.min(dim=0)
            if abs(bj) <= float(best_val.item()):
                bias_after += bj * bj
                continue
            k_star = int(best_k.item()) + 1
            chosen = cand[:k_star]
            Wq_corr[r0 + rr, chosen] = target_val[rr, chosen]
            total_flips += k_star
            b_new = bj - float(cumv[k_star - 1].item())
            bias_after += b_new * b_new

            # Diagnostics on the selected flips.
            sel_abs_v = v[rr, chosen].abs()
            sum_selected_abs_v += float(sel_abs_v.sum().item())
            min_selected_abs_v = min(min_selected_abs_v, float(sel_abs_v.min().item()))

        del e, e_sign, gap, v, eta, order, levels, cur, left, right
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if min_selected_abs_v == float("inf"):
        min_selected_abs_v = 0.0

    stats = NCCStats(
        flips=total_flips,
        bias_before=bias_before,
        bias_after=bias_after,
        channels=out_features,
        sum_selected_abs_v=sum_selected_abs_v,
        min_selected_abs_v=min_selected_abs_v,
        num_degenerate_dropped=num_degenerate_dropped,
    )
    return Wq_corr.to(W_fp.dtype), stats


@torch.no_grad()
def per_cell_shift_validation(
    W_fp: torch.Tensor,
    qres: QuantResult,
    log_density_grad: Optional[torch.Tensor] = None,
    row_chunk: int = 1024,
) -> dict:
    """Validation data for the asymmetric-cell residual-bias theorem.

    For every codeword cell that has both neighbours present, returns the realised
    mean residual E[c - w | cell] alongside the theoretical prediction
        pred = -(g+ - g-)/4  -  (g- + g+)^2 / 48 * (log f)'(c),
    so a scatter / regression of realised vs predicted validates the theorem and
    separates the geometric term from the curvature term.

    Parameters
    ----------
    W_fp : [out, in] full-precision weights.
    qres : QuantResult (indices, block info, level grid).
    log_density_grad : optional callable-free tensor of (log f)'(c) per codeword.
        Two accepted forms:
          - None: only the geometric term is used (curvature term set to 0); useful
            to isolate how much of the shift is pure cell asymmetry.
          - [L] or [n_blocks, L]: (log f)'(c) evaluated at each codeword level.
        If a per-channel learned density is unavailable, a fitted log-concave model
        (e.g. Laplace/GG) on the layer's weights can supply (log f)'.

    Returns
    -------
    dict with flat 1-D tensors over all qualifying cells:
        'g_minus', 'g_plus', 'c', 'realized', 'pred_geom', 'pred_curv', 'pred',
        'count' (weights per cell). Aggregate with weighting by 'count'.
    """
    device = W_fp.device
    out_features, in_features = W_fp.shape
    bs = qres.block_size
    n_blocks = (in_features + bs - 1) // bs
    col_block = torch.arange(in_features, device=device) // bs

    g_minus_all, g_plus_all, c_all = [], [], []
    realized_all, count_all = [], []
    pred_curv_all = []

    Wq_full = qres.W_dequant.to(device).float()
    indices_full = qres.indices.to(device)

    for r0 in range(0, out_features, row_chunk):
        r1 = min(r0 + row_chunk, out_features)
        rc = r1 - r0
        Wr = W_fp[r0:r1].float()
        Wq = Wq_full[r0:r1]
        idx = indices_full[r0:r1]
        levels = _block_realised_levels(qres, r0, r1, device)   # [rc, n_blocks, L]
        L = levels.shape[-1]

        blk = col_block.view(1, in_features, 1).expand(rc, in_features, L)
        levels_per_w = torch.gather(levels, 1, blk)            # [rc, in, L]
        cur = torch.gather(levels_per_w, 2, idx.unsqueeze(-1)).squeeze(-1)
        left_idx = (idx - 1).clamp(min=0)
        right_idx = (idx + 1).clamp(max=L - 1)
        left = torch.gather(levels_per_w, 2, left_idx.unsqueeze(-1)).squeeze(-1)
        right = torch.gather(levels_per_w, 2, right_idx.unsqueeze(-1)).squeeze(-1)
        del levels_per_w, blk

        gm = (cur - left).abs()
        gp = (right - cur).abs()
        e = cur - Wr                                           # residual c - w
        both = (idx > 0) & (idx < (L - 1))                     # interior cells only

        # curvature prediction term, if (log f)'(c) supplied
        if log_density_grad is not None:
            lg = log_density_grad.to(device).float()
            if lg.dim() == 1:                                  # [L]
                lg_per_w = lg[idx]                             # [rc, in]
            else:                                             # [n_blocks, L]
                lg_blk = lg[col_block]                         # [in, L]
                lg_per_w = torch.gather(
                    lg_blk.unsqueeze(0).expand(rc, in_features, L),
                    2, idx.unsqueeze(-1)).squeeze(-1)
            pred_curv = -((gm + gp) ** 2) / 48.0 * lg_per_w
        else:
            pred_curv = torch.zeros_like(e)

        m = both
        g_minus_all.append(gm[m]); g_plus_all.append(gp[m]); c_all.append(cur[m])
        realized_all.append(e[m]); pred_curv_all.append(pred_curv[m])
        count_all.append(torch.ones_like(e[m]))

        del levels, cur, left, right, e, gm, gp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    g_minus = torch.cat(g_minus_all); g_plus = torch.cat(g_plus_all)
    c = torch.cat(c_all); realized = torch.cat(realized_all)
    pred_curv = torch.cat(pred_curv_all); count = torch.cat(count_all)
    pred_geom = -(g_plus - g_minus) / 4.0
    pred = pred_geom + pred_curv

    return {
        "g_minus": g_minus, "g_plus": g_plus, "c": c,
        "realized": realized, "pred_geom": pred_geom,
        "pred_curv": pred_curv, "pred": pred, "count": count,
    }