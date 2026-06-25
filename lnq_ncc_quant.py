"""
lnq_ncc_quant.py — production LNQ (GuidedQuant Layer-wise Non-uniform Quantization)
backbone + NCC first-moment correction, wired into main.py as `--method lnq`.

This is the model-wide analogue of debug_ncc_mse_lnq.py. The debug harness only
touches the first --max-layers blocks and asserts the bias / awMSE invariants
layer by layer; this module reuses the SAME verified primitives (the real
`train_least_squares` from GuidedQuant, NCCQuant's `apply_ncc` + `QuantResult`,
the `_lnq_quant_result` bridge, the off-spec `apply_ncc_full_sigma` corrector and
the SqueezeLLM-style k-means init) but runs over EVERY nn.Linear in the decoder
blocks, writes the corrected weights back in place, and returns an aggregate
stats dict so main.py can save_pretrained() + evaluate exactly like the other
methods. No re-implementation of LNQ or NCC — the interfaces never drift from the
debug harness, which is where the theory is checked.

mse-guard-mode (default: full-greedy):
  diag         faithful published NCC: apply_ncc; --ncc-mse-guard toggles the
               diagonal Cor-2 gate (gap < 2|e|).
  full-screen  off-spec: apply_ncc_full_sigma with the pre-flip (H e) screen.
  full-greedy  off-spec (DEFAULT): apply_ncc_full_sigma with the exact rank-1
               (H e) update -> full awMSE provably non-increasing.

The off-spec full-* modes guarantee the full activation-weighted MSE never rises
while still reducing the first-moment bias; the diag mode is the published
separable rule (diagonal certificate only). See README_LNQ_NCC.md for the theory.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn

# The debug harness already contains the verified, production-faithful building
# blocks (lazy loaders for the real LNQ optimiser + NCCQuant, the QuantResult
# bridge, the full-Sigma corrector, the k-means init, the metric helpers). We
# import them rather than copy so this backbone can never drift from the harness
# the theory is validated against. The harness guards all heavy work behind
# functions / `if __name__ == "__main__"`, so importing it is side-effect free.
import debug_ncc_mse_lnq as _dbg


@torch.no_grad()
def _collect_layer_stats(
    *,
    model,
    tokenizer,
    module: nn.Linear,
    calib_texts: list[str],
    args,
    device,
):
    """Calibrate ONE linear: return (H, mu, sigma, X, cnt).

    Identical capture logic to debug_ncc_mse_lnq.main()'s per-layer hooks:
      H     = X^T X         (plain layer-wise LNQ objective, Eq. 1), or
              X^T diag(s) X (GuidedQuant saliency, Eq. 7, g=1) when --guided,
      mu    = E[x]          (activation mean),
      sigma = Var[x]        (diag of Sigma; the eta="cov" ordering uses this),
      X     = bounded activation sample for the awMSE diagnostics (or None),
      cnt   = token count (for the optional James-Stein shrinkage).
    """
    in_features = module.weight.shape[1]
    Hs = {"H": None, "s": None, "sq": None, "n": 0}
    xcache: list[torch.Tensor] = []
    sal = {"acc": None, "n": 0}
    keep_x = getattr(args, "lnq_keep_activation_sample", True)
    diag_max_tokens = getattr(args, "diag_max_tokens", 4096)

    def fwd_hook(_m, inp, out):
        x = inp[0] if isinstance(inp, tuple) else inp
        xf = x.reshape(-1, x.shape[-1]).detach().float()
        HxtX = xf.t() @ xf
        Hs["H"] = HxtX if Hs["H"] is None else Hs["H"] + HxtX
        s = xf.sum(0)
        Hs["s"] = s if Hs["s"] is None else Hs["s"] + s
        sq = (xf * xf).sum(0)
        Hs["sq"] = sq if Hs["sq"] is None else Hs["sq"] + sq
        Hs["n"] += xf.shape[0]
        if keep_x:
            kept = sum(t.shape[0] for t in xcache)
            if kept < diag_max_tokens:
                xcache.append(xf[: diag_max_tokens - kept].detach().cpu().clone())

    h_fwd = module.register_forward_hook(fwd_hook)
    h_bwd = None
    if args.guided:
        def out_bwd_hook(_m, grad_in, grad_out):
            g = grad_out[0] if isinstance(grad_out, tuple) else grad_out
            if g is None:
                return
            gf = g.reshape(-1, g.shape[-1]).detach().float()
            acc = (gf * gf).sum(0)
            sal["acc"] = acc if sal["acc"] is None else sal["acc"] + acc
            sal["n"] += gf.shape[0]
        h_bwd = module.register_full_backward_hook(out_bwd_hook)

    try:
        for text in calib_texts[: args.n_calib]:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
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
        s_bar = float((sal["acc"].sum() / (sal["n"] * sal["acc"].numel())).item())
        H = H * max(s_bar, 1e-30)

    diag_mean = torch.diag(H).mean().clamp(min=1e-12)
    H = H + (args.percdamp * diag_mean) * torch.eye(in_features, device=device)

    X = torch.cat(xcache, 0).to(device) if xcache else None
    return H, mu, sigma, X, cnt


@torch.no_grad()
def _quantize_one_linear(
    *,
    name: str,
    module: nn.Linear,
    train_least_squares,
    QuantResult,
    apply_ncc,
    H: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    X: torch.Tensor | None,
    cnt: int,
    args,
    device,
) -> dict:
    """LNQ-quantize a single linear, apply NCC, write back. Returns a row dict.

    This is the production sibling of debug_ncc_mse_lnq.run_lnq_layer: same LNQ
    init -> train_least_squares -> _lnq_quant_result bridge -> NCC corrector, but
    the assert-heavy diagnostics are downgraded to soft warnings so a single
    pathological layer cannot abort a full-model quantization run.
    """
    weight = module.weight.data.detach().clone().float()
    out_features, in_features = weight.shape

    init_labels, init_centroids = _dbg.kmeans_init_per_channel(
        weight.to(device), args.wbits, args.kmeans_iters
    )
    init_labels_np = init_labels.detach().cpu().numpy()
    init_centroids_np = init_centroids.detach().cpu().numpy()
    weight_np = weight.detach().cpu().numpy()
    H_lnq = H.detach().to(device).float().unsqueeze(0).cpu().numpy()

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

    labels = torch.from_numpy(labels).long().to(device)
    C = torch.from_numpy(C).float().to(device)
    W_lnq = torch.gather(C, 1, labels).float()

    qres = _dbg._lnq_quant_result(
        QuantResult=QuantResult,
        W_dequant=W_lnq,
        labels=labels,
        centroids=C,
        bits=args.wbits,
    )

    W_fp = weight.to(device)
    W_base = W_fp

    bias_before = _dbg.bias_metric(W_base, W_lnq, mu)
    awmse_before = _dbg.activation_weighted_mse(W_base, W_lnq, X) if X is not None else float("nan")
    awmse_orig_before = _dbg.activation_weighted_mse(W_fp, W_lnq, X) if X is not None else float("nan")

    mu_var_js = (sigma / cnt) if args.ncc_use_james_stein else None
    if args.mse_guard_mode == "diag":
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
        flips = int(getattr(stats, "flips", -1))
        s_bb = getattr(stats, "bias_before", None)
        s_ba = getattr(stats, "bias_after", None)
        blocked = {}
    else:
        mode = "full-greedy" if args.mse_guard_mode == "full-greedy" else "full-screen"
        W_corr, stats = _dbg.apply_ncc_full_sigma(
            W_fp=W_base,
            qres=qres,
            H=H,
            mu=mu,
            sigma_ii=sigma,
            budget_p=args.ncc_budget_p,
            score=args.score,
            cov_eps=args.cov_eps,
            mode=mode,
            row_chunk=args.row_chunk,
        )
        flips = int(stats["flips"])
        s_bb = stats["bias_before"]
        s_ba = stats["bias_after"]
        blocked = {"awmse_blocked": stats["awmse_blocked"], "overshoot_blocked": stats["overshoot_blocked"]}

    W_corr = W_corr.float()
    bias_after = _dbg.bias_metric(W_base, W_corr, mu)
    awmse_after = _dbg.activation_weighted_mse(W_base, W_corr, X) if X is not None else float("nan")
    awmse_orig_after = _dbg.activation_weighted_mse(W_fp, W_corr, X) if X is not None else float("nan")

    # Soft invariant checks (warn, do not abort a full-model run).
    if bias_after > bias_before * (1 + 1e-6) + 1e-12:
        print(f"  [warn] {name}: bias increased {bias_before:.3e} -> {bias_after:.3e} "
              f"(Thm-3 / no-overshoot violated on realised tensor).")
    if (args.mse_guard_mode == "full-greedy" and X is not None
            and awmse_before == awmse_before
            and awmse_after > awmse_before * (1 + 1e-4) + 1e-12):
        print(f"  [warn] {name}: full awMSE increased {awmse_before:.3e} -> {awmse_after:.3e} "
              f"under full-greedy (apply_ncc_full_sigma He/Delta bug).")

    # write back so later blocks calibrate on corrected weights
    module.weight.data = W_corr.reshape(module.weight.shape).to(module.weight.dtype)

    def pct(a, b):
        return (b - a) / a * 100.0 if a not in (0.0, float("nan")) else float("nan")

    extra = ""
    if blocked:
        extra = f" | blocked(awmse={blocked['awmse_blocked']},overshoot={blocked['overshoot_blocked']})"
    print(
        f"  {name:<34} [{args.mse_guard_mode}] flips={flips:>7}{extra} | "
        f"bias {bias_before:.3e}->{bias_after:.3e} ({pct(bias_before, bias_after):+.2f}%) | "
        f"awMSE[orig] {awmse_orig_before:.3e}->{awmse_orig_after:.3e} "
        f"({pct(awmse_orig_before, awmse_orig_after):+.2f}%) | lnq={dt_lnq:.1f}s"
    )

    row = {
        "layer": name,
        "flips": flips,
        "bias_before": bias_before,
        "bias_after": bias_after,
        "awmse_before": awmse_before,
        "awmse_after": awmse_after,
        "awmse_orig_before": awmse_orig_before,
        "awmse_orig_after": awmse_orig_after,
        "lnq_seconds": dt_lnq,
    }
    row.update(blocked)
    return row


@torch.no_grad()
def quantize_model_lnq(
    *,
    model,
    tokenizer,
    calib_texts: list[str],
    args,
    device,
) -> dict:
    """Quantize every decoder-block nn.Linear with LNQ + NCC, in place.

    Returns an aggregate stats dict (the shape main.py serialises into
    run_summary.json["quantization"], and run_lnq_ncc.sh greps for perplexity /
    method / flips / bias_before / bias_after).
    """
    apply_ncc, QuantResult = _dbg._load_ncc()
    train_least_squares = _dbg._load_lnq()

    if not (hasattr(model, "model") and hasattr(model.model, "layers")):
        raise RuntimeError("LNQ backbone expects a Llama-like model with model.model.layers")

    model.config.use_cache = False
    blocks = model.model.layers
    max_layers = getattr(args, "max_layers", 0) or len(blocks)
    n_blocks = min(max_layers, len(blocks))

    targets: list[tuple[str, nn.Linear]] = []
    for bi in range(n_blocks):
        for nm, m in blocks[bi].named_modules():
            if isinstance(m, nn.Linear):
                targets.append((f"layers.{bi}.{nm}", m))

    print(f"[lnq] backbone=lnq wbits={args.wbits} guided={args.guided} "
          f"lnq_iters={args.lnq_iters} cd_cycles={args.cd_cycles} "
          f"score={args.score} budget_p={args.ncc_budget_p} "
          f"mse_guard_mode={args.mse_guard_mode} mse_guard={args.mse_guard} | "
          f"{len(targets)} linears across {n_blocks} blocks")

    rows: list[dict] = []
    for name, module in targets:
        H, mu, sigma, X, cnt = _collect_layer_stats(
            model=model, tokenizer=tokenizer, module=module,
            calib_texts=calib_texts, args=args, device=device,
        )
        row = _quantize_one_linear(
            name=name, module=module,
            train_least_squares=train_least_squares,
            QuantResult=QuantResult, apply_ncc=apply_ncc,
            H=H, mu=mu, sigma=sigma, X=X, cnt=cnt, args=args, device=device,
        )
        rows.append(row)
        del H, mu, sigma, X
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- aggregate ----
    tot_flips = sum(r["flips"] for r in rows)
    tot_bias_b = sum(r["bias_before"] for r in rows)
    tot_bias_a = sum(r["bias_after"] for r in rows)
    valid = [r for r in rows if r["awmse_orig_before"] == r["awmse_orig_before"]]
    tot_awo_b = sum(r["awmse_orig_before"] for r in valid)
    tot_awo_a = sum(r["awmse_orig_after"] for r in valid)
    tot_awmse_blocked = sum(r.get("awmse_blocked", 0) for r in rows)
    tot_overshoot_blocked = sum(r.get("overshoot_blocked", 0) for r in rows)

    def _net(b, a):
        return (a - b) / b * 100.0 if b > 0 else float("nan")

    print("\n================ LNQ + NCC SUMMARY ================")
    print(f"layers quantized      : {len(rows)}")
    print(f"total flips           : {tot_flips}")
    if tot_bias_b > 0:
        print(f"total bias            : {tot_bias_b:.4e} -> {tot_bias_a:.4e} ({_net(tot_bias_b, tot_bias_a):+.2f}%)")
    if tot_awo_b > 0:
        v = "NET WIN" if tot_awo_a <= tot_awo_b else "NET REGRESSION"
        print(f"total act-wMSE[orig]  : {tot_awo_b:.4e} -> {tot_awo_a:.4e} "
              f"({_net(tot_awo_b, tot_awo_a):+.2f}%)  [{v}]  <- TRUE inference error")
    if args.mse_guard_mode in ("full-screen", "full-greedy"):
        print(f"blocked(awmse={tot_awmse_blocked}, overshoot={tot_overshoot_blocked})")
    print("===================================================\n")

    stats = {
        "method": "lnq",
        "backbone": "lnq",
        "bits": args.wbits,
        "guided": bool(args.guided),
        "lnq_iters": args.lnq_iters,
        "cd_cycles": args.cd_cycles,
        "score": args.score,
        "mse_guard_mode": args.mse_guard_mode,
        "mse_guard": bool(args.mse_guard),
        "ncc_budget_p": args.ncc_budget_p,
        "layers_quantized": len(rows),
        "flips": tot_flips,
        "bias_before": tot_bias_b,
        "bias_after": tot_bias_a,
        "awmse_orig_before": tot_awo_b,
        "awmse_orig_after": tot_awo_a,
        "awmse_blocked": tot_awmse_blocked,
        "overshoot_blocked": tot_overshoot_blocked,
        "per_layer": rows,
    }
    return stats