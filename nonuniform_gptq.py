"""
Non-uniform GPTQ adapters for RBVT-squeeze.

This keeps the Hessian-aware sequential error compensation from GPTQ, but uses
nearest-codeword projection on an existing non-uniform codebook. The codebook
can come from the built-in quantizers or from cached upstream LeanQuant /
SqueezeLLM grids.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from quantizers.base_codebook import CodebookContext
from quantizers.base_quantizer import QuantResult


@dataclass
class GPTQStats:
    method: str
    num_linear_layers: int
    skip_lmhead: bool
    total_error: float
    total_time_sec: float
    gptq_blocksize: int
    gptq_percdamp: float
    gptq_act_order: bool


def is_lmhead(name: str) -> bool:
    return "lm_head" in name.lower() or name.endswith("lm_head")


@torch.no_grad()
def _realized_codebooks(qres: QuantResult, device: torch.device) -> torch.Tensor:
    if qres.block_codebooks is not None:
        return qres.block_codebooks.to(device).float()
    q = qres.q_levels.to(device).float()
    scales = qres.block_scales.to(device).float()
    if qres.block_zeros is not None:
        zeros = qres.block_zeros.to(device).float()
        return scales.unsqueeze(-1) * (q.view(1, 1, -1) - zeros.unsqueeze(-1))
    return scales.unsqueeze(-1) * q.view(1, 1, -1)


@torch.no_grad()
def _nearest_codeword(w: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    idx = (w.unsqueeze(1) - levels).abs().argmin(dim=1)
    row_ids = torch.arange(w.shape[0], device=w.device)
    return levels[row_ids, idx]


class NonUniformGPTQ:
    def __init__(self, layer: nn.Linear, qres: QuantResult, reference_weight: torch.Tensor | None = None):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = (reference_weight if reference_weight is not None else layer.weight.data).clone().float()
        self.reference_weight = W
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.qres = qres
        self.codebooks = _realized_codebooks(qres=qres, device=self.dev)
        self.block_size = qres.block_size

    def add_batch(self, inp, _out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    @torch.no_grad()
    def fasterquant(
        self,
        blocksize: int = 128,
        percdamp: float = 0.01,
        actorder: bool = False,
        freeze_mask: torch.Tensor | None = None,
        capture_w_assigned: bool = False,
    ) -> tuple[float, float]:
        W = self.reference_weight.clone()
        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if freeze_mask is not None:
            freeze_mask = freeze_mask.to(device=self.dev, dtype=torch.bool)
            if freeze_mask.shape != W.shape:
                raise ValueError(
                    f"freeze_mask has shape {tuple(freeze_mask.shape)}, expected {tuple(W.shape)}"
                )

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            freeze_mask = freeze_mask[:, perm] if freeze_mask is not None else None
            invperm = torch.argsort(perm)
        else:
            perm = None
            invperm = None

        losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        # Snapshot of error-feedback-adjusted weights at the point of nearest
        # assignment (== what projection actually sees). In permuted coords if
        # actorder; un-permuted before return. Only when capture_w_assigned.
        W_assigned = torch.zeros_like(W) if capture_w_assigned else None

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                global_col = i1 + i
                orig_col = int(perm[global_col].item()) if perm is not None else global_col
                block_idx = orig_col // self.block_size

                w = W1[:, i]
                d = Hinv1[i, i]
                levels = self.codebooks[:, block_idx, :]
                q = _nearest_codeword(w=w, levels=levels)
                if freeze_mask is not None:
                    frozen = freeze_mask[:, global_col]
                    q = torch.where(frozen, w, q)
                Q1[:, i] = q
                if W_assigned is not None:
                    # w here is the adjusted weight nearest-assignment projects.
                    W_assigned[:, i1 + i] = w
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if actorder and invperm is not None:
            Q = Q[:, invperm]
            if W_assigned is not None:
                W_assigned = W_assigned[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        # Keep the GPTQ-projected dequant weights and the adjusted snapshot so an
        # on-top corrector (NCC) can act on the POST-GPTQ codebook assignment.
        self.Q_dequant = Q.detach().clone()
        self.W_assigned = W_assigned.detach().clone() if W_assigned is not None else None
        elapsed = time.time() - tick
        return float(torch.sum(losses).item()), elapsed

    @torch.no_grad()
    def gptq_quant_result(self) -> QuantResult:
        """Build a QuantResult reflecting the POST-GPTQ assignment.

        After fasterquant, module.weight == Q (dequant). NCC needs indices/
        W_dequant of THIS assignment, not the pre-GPTQ qres. We recompute the
        nearest-codeword index of Q per block on the same codebook.
        """
        if not hasattr(self, "Q_dequant"):
            raise RuntimeError("call fasterquant() before gptq_quant_result()")
        Q = self.Q_dequant.to(self.dev).float()           # [out, in]
        out_features, in_features = Q.shape
        bs = self.block_size
        n_blocks = self.codebooks.shape[1]
        L = self.codebooks.shape[2]
        indices = torch.empty(out_features, in_features, dtype=torch.long, device=self.dev)
        block_cb = torch.empty(out_features, n_blocks, L, device=self.dev)
        # per-block nearest index of Q on that block's codebook
        for b in range(n_blocks):
            c0, c1 = b * bs, min((b + 1) * bs, in_features)
            if c0 >= c1:
                continue
            levels = self.codebooks[:, b, :]              # [out, L] (row-shared grid)
            qslice = Q[:, c0:c1]                          # [out, w]
            # nearest level per element
            d = (qslice.unsqueeze(-1) - levels.unsqueeze(1)).abs()   # [out, w, L]
            idx = d.argmin(dim=-1)                        # [out, w]
            indices[:, c0:c1] = idx
            block_cb[:, b, :] = levels
        return QuantResult(
            W_dequant=Q.to(self.layer.weight.dtype),
            indices=indices,
            q_levels=self.qres.q_levels,
            block_scales=self.qres.block_scales,
            block_size=bs,
            block_codebooks=block_cb,
            block_zeros=self.qres.block_zeros,
        )

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


def _linear_layers(model, skip_lmhead: bool) -> List[Tuple[str, nn.Linear]]:
    linears = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if skip_lmhead:
        linears = [(name, module) for name, module in linears if not is_lmhead(name)]
    return linears


@torch.no_grad()
def _load_ncc_apply():
    """Import apply_ncc from the runtime-cloned NCCQuant package."""
    try:
        from NCCQuant.quantizers.ncc import apply_ncc  # type: ignore
        return apply_ncc
    except Exception:
        pass
    try:
        from quantizers.ncc import apply_ncc  # type: ignore
        return apply_ncc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "NCC requested but apply_ncc not importable. Clone NCCQuant "
            "(git clone https://github.com/anhnda/NCCQuant.git) so that "
            "NCCQuant/quantizers/ncc.py is on the path."
        ) from exc


def quantize_model_gptq(
    model,
    tokenizer,
    quantizer,
    calib_texts: List[str],
    device: str,
    skip_lmhead: bool = True,
    n_calib: int = 128,
    max_length: int = 512,
    row_chunk: int = 1024,
    gptq_blocksize: int = 128,
    gptq_percdamp: float = 0.01,
    gptq_act_order: bool = False,
    # --- NCC on-top options ---
    use_ncc: bool = False,
    ncc_baseline: str = "original",          # "original" | "adjusted"
    ncc_score: str = "cov",                  # "cov" | "lite"
    ncc_budget_p: float = 0.02,
    ncc_cov_eps: float = 1e-6,
    ncc_use_james_stein: bool = False,
):
    linears = _linear_layers(model, skip_lmhead=skip_lmhead)
    print(
        f"Quantizing {len(linears)} Linear layers "
        f"({'skipping' if skip_lmhead else 'including'} lm_head) | "
        f"method=gptq{'+ncc' if use_ncc else ''}"
        + (f" | ncc(score={ncc_score}, baseline={ncc_baseline}, p={ncc_budget_p})"
           if use_ncc else "")
    )

    apply_ncc = _load_ncc_apply() if use_ncc else None
    if use_ncc and ncc_baseline not in ("original", "adjusted"):
        raise ValueError(f"ncc_baseline must be original|adjusted, got {ncc_baseline}")

    total_error = 0.0
    total_time_sec = 0.0
    ncc_total_flips = 0
    ncc_bias_before = 0.0
    ncc_bias_after = 0.0

    for name, module in tqdm(linears, desc="Quantizing layers"):
        weight = module.weight.data
        qres = quantizer.quantize(weight, row_chunk=row_chunk)
        capture = use_ncc and ncc_baseline == "adjusted"
        gptq = NonUniformGPTQ(layer=module, qres=qres, reference_weight=weight)

        # Accumulate input-activation mean/var for NCC IN THE SAME calib pass,
        # so they reflect the activations this layer actually sees AFTER upstream
        # layers have been quantized (sequential). A stale one-shot pre-pass on
        # the full-precision model gives the wrong mu for deep layers.
        want_var = use_ncc and ncc_score == "cov"
        stat_sum = {"s": None, "sq": None, "n": 0}

        def add_batch(_m, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            gptq.add_batch(x.data, out.data)
            if use_ncc:
                xf = x.reshape(-1, x.shape[-1]).detach().float()
                s = xf.sum(0)
                stat_sum["s"] = s if stat_sum["s"] is None else stat_sum["s"] + s
                stat_sum["n"] += xf.shape[0]
                if want_var:
                    sq = (xf * xf).sum(0)
                    stat_sum["sq"] = sq if stat_sum["sq"] is None else stat_sum["sq"] + sq

        handle = module.register_forward_hook(add_batch)
        try:
            for i, text in enumerate(calib_texts[:n_calib]):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model(**inputs, use_cache=False)
                if (i + 1) % 16 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            handle.remove()

        err, elapsed = gptq.fasterquant(
            blocksize=gptq_blocksize,
            percdamp=gptq_percdamp,
            actorder=gptq_act_order,
            capture_w_assigned=capture,
        )
        total_error += err
        total_time_sec += elapsed

        # ---- NCC correction on top of the GPTQ assignment ----
        if use_ncc and stat_sum["n"] > 0:
            W_ref = gptq.reference_weight.to(device).float()        # original fp
            cnt = max(1, stat_sum["n"])
            mu = (stat_sum["s"] / cnt).to(device).float()
            sigma_ii = None
            if want_var and stat_sum["sq"] is not None:
                ex2 = (stat_sum["sq"] / cnt).to(device).float()
                sigma_ii = (ex2 - mu * mu).clamp(min=0.0)

            # baseline NCC corrects against
            if ncc_baseline == "adjusted":
                W_base = gptq.W_assigned
                if W_base is None:
                    raise RuntimeError(f"{name}: W_assigned missing (capture failed).")
                W_base = W_base.to(device).float()
            else:
                W_base = W_ref

            qres_post = gptq.gptq_quant_result()
            mu_var_js = (sigma_ii / cnt) if (ncc_use_james_stein and sigma_ii is not None) else None

            W_corr, stats = apply_ncc(
                W_fp=W_base,
                qres=qres_post,
                mu=mu,
                budget_p=ncc_budget_p,
                use_james_stein=ncc_use_james_stein,
                mu_var=mu_var_js,
                row_chunk=row_chunk,
                score=ncc_score,
                sigma_ii=sigma_ii if ncc_score == "cov" else None,
                cov_eps=ncc_cov_eps,
            )
            module.weight.data = W_corr.reshape(module.weight.shape).to(module.weight.dtype)
            ncc_total_flips += int(getattr(stats, "flips", 0))
            ncc_bias_before += float(getattr(stats, "bias_before", 0.0))
            ncc_bias_after += float(getattr(stats, "bias_after", 0.0))

        gptq.free()
        print(f"GPTQ layer done | {name} | error={err:.6e} | time={elapsed:.2f}s")

    print(f"GPTQ summary | total_error={total_error:.6e} | total_time_sec={total_time_sec:.2f}")
    if use_ncc:
        print(
            f"NCC summary | flips={ncc_total_flips} | "
            f"bias_before={ncc_bias_before:.6e} -> bias_after={ncc_bias_after:.6e}"
        )
    return model, GPTQStats(
        method="gptq+ncc" if use_ncc else "gptq",
        num_linear_layers=len(linears),
        skip_lmhead=skip_lmhead,
        total_error=total_error,
        total_time_sec=total_time_sec,
        gptq_blocksize=gptq_blocksize,
        gptq_percdamp=gptq_percdamp,
        gptq_act_order=gptq_act_order,
    )


@torch.no_grad()
def quantize_codebook_model_gptq(
    model,
    tokenizer,
    codebook,
    codebook_store,
    calib_texts: List[str],
    device: str,
    skip_lmhead: bool = True,
    n_calib: int = 128,
    max_length: int = 512,
    row_chunk: int = 1024,
    gptq_blocksize: int = 128,
    gptq_percdamp: float = 0.01,
    gptq_act_order: bool = False,
    sparse_store=None,
):
    linears = _linear_layers(model, skip_lmhead=skip_lmhead)
    print(
        f"Quantizing {len(linears)} Linear layers "
        f"({'skipping' if skip_lmhead else 'including'} lm_head) | method=gptq"
    )

    total_error = 0.0
    total_time_sec = 0.0
    total_sparse_values = 0

    for name, module in tqdm(linears, desc="Quantizing layers"):
        weight = module.weight.data
        sparse_residual = None
        sparse_mask = None
        dense_weight = weight
        if sparse_store is not None:
            sparse_residual = sparse_store.get(name, device=weight.device).to(weight.dtype)
            sparse_mask = sparse_residual != 0
            dense_weight = weight - sparse_residual
            total_sparse_values += int(sparse_mask.sum().item())

        cached_centers = codebook_store.get(name)
        codebook.set_context(CodebookContext(precomputed_centers=cached_centers))
        qres = codebook.quantize(dense_weight, row_chunk=row_chunk)
        gptq = NonUniformGPTQ(layer=module, qres=qres, reference_weight=dense_weight)

        def add_batch(_m, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            gptq.add_batch(x.data, out.data)

        handle = module.register_forward_hook(add_batch)
        try:
            for i, text in enumerate(calib_texts[:n_calib]):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model(**inputs, use_cache=False)
                if (i + 1) % 16 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            handle.remove()

        err, elapsed = gptq.fasterquant(
            blocksize=gptq_blocksize,
            percdamp=gptq_percdamp,
            actorder=gptq_act_order,
            freeze_mask=sparse_mask,
        )
        output = module.weight.data
        if sparse_residual is not None:
            output = output + sparse_residual
        module.weight.data = output.to(weight.dtype)
        codebook.set_context(None)
        gptq.free()

        total_error += err
        total_time_sec += elapsed
        print(f"GPTQ layer done | {name} | error={err:.6e} | time={elapsed:.2f}s")

    print(f"GPTQ summary | total_error={total_error:.6e} | total_time_sec={total_time_sec:.2f}")
    if sparse_store is not None:
        print(f"SqueezeLLM sparse summary | restored_values={total_sparse_values}")
    return {
        "method": "gptq",
        "num_linear_layers": len(linears),
        "skip_lmhead": skip_lmhead,
        "codebook": codebook.name,
        "bits": codebook.bits,
        "sparse_values": total_sparse_values,
        "total_error": total_error,
        "total_time_sec": total_time_sec,
        "gptq_blocksize": gptq_blocksize,
        "gptq_percdamp": gptq_percdamp,
        "gptq_act_order": gptq_act_order,
    }