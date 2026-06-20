"""
Base interface for non-uniform per-block scalar codebook quantizers.

Now supports a global ASYM flag. When ASYM is True, factorised formats (NF/NVFP4)
use an affine per-block mapping s*(q - z) with a per-block zero-point z fit to the
block's [min, max], instead of the symmetric absmax mapping s*q. The realised
levels per block become s*(q - z), which NCC reads via block_codebooks (we
materialise them in asym mode so the neighbour query stays correct).

Layout convention
-----------------
"channel" = output channel = a ROW of nn.Linear weight [out_features, in_features].
Blocks partition the INPUT dimension (columns). Each (row, block) owns a scale
(and, in asym mode, a zero-point).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch

# --------------------------------------------------------------------------- #
# Global asymmetric-quantization flag.
# Symmetric (False): grid = scale * q,        scale = absmax / max|q|.
# Asymmetric (True):  grid = scale * (q - z),  fit to block [min, max].
# --------------------------------------------------------------------------- #
ASYM: bool = True


@dataclass
class QuantResult:
    """Result of quantizing a single weight matrix W [out, in].

    Attributes
    ----------
    W_dequant : [out, in]
        Dequantized weights Wq (same dtype as input W).
    indices : [out, in] long
        Per-weight codeword index into that weight's *block* codebook.
    q_levels : [L]
        Canonical normalised level shape (max|level| = 1). Shared across blocks
        for symmetric factorised formats. In asym mode the realised levels are
        per-block and live in block_codebooks.
    block_scales : [out, n_blocks]
        Per-(row, block) scale s_{j,b}.
    block_size : int
        Number of input columns per block.
    block_codebooks : Optional[[out, n_blocks, L]]
        Materialised realised levels per (row, block) when a format does not
        factorise as scale*shape (learned codebooks, or any asym factorised
        format). When None, reconstruct as block_scales[:, b, None] * q_levels.
    block_zeros : Optional[[out, n_blocks]]
        Per-(row, block) zero-point z (in q-units) for asym factorised formats.
        Realised level for canonical level q is scale*(q - z). None for symmetric.
    """

    W_dequant: torch.Tensor
    indices: torch.Tensor
    q_levels: torch.Tensor
    block_scales: torch.Tensor
    block_size: int
    block_codebooks: Optional[torch.Tensor] = None
    block_zeros: Optional[torch.Tensor] = None


class BaseQuantizer(ABC):
    """Abstract non-uniform per-block scalar-codebook quantizer.

    Subclasses provide the canonical level shape via `q_levels`. The shared
    block-wise nearest-codeword `quantize` honours the global ASYM flag: in asym
    mode it fits a per-block (scale, zero) affine map of the canonical levels onto
    the block's [min, max] and materialises the realised levels into
    block_codebooks so NCC's realised-neighbour query is exact.
    """

    name: str = "base"

    def __init__(self, bits: int, block_size: int = 64):
        self.bits = bits
        self.block_size = block_size

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #
    @property
    @abstractmethod
    def q_levels(self) -> torch.Tensor:
        """Canonical level shape, 1-D, sorted ascending, normalised max|q| = 1."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared block-wise nearest-codeword assignment (factorised formats)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def quantize(self, W: torch.Tensor, row_chunk: int = 1024) -> QuantResult:
        """Block-wise nearest-codeword quantization of W [out, in].

        Symmetric (ASYM=False): per (row, block) absmax scale s_{j,b} =
        max|W in block| / max|q|; grid = s * q.

        Asymmetric (ASYM=True): fit the canonical level span [q_min, q_max] onto
        the block's [w_min, w_max]:
            scale = (w_max - w_min) / (q_max - q_min)
            z     = q_min - w_min / scale          (zero-point in q-units)
            grid  = scale * (q - z)                 (so grid spans [w_min, w_max])
        Realised per-block levels are materialised into block_codebooks.

        Processed in row chunks to bound the transient [chunk, in, L] tensor.
        Result is bit-identical regardless of chunk size (rows independent).
        """
        device = W.device
        out_features, in_features = W.shape
        q = self.q_levels.to(device).float()                 # [L]
        L = q.numel()
        qmax = q.abs().max().clamp(min=1e-12)
        qlo = q.min()
        qhi = q.max()
        qspan = (qhi - qlo).clamp(min=1e-12)
        bs = self.block_size
        n_blocks = (in_features + bs - 1) // bs

        W_dequant = torch.empty_like(W)
        indices = torch.empty(out_features, in_features, dtype=torch.long, device=device)
        block_scales = torch.empty(out_features, n_blocks, device=device, dtype=torch.float32)
        block_zeros = (
            torch.zeros(out_features, n_blocks, device=device, dtype=torch.float32)
            if ASYM else None
        )
        block_codebooks = (
            torch.zeros(out_features, n_blocks, L, device=device, dtype=torch.float32)
            if ASYM else None
        )

        for r0 in range(0, out_features, row_chunk):
            r1 = min(r0 + row_chunk, out_features)
            Wr = W[r0:r1].float()                              # [rc, in]
            rc = r1 - r0
            for b in range(n_blocks):
                c0 = b * bs
                c1 = min(c0 + bs, in_features)
                Wb = Wr[:, c0:c1]                              # [rc, bw]

                if ASYM:
                    wmin = Wb.amin(dim=1, keepdim=True)        # [rc,1]
                    wmax = Wb.amax(dim=1, keepdim=True)        # [rc,1]
                    scale = ((wmax - wmin) / qspan).clamp(min=1e-12)  # [rc,1]
                    z = qlo - wmin / scale                     # [rc,1] zero-point (q-units)
                    block_scales[r0:r1, b] = scale.squeeze(1)
                    block_zeros[r0:r1, b] = z.squeeze(1)
                    grid = scale * (q.unsqueeze(0) - z)        # [rc, L]
                    block_codebooks[r0:r1, b, :] = grid
                else:
                    absmax = Wb.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)  # [rc,1]
                    scale = absmax / qmax                      # [rc,1]
                    block_scales[r0:r1, b] = scale.squeeze(1)
                    grid = scale * q.unsqueeze(0)              # [rc, L]

                diff = (Wb.unsqueeze(-1) - grid.unsqueeze(1)).abs()  # [rc, bw, L]
                idx = diff.argmin(dim=-1)                      # [rc, bw]
                deq = torch.gather(grid, 1, idx)               # [rc, bw]
                W_dequant[r0:r1, c0:c1] = deq.to(W.dtype)
                indices[r0:r1, c0:c1] = idx
                del diff, grid

        return QuantResult(
            W_dequant=W_dequant,
            indices=indices,
            q_levels=q,
            block_scales=block_scales,
            block_size=bs,
            block_codebooks=block_codebooks,   # None (sym) or realised levels (asym)
            block_zeros=block_zeros,           # None (sym) or per-block zero-points
        )

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name={self.name!r}, bits={self.bits}, "
                f"block_size={self.block_size}, asym={ASYM})")