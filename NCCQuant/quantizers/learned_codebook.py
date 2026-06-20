"""
Learned scalar codebook quantizer: Codebook3 / Codebook4 — block-wise (standard).

A learned codebook drops the fixed shape: the 2**bits levels are searched from the
weights themselves by 1-D k-means (Lloyd) per (row, block). This is inherently
asymmetric — centers are free to sit anywhere — so the global ASYM flag does not
change it. Realised levels are stored in block_codebooks for NCC to read.
"""

from __future__ import annotations

import torch

from .base_quantizer import BaseQuantizer, QuantResult


class LearnedCodebookQuantizer(BaseQuantizer):
    """Per-(row, block) learned scalar codebook via 1-D k-means."""

    def __init__(self, bits: int, block_size: int = 64, n_iters: int = 20, seed: int = 0):
        if bits not in (3, 4):
            raise ValueError(f"LearnedCodebook supports bits in {{3,4}}, got {bits}")
        super().__init__(bits=bits, block_size=block_size)
        self.name = f"codebook{bits}"
        self.num_levels = 2 ** bits
        self.n_iters = n_iters
        self.seed = seed

    @property
    def q_levels(self) -> torch.Tensor:
        # No canonical shape for a learned codebook; expose a placeholder grid so
        # callers that only need L work. Realised levels live in block_codebooks.
        return torch.linspace(-1.0, 1.0, self.num_levels)

    @torch.no_grad()
    def _kmeans_blocks(self, Wb: torch.Tensor) -> torch.Tensor:
        """Vectorised 1-D Lloyd k-means over a stack of blocks.

        Wb : [G, bw]  (G = rows-in-chunk * n_blocks flattened, bw = block width)
        returns centers [G, K] sorted ascending, strictly increasing.
        """
        G, bw = Wb.shape
        K = self.num_levels
        device = Wb.device

        qs = torch.linspace(0.0, 1.0, K, device=device, dtype=torch.float32)
        centers = torch.quantile(Wb, qs, dim=1).t().contiguous()   # [G, K]
        zc = centers.abs().argmin(dim=1)
        centers[torch.arange(G, device=device), zc] = 0.0
        centers, _ = torch.sort(centers, dim=1)

        for _ in range(self.n_iters):
            d = (Wb.unsqueeze(-1) - centers.unsqueeze(1)).abs()    # [G, bw, K]
            assign = d.argmin(dim=-1)                              # [G, bw]
            del d
            new_centers = centers.clone()
            for k in range(K):
                mask = (assign == k)
                cnt = mask.sum(dim=1).clamp(min=1)
                summ = (Wb * mask).sum(dim=1)
                mean_k = summ / cnt
                owns = mask.any(dim=1)
                new_centers[:, k] = torch.where(owns, mean_k, centers[:, k])
            centers, _ = torch.sort(new_centers, dim=1)

        # Strictly increasing (NCC neighbour reads require it).
        eps = 1e-7
        for k in range(1, K):
            bad = centers[:, k] <= centers[:, k - 1]
            centers[:, k] = torch.where(bad, centers[:, k - 1] + eps, centers[:, k])
        return centers

    @torch.no_grad()
    def quantize(self, W: torch.Tensor, row_chunk: int = 1024) -> QuantResult:
        torch.manual_seed(self.seed)
        device = W.device
        out_features, in_features = W.shape
        K = self.num_levels
        bs = self.block_size
        n_blocks = (in_features + bs - 1) // bs

        W_dequant = torch.empty_like(W)
        indices = torch.empty(out_features, in_features, dtype=torch.long, device=device)
        block_codebooks = torch.zeros(out_features, n_blocks, K, device=device, dtype=torch.float32)
        block_scales = torch.zeros(out_features, n_blocks, device=device, dtype=torch.float32)

        for r0 in range(0, out_features, row_chunk):
            r1 = min(r0 + row_chunk, out_features)
            Wr = W[r0:r1].float()
            rc = r1 - r0
            for b in range(n_blocks):
                c0 = b * bs
                c1 = min(c0 + bs, in_features)
                Wb = Wr[:, c0:c1]                              # [rc, bw]
                centers = self._kmeans_blocks(Wb)             # [rc, K]
                block_codebooks[r0:r1, b, :] = centers
                block_scales[r0:r1, b] = Wb.abs().amax(dim=1)

                diff = (Wb.unsqueeze(-1) - centers.unsqueeze(1)).abs()  # [rc, bw, K]
                idx = diff.argmin(dim=-1)
                deq = torch.gather(centers, 1, idx)
                W_dequant[r0:r1, c0:c1] = deq.to(W.dtype)
                indices[r0:r1, c0:c1] = idx
                del diff

        return QuantResult(
            W_dequant=W_dequant,
            indices=indices,
            q_levels=self.q_levels.to(device),
            block_scales=block_scales,
            block_size=bs,
            block_codebooks=block_codebooks,   # realised per-(row,block) levels
            block_zeros=None,                  # centers already encode any shift
        )