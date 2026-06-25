# debug_ncc_mse_lnq.py — LNQ + NCC debug harness

A standalone harness that sits **NCC** (Non-uniform Codebook Correction) on top of
**LNQ** (the GuidedQuant Layer-wise Non-uniform Quantization algorithm) and measures,
per layer, whether the first-moment **bias** goes down without the **activation-weighted
MSE** going up. It is the LNQ analogue of `debug_ncc_bias_mae_mse.py` (which uses the
GPTVQ-1D / NF backends).

It only touches the first `--max-layers` decoder blocks, so it is cheap and meant for
diagnosis, not production quantization.

---

## What it does, end to end

For each `nn.Linear` in the first N decoder blocks:

1. **Calibrate.** Run the model on a few WikiText-2 sequences, capturing per layer:
   - the layer-input Hessian `H = Xᵀ X` (the plain layer-wise LNQ objective, Eq. 1),
   - the activation mean `mu` and per-channel variance `sigma` (= diag of Σ),
   - a bounded sample `X` of activations for the activation-weighted MSE metric.
   With `--guided`, a single backward pass on the LM loss adds the GuidedQuant saliency
   so the Hessian becomes `H = Xᵀ diag(s) X` (Eq. 7, g = 1) with
   `s = mean over output channels of (∂loss/∂z_j)²`.

2. **Quantize with LNQ.** Seed a per-output-channel codebook by SqueezeLLM-style 1-D
   k-means, then run the **real** `train_least_squares` from GuidedQuant
   (`any_precision.quantization.layerwise_quantize`): closed-form codebook update +
   cyclic coordinate-descent assignment update (Algorithm 2). No re-implementation.

3. **Bridge to NCC.** LNQ emits per-channel `(centroids C [out, K], labels [out, in])`.
   These are converted into NCC's block-wise `QuantResult` as a **single block** spanning
   the whole input dim (`block_size = in_features`), with `block_codebooks = sorted(C)` and
   labels remapped through the sort permutation. This is the LNQ single-block analogue of
   `gptvq_rbvt_benchmark._gptvq_quant_result`.

4. **Apply NCC** (one of three modes — see below) and write the corrected weights back so
   later blocks see them.

5. **Measure and assert.** Bias, MAE, MSE, full activation-weighted MSE (vs the LNQ
   baseline and vs the original fp = true inference error), and the **diagonal**
   activation-weighted MSE. Invariants are asserted on the realised tensor.

---

## The three correction modes (`--mse-guard-mode`)

| mode | corrector | what it guarantees | cost |
|------|-----------|--------------------|------|
| `diag` (default) | the **published** `apply_ncc` from NCCQuant | bias ↓ (Thm 3); with `--mse-guard` the **diagonal** awMSE ↓ (Cor 2) | cheap, vectorized |
| `full-screen` | off-spec `apply_ncc_full_sigma` | bias ↓; full awMSE gated by a first-order screen (no flip coupling) | medium |
| `full-greedy` | off-spec `apply_ncc_full_sigma` | bias ↓ **and** full awMSE never ↑ (exact) | slow (per-row Python loop) |

### Why the modes exist

The published NCC selection rule is **separable** and uses only the **diagonal** of Σ:
the score `η = |mu_i| / ((Σ_ii + ε) g)` and the optional `--mse-guard` (`gap < 2|e|`,
Corollary 2) bound only the **diagonal** activation-weighted MSE
`Σ_i Σ_ii e_i²`. On layers whose Σ has strong **off-diagonal** mass — exactly what the
GuidedQuant Fisher figures show for LLM linear layers — the diagonal certificate is silent
about the cross-term

```
ΔawMSE = 2⟨Δ, Σe⟩ + tr(ΔᵀΣΔ)
       = Σ_i Σ_ii g (g − 2|e_i|)        ← diagonal, what mse_guard controls
       + Σ_{i≠k} Δ_i Σ_ik (2e_k + Δ_k)  ← OFF-DIAGONAL, uncontrolled
```

so the **full** awMSE (the quantity inference actually pays, `eᵀ(XᵀX)e`) can rise even
when bias drops 50%+ and the diagonal stays flat. Shrinking the budget does not fix this —
it is not over-flipping, it is flipping into off-diagonal damage the diagonal score cannot see.

`full-greedy` replaces the diagonal gate with the **exact** per-flip change of the full
awMSE against the layer Hessian `H`:

```
flip of column i:  e_i → e_i + d_i ,  d_i = target_val_i − Wq_i = −sign(e_i) g_i
Δ_full = 2 d_i (H e)_i + d_i² H_ii
```

It keeps NCC's bias-progress **ordering** (so each accepted flip still pulls bias down the
most per move) but only **accepts** a flip when (1) it is sign-aligned, (2) it does not
overshoot the bias, and (3) `Δ_full ≤ 0`. After each accepted flip it does a rank-1 update
`He += d_i · H[:, i]`, which is why it is exact and why it is O(flips · in) per row.
`full-screen` skips the rank-1 update (uses pre-flip `He`) → cheaper, approximate.

> **Caveat for the writeup:** `full-greedy`/`full-screen` are **not** NCC-as-published.
> They answer *"do bias-reducing, awMSE-non-increasing complementary flips exist, and how
> much bias can they remove?"* — not *"does the published separable diagonal rule work?"*.
> Keep that distinction explicit.

---

## Running it

```bash
# faithful published NCC (diagonal), with the Corollary-2 diagonal guard
python debug_ncc_mse_lnq.py \
    --model-path /path/to/Meta-Llama-3.1-8B \
    --device cuda:0 --max-layers 2 --n-calib 16 --max-length 512 \
    --wbits 3 --lnq-iters 3 --cd-cycles 4 --guided \
    --ncc-budget-p 0.02 --score cov --mse-guard --mse-guard-mode diag

# cheap full-Σ screen — run this FIRST to see if the exact greedy is even needed
python debug_ncc_mse_lnq.py ... --score cov --ncc-budget-p 0.005 --mse-guard-mode full-screen

# exact full-Σ greedy — guarantees full awMSE never increases (slow)
python debug_ncc_mse_lnq.py ... --score cov --ncc-budget-p 0.005 --mse-guard-mode full-greedy
```

Prerequisites on the path: a GuidedQuant checkout (for `train_least_squares`, loaded by
file path so its heavy `any_precision/__init__` chain is **not** triggered) and an NCCQuant
checkout (`git clone https://github.com/anhnda/NCCQuant.git NCCQuant`). LNQ uses CUDA
internally (`train_least_squares` / `update_C` / `update_P`), so a CUDA device is required.

### Key flags

- `--wbits` — LNQ codebook bit-width (K = 2^wbits levels per output channel).
- `--lnq-iters` / `--cd-cycles` — LNQ alternating-minimization iters T and CD cycles K.
- `--guided` — use the GuidedQuant saliency Hessian instead of plain `Xᵀ X`.
- `--score {lite, cov}` — NCC ordering: `lite = |mu|/g`, `cov = |mu|/((Σ_ii+ε)g)`.
- `--ncc-budget-p` — budget fraction p (B_j = ⌈p|A_j|⌉ per channel).
- `--mse-guard` — Corollary-2 diagonal gate (only meaningful in `diag` mode).
- `--mse-guard-mode {diag, full-screen, full-greedy}` — corrector selection (above).

---

## Reading the output

Per-layer line:

```
layers.0.self_attn.k_proj [full-greedy] flips=2878 | blocked(awmse=7896,overshoot=124) |
  bias 1.10e-03->1.02e-03 (-7.71%) | MSE ... | awMSE[base] ...(-0.01%) |
  awMSE[orig] ...(-0.01%) | diag ...(-0.00%) | lnq=6.8s
```

- **bias** — `B_hat = Σ_j (muᵀ e_j)²`, the quantity NCC targets. Must not increase (Thm 3).
- **MSE** — plain weight MSE `mean (Wq − Wfp)²`.
- **awMSE[base] / awMSE[orig]** — **full** activation-weighted MSE `mean (xᵀe)²` vs the LNQ
  baseline and vs the original fp weights. This is the real inference error. In `full-greedy`
  it is asserted non-increasing.
- **diag** — **diagonal-only** awMSE `Σ Σ_ii e_i²`, the exact quantity Corollary 2 / `--mse-guard`
  controls. The gap between `diag` and full awMSE is precisely the off-diagonal cross-term.
- **blocked(awmse=…, overshoot=…)** — (full-Σ modes only) how many bias-reducing candidates
  were rejected by the full-awMSE gate vs the no-overshoot bias gate. A large `awmse=` means
  most bias-good flips are antagonistic to full awMSE.

Summary block (NET, summed over checked layers): total flips, total bias change, total full
awMSE[base]/[orig], total diagonal awMSE, plus counts of layers where each metric increased.

### Asserted invariants (fail loudly if violated)

- bias never increases (Theorem 3 / no-overshoot), all modes;
- with `--mse-guard`, diagonal awMSE never increases (Corollary 2) — if this fails the guard
  logic in `ncc.py` is wrong;
- in `full-greedy`, full awMSE never increases — if this fails the bug is in the rank-1 `He`
  update or the Δ sign in `apply_ncc_full_sigma`;
- NCC-internal `bias_before/after` match the externally measured `B_hat` on the same baseline.

---

## How to interpret the measured result

The harness turns "does NCC help?" into a quantified statement of the bias–MSE trade-off:

- If **diag** mode shows full awMSE rising while **diag** stays flat → the regression lives
  entirely in the off-diagonal cross-term, which the published diagonal certificate cannot
  control. Not a bug.
- If **full-greedy** holds full awMSE flat (≤0%) while **bias** still drops a meaningful
  amount → bias-reducing, MSE-safe flips exist; report the achievable bias reduction under
  the hard "no MSE increase" constraint.
- If **full-greedy** drives `blocked(awmse=…)` very high and bias reduction collapses → on
  this layer almost every bias-reducing flip is antagonistic to full awMSE; bias and MSE are
  in tension and NCC cannot win both. This is a clean negative result, consistent with the
  Mistral finding (bias down, downstream/MSE not helped).

Example from Llama-3.1-8B, NF3-scale LNQ+GuidedQuant, `full-greedy`, p = 0.005:
full awMSE held flat (−0.01% to −0.02%) while bias fell ~8%, with the majority of
bias-reducing candidates (`awmse≈7896` vs `flips≈2878`) blocked for hurting full awMSE —
i.e. a small, MSE-safe slice of the bias is recoverable; the rest is antagonistic.

---

## Files

- `debug_ncc_mse_lnq.py` — the harness (LNQ bridge, full-Σ corrector, metrics, asserts).
- Reuses, unmodified: GuidedQuant `train_least_squares`; NCCQuant `apply_ncc` + `QuantResult`.