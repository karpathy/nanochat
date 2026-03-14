# muP Adaptation for Muon+AdamW in nanochat

## Context

Standard muP (Yang et al., "Tensor Programs V", arXiv:2203.03466) was derived for SGD and Adam optimizers. nanochat uses a **mixed optimizer**: Muon (with Polar Express orthogonalization) for transformer hidden weights, and AdamW for embeddings, scalars, and the output head (lm_head). This document describes the adaptations required and the empirical evidence behind them.

## The One Essential muP Ingredient

The output logit scaling in the forward pass is the core of muP and remains unchanged:

```python
# gpt.py forward()
logits = self.lm_head(x)
if self.config.mup_base_width > 0:
    logits = logits * (self.config.mup_base_width / self.config.n_embd)
```

Without this, logit magnitudes grow as O(√width) because the lm_head dot product sums over `n_embd` terms. The multiplier `base_width / n_embd` (= `1/m_d` where `m_d = width/base`) keeps logits O(1) at all widths. This is what enables hyperparameter transfer — it's the mechanism that makes the loss landscape shape-invariant across widths.

## What We Changed (and Why)

### Change 1: Removed output LR scaling

**Before:**
```python
output_lr_scale = base_width / model_dim  # e.g., 128/1024 = 0.125
```

**After:**
```python
output_lr_scale = 1.0  # No width-dependent LR scaling for lm_head
```

#### Why the paper's prescription doesn't apply here

The paper (Table 8, "MUP" row) prescribes output layer LR ∝ `1/width`. The reasoning: in vanilla SGD, the lm_head gradient magnitude scales with width, so the LR must compensate. For Adam, the second moment normalizes gradients, but the paper still prescribes `1/width` because the *signal-to-noise ratio* of the Adam update changes with width.

However, this analysis doesn't account for the output logit scaling. Here's the interaction:

1. **Forward pass**: `logits = (base/width) × h @ W_out^T`
2. **Backward pass**: `∂L/∂W_out = (base/width) × (∂L/∂logits)^T @ h`
   — the gradient already carries a `base/width` factor from the chain rule through the output multiplier
3. **Adam step**: Adam normalizes by `√(E[grad²])`, which is O(base/width). The normalized step is O(1).
4. **LR application**: If LR is also scaled by `base/width`, the effective update becomes O(base/width).
5. **Effect on logits**: `Δlogits = (base/width) × h @ ΔW^T`, contributing another `base/width` factor.

**Net effect**: The logit change per step scales as O((base/width)²) — quadratic suppression. At width=1024 with base=128, this is a **64× reduction** in the effective output learning rate. The lm_head is barely learning.

#### Empirical evidence

Using `--sweep-mode adamw-only` (sweep only AdamW LR, hold Muon fixed):

| Width | Old muP optimal mult | Fixed muP optimal mult |
|-------|---------------------|----------------------|
| 128   | 32                  | 32                   |
| 256   | 64                  | 32                   |
| 512   | 128                 | 32                   |
| 1024  | 256                 | 32                   |

**Old muP**: Optimal multiplier doubles with each width doubling (spread = 3.0 log2). The sweep is perfectly compensating for the over-reduction — the optimizer needs `m_d` times more LR to undo the `1/m_d` scaling.

**Fixed muP**: Optimal multiplier = 32 at all widths (spread = 0.0 log2). Perfect transfer.

### Change 2: Set Muon LR exponent to 0

**Before:**
```python
hidden_lr_scale = base_width / model_dim  # 1/m_d scaling for Muon hidden weights
```

**After:**
```python
hidden_lr_scale = (base_width / model_dim) ** muon_lr_exponent  # default exponent = 0.0 → scale = 1.0
```

#### Why standard muP LR scaling is redundant for Muon

The paper prescribes hidden layer LR ∝ `1/fan_in` = `base/width` for Adam. This compensates for Adam updates scaling with fan_in: with n_embd input dimensions, each element of the update is O(1/√n_embd) after Adam normalization, but the net change to the residual stream (summing over n_embd) is O(√n_embd). The `1/width` LR tames this.

**Muon doesn't have this problem.** Muon's Polar Express orthogonalization produces an update with `||update||_F ≈ 1` regardless of matrix dimensions. The update's Frobenius norm is O(1), and its contribution to the residual stream is also O(1) — it doesn't grow with width. Applying an additional `1/width` factor makes the update O(1/width), which *vanishes* at large width.

#### Empirical evidence

We tested three exponents with `--sweep-mode all`:

| muon_lr_exponent | muP optimal LR spread (log2) |
|-----------------|------------------------------|
| 0.0             | 2.0                          |
| 0.5             | 3.0                          |
| 1.0             | 2.0                          |

Exponents 0.0 and 1.0 give **identical spread** (2.0). The Muon LR exponent literally doesn't matter — Polar Express dominates the update magnitude regardless of LR scaling. We default to 0.0 (no scaling) as the simplest correct choice.

(The spread of 2.0 in these experiments was caused by the output LR scaling bug, which was still active. After fixing Change 1, the overall spread dropped to 1.0 for all exponents.)

## What Remains Unchanged

| Component | Value | Paper requirement | Status |
|-----------|-------|-------------------|--------|
| Output logit scaling | `logits *= base/width` | Required | ✅ Correct |
| Embedding LR | No width scaling | Constant with width | ✅ Correct |
| lm_head init std | `0.001 × √(base/width)` | Width-scaled init | ✅ Correct |
| Weight decay | Not width-scaled | Constant with width | ✅ Correct |
| Momentum (Adam β₁, Muon) | Not width-scaled | Constant with width | ✅ Correct |
| c_proj init | Non-zero uniform, std=√(3/n_embd) | Paper recommends zero | ⚠️ Intentional divergence |

**On c_proj init**: The paper recommends zero-initializing output projections (attn c_proj, MLP c_proj) for cleaner transfer. nanochat uses non-zero init because zero init causes vanishing attention/FFN outputs when combined with Muon's LR dynamics — the first Muon update from a zero matrix produces an orthogonal matrix with O(LR) norm, which is too small when LR is already small. This is a known interaction between Muon and residual-stream architectures; the non-zero init provides a stable starting point.

## Summary: muP for Muon+AdamW

For a mixed Muon+AdamW optimizer, muP simplifies dramatically:

| Parameter group | muP prescription | Reason |
|----------------|-----------------|--------|
| **Output logits** | `logits *= base/width` in forward | The essential ingredient — makes loss landscape shape-invariant |
| **lm_head init** | `std *= √(base/width)` | Keeps initial logit magnitudes O(1) |
| **lm_head LR** | No width scaling | Logit scaling already propagates into gradient; Adam normalizes; additional LR scaling over-reduces |
| **Muon (hidden) LR** | No width scaling | Polar Express makes `||update||_F ≈ 1` regardless of width |
| **Embedding LR** | No width scaling | Standard muP (embeddings are lookup tables, not matrix multiplies) |
| **Scalar LR** | No width scaling | Standard muP |

**The punchline**: With Muon+AdamW, muP reduces to scaling output logits by `base/width` in the forward pass (plus corresponding init adjustment). No LR scaling is needed anywhere — Muon's orthogonalization and Adam's second-moment normalization both already produce width-independent updates.

## Verification

```bash
# Full transfer check (should show muP spread ≤ 1.0, SP spread ≥ 2.0)
python -m scripts.mup_transfer_check --compare --widths 128,256,512,1024,2048 \
    --steps 50 --num-batches 200 --save-dir temp/mup_transfer

# Coordinate check (activation magnitudes should be flat across widths for muP)
python -m scripts.mup_coord_check --compare --steps 10 --detailed --save-dir temp/mup_coord

# Automated tests
python -m pytest tests/test_mup.py -v
```

## Files Changed

| File | Changes |
|------|---------|
| `nanochat/gpt.py` | `output_lr_scale`: `base/width` → `1.0`; added `muon_lr_exponent` param (default `0.0`); updated comments |
| `scripts/mup_coord_check.py` | Added `--detailed` flag (grad norms, update norms, attn logit magnitudes), `--muon-lr-exponent` |
| `scripts/mup_transfer_check.py` | Wider default LR range (1024×), `--sweep-mode {all,muon-only,adamw-only}`, `--num-random-trials`, `--num-batches`, `--sweep-init-scale`, `--sweep-output-mult`, `--muon-lr-exponent`, default steps 100→200 |

## References

- Yang et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer", arXiv:2203.03466 (2022). Sections B.1 (muP table), C.1 (Frobenius-normalizing optimizers), F (GPT-3 experiments).
- EleutherAI muP blog: https://blog.eleuther.ai/mutransfer/
- Polar Express: Amsel et al., arXiv:2505.16932 (2025).
- Muon: https://kellerjordan.github.io/posts/muon/
