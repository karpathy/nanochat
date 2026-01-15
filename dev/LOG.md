# Experiment Log

A running summary documenting some experiments and findings. Started ~Jan 7 2026.

---

## 2026-01-13: Varlen Attention (Negative Result)

Attempted to prevent attention from "leaking" across document boundaries using Flash Attention's `flash_attn_varlen_func`, similar to modded-nanogpt's approach.

### Background

With the BOS-aligned dataloader, multiple documents are packed into each row. Standard attention allows tokens to attend across document boundaries within a row. The hypothesis was that preventing this "leakage" via varlen attention might improve training.

### Approach: Compute cu_seqlens from inputs

- Find BOS positions: `(inputs.view(-1) == bos_token_id).nonzero()`
- Gotcha 1: Variable-length `cu_seqlens` caused torch.compile recompilation (25s/iter!) - fixed by padding to fixed size
- Gotcha 2: `nonzero()` inside compiled model hit recompile limit - fixed by moving computation outside compiled region

### Final Results (d16)

| Metric | Baseline | Varlen |
|--------|----------|--------|
| val_bpb | 0.85427 | 0.85407 |
| MFU | ~same | ~same |
| tok/sec | ~same | ~same |

Essentially identical. The 0.0002 bpb improvement is almost noise.

### Conclusion

Not worth the code complexity. The "leakage" across document boundaries within a row is not harmful - the model handles it fine. The BOS-aligned dataloader already provides the key benefit (every row starts with proper context). Not merging to master.

---

## 2026-01-13: BOS-Aligned Dataloader with Bin Packing

Redesigned the pretraining and midtraining dataloader to ensure every sequence starts with a BOS token, and explored bin-packing algorithms to minimize wasted tokens.

### Problem Statement

The original dataloader streams tokens into a flat buffer and reshapes into batches. This means some rows start mid-document (no BOS), which could confuse the model during training. We want every row to start with BOS and contain well-formed documents.

### Approach 1: Greedy-Crop BOS (Simple)

Each row is built independently:
- Start with a document (which has BOS prepended)
- Pack more documents until row is full
- If a document doesn't fit, **crop it** to fill remaining space (discard the rest)
- 100% utilization (no padding), but wastes cropped tokens

### Waste Analysis

Measured token waste empirically on real data (T=2048):
- **39.4% of tokens are cropped** (discarded when docs don't fit)
- **22.9% is the theoretical minimum** (tokens in docs longer than T+1 that can never fit)
- The extra ~16.5% comes from "unlucky" cropping when a long doc starts near the end of a row

### Bin Packing Algorithms Explored

| Algorithm | Util% | Crop% | Pad% | Notes |
|-----------|-------|-------|------|-------|
| Greedy-Crop (baseline) | 100% | 39.4% | 0% | Simple, no wasted compute |
| Greedy-Pad | 78% | 23.0% | 22% | Pads instead of crops - wastes compute |
| First-Fit Decreasing (FFD) | 99.7% | 23.0% | 0.3% | Near-optimal packing, minimal padding |
| **BestFit-Crop** | 100% | 34.6% | 0% | Smart cropping, no padding |

### BestFit-Crop Algorithm

A middle ground that maintains 100% utilization while reducing cropping:

1. Buffer N documents
2. For each row, greedily pick the **largest doc that fits entirely**
3. Repeat until nothing fits
4. When nothing fits, crop a doc to fill remaining space exactly

This avoids "unlucky" crops by searching the buffer for better-fitting documents.

**Results (T=2048):**
- Crop waste reduced from 39.4% → 34.6% (~12% relative improvement)
- Still achieves 100% utilization (no padding, every token trains)
- Slightly more rows than baseline (uses more documents per batch)

### Decision: Keep Two Implementations

1. Keep the original implementation which is very simple, efficient and has 100% token utilization in the batch (no padding with ignore tokens), but creates slightly more confusing token streams for the LLM because documents during training can start abruptly from the middle with no context. Note that this never happens at test time, where BOS is always present.

2. **`_bos_bestfit` (BestFit-Crop, new default)**: Slightly more complex but still keeps 100% token utilization in the batch (no padding), but at the cost of discarding documents when they don't fit. In practice, about 34% of tokens are discarded with this approach. This is ok because for most models we care about we have plenty of data without having to go to multiple epochs. One more subtle effect is that it does skew the data distribution a tiny bit because, reliably and necessarily, tokens at the tails of long documents will be discarded. However, this doesn't seem to impact actual downstream performance.

### Midtraining

The midtraining dataloader was also updated. Because conversations are on average a lot shorter than pretraining documents, only about 3.3% of tokens get cropped.

### NOTE: loss scale

Do note that switching to the BOS dataloader changes the validation loss and makes all previous experiments not comparable in absolute value of the loss, because we have a lot fewer "confusing" tokens in the train/val batches. All tokens can look back and find the BOS token and have the full context of that document to make predictions. Therefore, the loss appears lower but this is "fake" to some extent, and the expectation is that the vast majority of relative comparisons done so far would agree with those before and after this change.

---

## 2026-01-13: Number Token Split Pattern

Validated the `\p{N}{1,2}` pattern in `SPLIT_PATTERN` (tokenizer.py line 30), which I only guessed earlier and had a TODO for to validate. GPT-4 uses `\p{N}{1,3}` to group number sequences of up to 3 digits into tokens, but we suspected smaller vocab sizes benefit from grouping fewer digits per token.

**Results (d12, vocab=32K):**
| Pattern | val_bpb |
|---------|---------|
| `\p{N}{1,1}` | 0.969 |
| `\p{N}{1,2}` | **0.965** |
| `\p{N}{1,3}` | 0.972 |

**Conclusion:** `{1,2}` is optimal for vocab size 32K. Grouping 3 digits wastes tokens on rare 3-digit combinations; grouping 1 digit is too fine-grained and bloats token sequences. Keeping `{1,2}` as default.

---

## 2026-01-13: FP8 Training for lm_head

Attempted to use FP8 (8-bit floating point) for the lm_head layer to speed up the large vocab projection matmul. H100 GPUs have FP8 tensor cores that can theoretically provide ~2x speedup over BF16.

### Implementation Approaches Tried

**1. Dynamic Scaling (failed)**
- Compute `x.abs().max()` and `w.abs().max()` each forward to determine scales
- Problem: `.item()` calls cause graph breaks with torch.compile
- Tried `@torch._dynamo.allow_in_graph` pattern (like torchao.float8) - worked but no speedup
- Tried `torch.library.custom_op` with float scales - caused NaN gradients after first optimizer step
- Root cause: interaction between custom ops, dynamic scale computation, and torch.compile is fragile

**2. Static Scaling (partial success)**
- Pre-set scales at init time like modded-nanogpt: `x_scale=10/448, w_scale=0.1/448`
- `grad_scale` computed dynamically from batch size (safe since it's just `1/(B*T)/57344` due to the gradient expression of cross entropy). modded-nanogpt has a bug here probably because they set `grad_scale = 0.75/448`, but grads are in E5M2 so this should probably be `1/57344`, 1 being the amax of any individual element of cross entropy loss, and no normalization by B,T because they use sum reduction not mean reduction.
- Uses `torch.library.custom_op` with `@torch.compile` on inner kernels
- This works correctly - no NaNs, proper gradients

### Results (d12)

| Metric | BF16 Baseline | FP8 lm_head |
|--------|---------------|-------------|
| GPU Memory | 34 GB | 36 GB |
| tok/sec | baseline | ~1% faster |

### The Memory Mystery

FP8 *should* save memory since we store `x_f8` (1 byte) instead of `x` (2 bytes) for backward. But we see 2GB *increase*. Suspected causes:
- `torch.compile` on inner kernels creating extra buffers/specializations
- `torch._scaled_mm` internal workspace allocations
- Custom op registration machinery overhead

Tried saving original weight `w` (just a reference to parameter) instead of `w_f8` in backward, then re-quantizing on the spot during backward - didn't help. Still saw bump.

### Microbenchmark vs Reality

Raw microbenchmark showed promise:
- BF16 matmul: 16.95 ms
- FP8 matmul (static scales): 10.31 ms (1.64x faster)
- FP8 with dynamic scaling: 12.25 ms (1.38x faster)

But in full training, the ~1% tok/sec improvement doesn't justify the 2GB memory increase and the added code complexity and the need to tune scale factors for both x and w.

### Code Artifacts

See the branch `fp8_attempt_fail` for:

- `nanochat/fp8_static.py` - Static scaling implementation (working)
- `nanochat/fp8_dynamic.py` - Dynamic scaling implementation (torchao-style, working but slow)
- `gpt.py` imports `fp8_static.LinearFP8` and simply swaps it for `lm_head` in `gpt.py`.

### Open Questions

- Why does the custom op approach use more memory than vanilla BF16?
- Why is the bump in tok_per_sec so low? We should see ~1.6X speedup in both the forward pass and also (twice) in backward pass for the gradients. Granted, Ahmdal's law is part of the solution because our vocab_size is only 32K so the final layer isn't a huge part of the profile but the expected speedup is still not fully realized.

**Conclusion:** Negative result for now. The implementation works correctly but provides marginal speedup with *increased* memory usage. I'm not understanding the torch.compile interaction here. The complexity of FP8 custom ops isn't justified for lm_head alone. TODO to study in more detail the way this is implemented in other libraries, e.g. torchao.

---

## 2026-01-12: Multi-Token Prediction (MTP)

Ported multi-token prediction from modded-nanogpt. Instead of predicting just the next token, predict the next n tokens at each position with weighted loss.

### Implementation

- Instead of calling the loss `n_predict` times, uses a fancy batched computation using `unfold` + `gather` + cross-entropy decomposition (`CE = logsumexp - logits[target]`)
- Schedule anneals from 3-token to 1-token prediction:
  - 0-33%: `[1.0, 0.5, 0.25→0]` (3rd token fades)
  - 33-67%: `[1.0, 0.5→0]` (2nd token fades)
  - 67-100%: `[1.0]` (standard next-token)
- Weights normalized to sum to 1

### Results (d12)

| Metric | Baseline | MTP |
|--------|----------|-----|
| GPU Memory | 34 GB | 47 GB |
| MFU | 41% | 40% |
| val/bpb (per step) | baseline | same/slightly worse |
| val/bpb (wall clock) | baseline | noticeably worse |

**Conclusion:** Negative result for nanochat. The extra memory and compute overhead from predicting multiple tokens doesn't pay off, in fact the results get worse. The auxiliary loss signal may help in other settings (larger models, different architectures?), but for our setup it's pure overhead at the moment.

---

## 2026-01-11: Sliding Window Attention

Added configurable sliding window attention, inspired by GPT-3's alternating short/long pattern.

**Pattern string configuration:**
- New `--window_pattern` CLI arg and `GPTConfig.window_pattern` field
- Pattern is tiled across layers (e.g., `SSSL` for 20 layers → `SSSLSSSLSSSLSSSLSSSL`)
- Final layer always forced to L (full context) regardless of pattern
- Short window = `sequence_len // 2`
- Long window = `sequence_len` (full context)
- All previous models so far have been simply `L` and checkpoint loading is modified accordingly to fill in this param for old models, see `_patch_missing_config_keys`

Quick experiments showed `SSSL` (every 4th layer is long) works well - provides a good balance between compute savings and model quality. This is now the default.

---

## 2026-01-11: Flash Attention 3 Integration

Replaced PyTorch's `scaled_dot_product_attention` (FA2) with Flash Attention 3 for training and inference.

### Changes Made

**1. FA3 via `kernels` package**
- Official FA3 is "beta" and requires building from source (painful)
- Using `kernels` package from HuggingFace Hub: `get_kernel('varunneal/flash-attention-3')`
- Loads pre-built wheels, works out of the box on H100

**2. Simplified attention code**
- FA3 uses `(B, T, H, D)` layout matching our projection output directly - no transpose needed
- Training: `flash_attn.flash_attn_func(q, k, v, causal=True)`
- Inference: `flash_attn.flash_attn_with_kvcache()` handles all cache cases in one call
- Removed 3 separate FA2 code paths (training, single-token, chunk inference)
- GQA handled automatically when n_kv_heads < n_heads

**3. Rewrote KVCache for FA3**
- Old format: `(num_layers, 2, B, H, T, D)` combined tensor
- New format: separate `k_cache` and `v_cache` of shape `(num_layers, B, T, H, D)`
- FA3 updates cache in-place during `flash_attn_with_kvcache`
- Position tracked via `cache_seqlens` tensor (int32, per batch element)
- Simpler API: `get_layer_cache()`, `advance()`, `reset()`, `prefill()`

### Results

- **~9% improvement in tok/sec** during training out of the box
- Benchmarks showed FA3 is 2x faster than FA2 at realistic training sizes (batch=32, seq=2048)
- FA3 supports sliding window via `window_size=(left, 0)`, which is huge and expected to give further improvements. This is ready to tune but keeping full context for now.

---

## 2026-01-11: Per-Layer Residual Scalars (x0 & resid lambdas)

Cherry-picked an idea from modded-nanogpt around learnable per-layer residual connections.

### Changes Made

**1. x0_lambdas (x0 residual connections)**
- Save initial normalized embedding as `x0` after `norm(wte(idx))`
- At each layer, blend x0 back in: `x = resid_lambdas[i] * x + x0_lambdas[i] * x0`
- Zero-initialized, so disabled at start; model learns which layers benefit from the shortcut
- Provides direct path from embedding to deep layers, helps preserve token information

**2. resid_lambdas (residual stream scaling)**
- Per-layer multiplicative scaling of the residual stream
- Initialized to 1.0 (neutral, standard transformer behavior)
- Allows model to learn to amplify/dampen residual at each layer

**3. DistAdamW small parameter handling**
- Added support for parameters with < 1024 elements (like the scalar lambdas)
- Small params use `all_reduce` instead of `reduce_scatter`/`all_gather`
- Fixes crash when param shape isn't divisible by world_size

### Key Finding: Different LR Sensitivity

The two scalar types need very different learning rates:
- **x0_lambdas (additive)**: Can use normal LR (~0.5). Adding a fraction of x0 is forgiving.
- **resid_lambdas (multiplicative)**: Needs ~100x smaller LR (~0.005). Multiplying the residual compounds through layers.

Implementation: `resid_params` gets `scalar_lr * 0.01`, `x0_params` gets full `scalar_lr`.

### Experiment Results

Swept `--scalar_lr` (controlling x0_lambdas) at multiple depths:

| Depth | Baseline (disabled) | Best scalar_lr | Best val_bpb | Δ bpb |
|-------|---------------------|----------------|--------------|-------|
| d8    | 1.0885              | 0.20           | 1.0782       | -0.0103 |
| d12   | 0.9770              | 0.60           | 0.9693       | -0.0077 |
| d16   | 0.9059              | 0.20           | 0.9002       | -0.0057 |
| d20   | 0.8565              | 0.10           | 0.8526       | -0.0039 |

**Observations:**
- Consistent improvement across all model sizes
- Optimal LR varies by depth; default of 0.5 is reasonable, but 0.6 is better for d12
- Adding resid_lambdas (with 0.01x LR) gives small additional improvement over x0 alone

### Meta Device Footgun

Important lesson: `__init__` runs in meta device context, so any tensor values set there are fake. Must initialize actual values in `init_weights()`. Added docstring warning to `__init__`.

### Summary

Added `--scalar_lr` (default 0.5) controlling learnable per-layer scalars. The formula `x = resid_lambdas[i] * x + x0_lambdas[i] * x0` gives the model control over residual scaling and direct shortcuts to the initial embedding. Solid improvement with essentially no compute overhead.

---

## 2026-01-10: Muon Optimizer Upgrades & Cautious Weight Decay

Cherry-picked improvements from NorMuon (modded-nanogpt) into our simpler Muon implementation. Decided against using NorMuon directly due to hard-coded architecture assumptions (expects 32 params split 10 attn + 22 mlp), parameter labeling requirements, and complexity.

### Changes Made

**1. Polar Express Orthogonalization**
- Replaced Newton-Schulz iteration with "Polar Express Sign Method" from [arxiv.org/pdf/2505.16932](https://arxiv.org/pdf/2505.16932)
- Uses 5 different coefficient tuples (one per iteration) instead of fixed coefficients
- Both methods kept in code for easy comparison (`zeropower_via_polar_express` vs `zeropower_via_newtonschulz5`)
- **Result:** No dramatic/noticeable difference in training, but keeping the new Polar Express as default.

**2. Variance Reduction (NorMuon-style)**
- Added low-rank variance estimator similar to Adafactor ([arxiv.org/pdf/2510.05491](https://arxiv.org/pdf/2510.05491))
- Maintains `second_momentum_buffer` with shape `[rows, 1]` or `[1, cols]` (whichever is smaller)
- Normalizes updates based on running per-row/col variance estimate (beta2=0.95)
- Memory overhead: ~1/max(rows, cols) per param, negligible
- **Result:** Led to a very small improvement, kept and enabled by default.

**3. Cautious Weight Decay**
- Only decays weights where `update * weight >= 0` (same sign) from [arxiv.org/abs/2411.16085](https://arxiv.org/abs/2411.16085)
- Standard WD always pulls toward zero; cautious WD skips decay when gradient is pushing weight away from zero
- **Implementation note:** Had to inline the logic rather than use a separate `@torch.compile` function. Passing changing float values (like `weight_decay` during scheduling) as function arguments triggers recompilation. Reading from `group["weight_decay"]` inside the step avoids this.
- **Result:** Solid improvements, especially the cautious version was better than standard wd.
- Now defaults to ON for Muon via the `weight_decay` param. AdamW still has no weight decay and is hardcoded to 0 weight decay, might try to re-tune this later.

**4. Weight decay schedule**
- Added a linear schedule to weight decay that is default on from 1.0 to 0.0 (i.e. start with max weight decay in the beginning of training, them ramp to 0 by the end). Worked better than a static setting in experiments. (modded-nanogpt has the same schedule but it is imlpemented in a more confusing way by multiplying twice by the learning rate, which is already wired up to a decay schedule).

### Weight Decay Scaling Experiments

Swept weight decay values at d8, d12, d16, d20 to find optimal values and scaling law.

**Optimal Values Found:**
| Depth | Width (channels) | Optimal WD |
|-------|------------------|------------|
| d8    | 512              | ~0.40      |
| d12   | 768              | ~0.22      |
| d16   | 1024             | ~0.10      |
| d20   | 1280             | ~0.08      |

**Scaling Law:**
- Fit power law: `WD = k / channels^α` in log-log space
- Found α ≈ 1.97 (approximately 2), meaning WD ∝ 1/width²

**Practical Formula:**
```
WD_target = WD_reference × (d_reference / d_target)²
```
Example: If d12 optimal is 0.22, then d20 optimal ≈ 0.22 × (12/20)² ≈ 0.08

**Reference:** Moonlight paper uses fixed WD=0.1 for their 15B MoE model. Our experiments indicated a scaling law where the optimal WD changed with depth, so we go along with the empirical scaling law.

### Summary

Muon was changed to use Polar Express, added Adafactor-style variance reduction, and cautious weight decay with schedule that ramps linearly to zero. All of these changes follow modded-nanogpt repo, but all of them were also validated piece by piece to yield improvements in nanochat with the exception of the Polar Express change which was in the noise. This is default on and configurable with `--weight_decay`, using simply 0.2 and ∝ 1/width² scaling. The kwarg `--weight_decay` is therefore changing as of this change. It used to configure AdamW via standard weight decay and now it becomes exclusively used in Muon (AdamW is hardcoded to 0.0), and it is scaled based on depth.

---

## 2026-01-08: exp_grad_clip - Gradient Clipping

**Hypothesis:** Gradient clipping may be unnecessary overhead. Tested L2 norm clipping at various thresholds (0.25, 0.5, 1.0, 2.0) and elementwise clipping.

**Results:**
- No benefit at any scale tested (d12, d20)
- All variants within noise (~0.9827 val_bpb)
- Grad norm never exceeds 1.0 naturally, so clipping is always inactive
- Clipping adds ~2% time overhead from the all-reduce

**Bug Found:** Original implementation clipped local gradients before sync. Since this codebase doesn't use DDP (gradient sync is in the optimizers), each rank was clipping based on its own local norm. Fixed on the branch with proper distributed all-reduce.

**Observartion:** modded-nanogpt does not appear to clip either right now.

**Summary:** Deleted all grad-clip code paths. The code naturally produces well-behaved gradients. This improves a bit of MFU because we don't have to calculate and sync grad norms.
