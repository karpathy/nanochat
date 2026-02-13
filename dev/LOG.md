# Experiment Log

A running summary documenting some experiments and findings. Started ~Jan 7 2026.

---

## 2026-02-05: Auto Batch Size Scaling

### Background

So far, the `--total-batch-size` was hardcoded to be `2**19 = 524,288` ~= 0.5M tokens. This was the optimal setting for d12, but when I tried to re-tune it for d26 (GPT-2), I noticed that the optimal was closer to `2**20 = 1,048,576` ~= 1M tokens. This is to be expected - larger models prefer a higher optimal total batch size. However, we have to make sure that all settings of `--depth` get their own optimal batch size calculated in some principled way. Here, I referenced the "Power Lines" paper from Cerebras ([arXiv:2505.13738](https://arxiv.org/abs/2505.13738)) for a lot of related experimentation. In particular, they found that **Bopt ∝ D^0.383** (where D is the number of training tokens, not the number of parameters!). So the idea is to tune the optimal batch size on d12, and then extrapolate it with this power law to bigger models. The 0.383 exponent means batch size grows slowly: 10× more tokens only justifies ~2.4× bigger batch. For nanochat's compute-optimal training (D ∝ N via `--target-param-data-ratio`), this means deeper models naturally want larger batches.

### Implementation

Added `--total-batch-size=-1` (now the default) to auto-compute optimal batch:

```python
get_scaling_params = lambda m: m.num_scaling_params()['transformer_matrices'] + m.num_scaling_params()['lm_head']
if args.total_batch_size == -1:
    D_REF = args.target_param_data_ratio * get_scaling_params(build_model_meta(12))
    B_REF = 2**19
    args.total_batch_size = 2 ** round(math.log2(B_REF * (target_tokens / D_REF) ** 0.383))
```

Reference point: d=12 model with B=2^19 (empirically validated). The reference is computed dynamically so that if the architecture changes (e.g., different `--aspect-ratio`), the math automatically adjusts. However, if the model actually does change too much, one would also want to re-tune the optimal batch size for d=12.

### Results

With this formula, we currently get:

| Depth | Scaling Params | Target Tokens | Auto Batch |
|-------|---------------|---------------|------------|
| d=8   | 42M           | 0.44B         | 2^18 = 262K |
| d=10-16 | 70M-235M    | 0.7B-2.5B     | 2^19 = 524K |
| d=18-26 | 324M-918M   | 3.4B-9.6B     | 2^20 = 1.05M |
| d=32-50 | 1.7B-6.2B   | 17.6B-65.6B   | 2^21 = 2.1M |

In particular, this matches empirical observations that d26 prefers ~2^20 while d12 prefers ~2^19.

### Code Cleanup

Also refactored model initialization to use `build_model_meta(depth)` helper and `dataclasses.asdict()` for cleaner config handling.

### Useful references

- [Bergsma et al., Power Laws for Batch Size, Model Size, and Training Horizon](https://arxiv.org/abs/2505.13738)
- [McCandlish et al., An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)
- [Brown et al., Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
- [Merrill et al., The Batch Size–Critical Batch Size Myth](https://arxiv.org/abs/2505.23971)

### One more thing (batch size ramp)

Tried batch size ramping. The simplest implementation I could think of "tricks" the existing training loop by slicing each micro-batch into smaller pieces and calling optimizer.step() more frequently early in training (1/8 → 1/4 → 1/2 → full batch over the first x% of training, with sqrt LR scaling). Also required a torch.compile warmup phase to pre-compile all slice sizes and avoid recompilation spikes during training. While the idea is sound and small gains were observed, they weren't sufficient to justify the code complexity introduced (conditional slicing logic, warmup with state save/restore, etc.). Not merged for now.

---

## 2026-02-05: SwiGLU Activation (Negative Result)

Replaced ReLU² MLP activation with SwiGLU (inspired by [twitter](https://x.com/_xjdr/status/2019141521690567058)). SwiGLU uses three projections instead of two, so to match parameters and FLOPs we scale hidden_dim from 4× to 8/3×:

```python
# Old ReLU²: 2 matrices, 4x expansion
#   params: 2 × n × 4n = 8n²
#   flops:  2 × 2n × 4n = 16n² per token
self.c_fc   = Linear(n_embd, 4 * n_embd)
self.c_proj = Linear(4 * n_embd, n_embd)
x = c_proj(relu(c_fc(x)).square())

# New SwiGLU: 3 matrices, 8/3x expansion
#   params: 2 × n × (8n/3) + (8n/3) × n = 8n²  ✓ matches
#   flops:  3 × 2n × (8n/3) = 16n² per token   ✓ matches
hidden_dim = (8 * n_embd) // 3
self.w1 = Linear(n_embd, hidden_dim)  # gate
self.w2 = Linear(n_embd, hidden_dim)  # up
self.w3 = Linear(hidden_dim, n_embd)  # down
x = w3(silu(w1(x)) * w2(x))
```

Tested at both d12 and d24 (GPT-2 scale). Worse on all measures — step efficiency, wall clock time, and FLOPs. ReLU² remains superior for nanochat. **Not adopted.**

---

## 2026-02-03: Flip Muon MLP LR Multiplier (PR #492)

Tested flipping the shape-based LR heuristic in Muon from boosting tall matrices (input projections like `c_fc`) to boosting wide matrices (output projections like `c_proj`). The original code applies `max(1, rows/cols)^0.5`, giving ~2x LR to `c_fc`. The flipped version gives ~2x LR to `c_proj` instead, which aligns with classical fan-in/fan-out scaling conventions. This was proposed in [PR #492](https://github.com/karpathy/nanochat/pull/492) and showed improvements in modded-nanogpt.

**Result:** Quick d12 experiment: slightly worse **Not adopted.**

---

## 2026-02-03: Skip AdamW Every Other Step

Inspired by modded-nanogpt, tried stepping AdamW only on odd iterations while Muon steps every iteration. The idea is that small AdamW params (embeddings, scalars, gates) don't need updates as frequently as the large weight matrices, and skipping saves both compute and communication.

Added `skip_adamw` parameter to `MuonAdamW.step()` and `DistMuonAdamW.step()` plus a matching `zero_grad(skip_adamw=...)` to let AdamW gradients accumulate over 2 steps. Used `lr *= 2**-0.5` (sqrt scaling) to compensate for the 2x effective batch size on AdamW params.

**Result:** for nanochat d12, we see ~2% faster tok/s, but each step is slightly worse in loss. On net, when plotting against wall clock time, it's slightly worse. **Not adopted.**

---

## 2026-02-02: FP8 Training with torchao

Integrated FP8 training using `torchao.float8` to accelerate Linear layer matmuls on H100 GPUs.

### Background

FP8 (8-bit floating point) uses H100's FP8 tensor cores for ~2x theoretical matmul throughput. The tradeoff is quantization overhead: computing scales and casting tensors to/from FP8. Still, as an example torchtitan (Meta's distributed training framework) reports 25-28% speedups with FP8 for some of their experiments.

**Previous attempt (Jan 2026):** FP8 on just `lm_head` following modded-nanogpt with custom ops → 1% speedup, +2GB memory. Failed due to fragile torch.compile interaction. But this experiment was also done on ~d12 scale back then instead of the bigger model that gets GPT-2 capability of approx d24.

**This attempt:** Use torchao's `convert_to_float8_training()` on ALL Linear layers, increase model size to d24. The core snippet is:

```python
from torchao.float8 import Float8LinearConfig, convert_to_float8_training
config = Float8LinearConfig.from_recipe_name("tensorwise")
convert_to_float8_training(model, config=config)
```

But in practice it's more involved (see base_train.py).

### Results

**Microbenchmark (d26 MLP, 65536x1664 @ 1664x6656):**

| Method | Forward | Fwd+Bwd | Speedup |
|--------|---------|---------|---------|
| BF16 + compile | 2.00ms | 4.79ms | 1.00x |
| FP8 rowwise + compile | 1.84ms | 4.55ms | 1.08x |
| FP8 tensorwise + compile | 1.45ms | 4.06ms | **1.38x** |
| FP8 rowwise (no compile) | 2.89ms | 21.86ms | 0.23x ❌ |

torch.compile is MANDATORY. Without it, FP8 is 4x slower due to unfused scaling ops.

**Full training (d26):**

| Config | tok/sec | vs baseline |
|--------|---------|-------------|
| BF16 baseline | 630K | 1.00x |
| FP8 rowwise | 564K | 0.90x ❌ |
| FP8 tensorwise | 740K | **1.17x** ✓ |

Memory usage also decreases quite a bit, by ~9GB (activations stored as FP8 instead of BF16).

Seeing 17% speedup is encouraging but we're still not done yet because each step is now in lower precision and less powerful individually, so to make up for the precision drop we have to train longer. Empirically, running some sweeps overnight on d24 scale, I saw that the actual speedup (when you match performance) is closer to 5%. It's possible that our LLMs at ~d24 scale are still too small to confidently enjoy the speedups that come from fp8 for bigger models.

### Key Learnings

For nanochat at approximate scale of interest (~GPT-2 capability, ~d24):

1. **Tensorwise >> Rowwise** - Rowwise computes per-row scales, overhead exceeds benefit. Tensorwise uses one scale per tensor.
2. **Filter small layers** - Layers with dims not divisible by 16 must be skipped (FP8 hardware requirement)
3. **Larger models benefit more** - d12 was still slower with FP8; d26+ shows gains. Therefore, in some depths there is a benefit to fp8 and in some there isn't. Keeping it configurable for now, passed in via kwargs and default off.
4. **The effective, capability-matched speedup is lower still** - because each step is of slightly lower precision/quality.

### Integration

Added `--fp8` flag to `base_train.py`, default recipe is "tensorwise", example of turning on:

```bash
torchrun --nproc_per_node=8 -m scripts.base_train --depth=24 --fp8
```

Uses tensorwise by default. Requires `torchao==0.15.0` (compatible with torch 2.9.1), which was added to dependencies.

**TLDR**: turning on fp8 for GPT-2 capability nanochat model gives approx +5% capability-matched speedup.

---

## 2026-01-29: Hyperball/MuonH Experiments (Negative Result)

Explored Hyperball optimization from [this post](https://psychedelic-sunstone-851.notion.site/Fantastic-Pretraining-Optimizers-and-Where-to-Find-Them-2-1-Hyperball-Optimization-2e924306e6f280e7a5ffee00eb40a0dd) (saved to `knowledge/muonh.md`). Constrains weights to sphere of radius R (initial norm): `W_{t+1} = R · Normalize(W_t - η·R · Normalize(u_t))`. Had to change a number of details in a branch, e.g. not use zero init for our projections (or the initial norm would be zero), keep track of the initial norm, adjust Muon -> MuonH for the update.

Experiments on d12:

| Experiment | Result |
|------------|--------|
| MuonH for matrix params | Worse than baseline |
| MuonH + LR sweep (2.5e-3 to 1e-2) | Still worse |
| Added learnable RMSNorm scales (paper says γ preserves expressivity) | Still worse |
| Various RMSNorm init tweaks, e.g. 0 at init to residual | Still worse |
| AdamH for lm_head (paper recommends this) | Broken - loss plateaus (see below) |
| AdamH + learnable output scales | Still worse |

Could not outperform the baseline implementation. The article doesn't go into too much detail on how AdamH is applied to `lm_head` exactly. The classifier layer has to be able to increase in magnitude to make more confident predictions over time. Tried a sensible version with added 0-D learnable scalar, and also with RMSNorms with per-channel learnable scalars both pre and post resnet blocks.

**Result:** This was not an out-of-the-box win for nanochat even with a mild attempt over a few hours at a bit of tuning and debugging. The idea itself is intuitively appealing. Might come back around later to try harder later.

---

## 2026-01-28: Reverted Bigram Hash Embeddings

Removed bigram embeddings (engram-lite) from the codebase. At larger scale (d25), the improvement was tiny and disappeared entirely when measured by wall clock time. It also bloated the VRAM used. The extra parameters and complexity aren't justified.

---

## 2026-01-27: Bigram Hash Embeddings (Engram-lite)

Explored N-gram memory modules inspired by the [DeepSeek Engram paper](https://arxiv.org/abs/2601.07372) and [modded-nanogpt PR #201](https://github.com/KellerJordan/modded-nanogpt/pull/201).

### Background

The Engram paper introduces "conditional memory" as a complement to MoE - using O(1) hash lookups to retrieve static N-gram patterns instead of reconstructing them through computation. Key insight: transformers waste early layers "simulating retrieval through computation" for patterns like named entities and formulaic phrases that could be simple table lookups.

### What We Tried

**1. Full Engram module with context-aware gating (paper design)**
```python
# Hash bigrams to retrieve embeddings, then gate with hidden state
e = embed(hash(prev_token, curr_token))
q = RMSNorm(h)           # hidden state as query
k = RMSNorm(W_k @ e)     # projected embedding as key
v = W_v @ e
α = sigmoid(q · k / √d)  # scalar gate per position
output = α * v
```
- Injected after block 1 (paper found early injection optimal)
- Slight improvement, but quite a bit of complexity added.

**2. Early-layer only injection**
- Only inject bigram signal in first 4 layers (where paper claims static pattern offloading helps most)
- **Result:** Actually hurt performance. The model seems to need uniform injection across all layers.

**3. Trigrams**
- Extended to hash both 2-grams and 3-grams, concatenating embeddings
- **Result:** No improvement over bigrams alone. Dilutes capacity from more frequent 2-gram patterns.

**4. Bigram-only with x0-style injection (modded-nanogpt engram-lite approach)**
- Simple hash: `(36313 * curr) XOR (27191 * prev) mod table_size`
- Zero-init embedding table, learned per-layer lambdas
- Add to residual at every layer: `x = resid_λ[i]*x + x0_λ[i]*x0 + bigram_λ[i]*x0_bigram`
- **Result:** This simple approach works and provides a consistent improvement.

TLDR The winning approach follows modded-nanogpt's "engram-lite", simply adding the following module and feeding its output into the residual branch (gated by a per-layer learnable \lambda) before every single block:

```python
class BigramEmbed(nn.Module):
    def __init__(self, vocab_size, embed_dim, table_multiplier=5):
        self.embed = nn.Embedding(vocab_size * table_multiplier, embed_dim)

    def forward(self, idx):
        h = (36313 * idx[:, 1:]) ^ (27191 * idx[:, :-1]) % (table_size - 1)
        return self.embed(h)
```

As for optimal hyperparameters:

- **Table size:** `vocab_size * 5` (~164K entries for 32K vocab). Swept a number of settings and 5 was optimal.
- **Injection:** Every layer via learned `bigram_lambdas` (init 0.1 was better than 0.0).
- **Normalization:** Also tried adding a `norm()` to the embeddings (mirroring the token embeddings), this was slightly worse.
- **Init:** Zero-init embedding, so starts as identity (tried small noisy init, it's worse)
- **Optimizer:** AdamW with same LR as token embeddings

### Key Learnings

1. **Gating didn't help at our scale.** The paper's context-aware gating mechanism (sigmoid dot-product gate) added parameters and complexity without improvement. modded-nanogpt found the same: "simple direct addition to the residual stream outperformed by a decent margin."

2. **Uniform injection beats early-only.** Despite the paper's finding that early layers benefit most, restricting injection to early layers hurt. The x0-style "add everywhere with learned lambda" pattern works better for our architecture/scale.

3. **Bigrams are sufficient.** Trigrams didn't help - the extra context doesn't pay for the diluted capacity.

4. **Scale matters.** The Engram paper's results are at 27B params with MoE. At our ~100M-1B scale, the simpler approach wins. The elaborate gating mechanism may become useful at larger scales where collision handling matters more.

### Parameters Added

For d12 model with `table_multiplier=5`:
- Bigram embedding: 32768 × 5 × 768 = ~126M params
- Per-layer lambdas: 12 scalars (negligible)

If you're keeping track, we now have *a lot* of parameters, a significant amount of them in embeddings (token embeddings, bigram embeddings, value embeddings). For example, for a d12 we now have:

```
Parameter counts:
wte                     : 25,165,824
bigram_embed            : 125,829,120
value_embeds            : 150,994,944
lm_head                 : 25,165,824
transformer_matrices    : 84,935,808
scalars                 : 36
total                   : 412,091,556
```

In other words, only about a quarter of parameters are now weight projections and the vast majority is embedding tables.

Still, on all axes (steps, wall clock time, flops), this somewhat parameter-bloated architecture beats the baseline and will now become the default.

After adding the engram-lite, I re-ran the scaling laws to determine the new optimal tokens:params ratio. I swept FLOPs in the range 1e18..1e19, exponentially strided in 4 settings (1e18, 2e18, 5e18, 1e19). I looked at a number of ways of determining the effective parameter count for the purposes of the scaling laws. The results looked like this:

```
Kaplan-style (all projections including lm_head and no embeddings)

Optimal configurations (from quadratic fits):
FLOPs        Eff Params      Tokens          Ratio      Val BPB
-----------------------------------------------------------------
1e+18        110,678,115     1,241,505,403   11.2       0.8972
2e+18        167,797,457     1,785,336,422   10.7       0.8616
5e+18        250,650,865     2,642,234,152   10.8       0.8293
1e+19        381,758,347     3,806,871,243   10.3       0.7999

N \propto C^0.54, D \propto C^0.49

Chinchilla-style (all parameters, period.)

Optimal configurations (from quadratic fits):
FLOPs        Eff Params      Tokens          Ratio      Val BPB
-----------------------------------------------------------------
1e+18        416,320,605     1,232,157,011   3.0        0.8974
2e+18        560,239,841     1,763,669,281   3.2        0.8616
5e+18        741,495,903     2,629,909,368   3.6        0.8291
1e+19        988,644,331     3,884,841,895   4.0        0.7999

N \propto C^0.37, D \propto C^0.50

Transformer-only-style (only the projections inside the transformer)

Optimal configurations (from quadratic fits):
FLOPs        Eff Params      Tokens          Ratio      Val BPB
-----------------------------------------------------------------
1e+18        80,259,665      1,315,639,547   17.2       0.8966
2e+18        131,488,566     1,864,134,141   14.5       0.8622
5e+18        220,985,474     2,595,328,843   12.1       0.8302
1e+19        401,213,504     3,328,704,512   8.5        0.7994

N \propto C^0.70, D \propto C^0.41
```

Clearly, the Kaplan-style ratios are most consistent and produce stable ~0.5 exponents for both params and tokens, meaning we can have a single fixed ratio of tokens:params for compute optimal models. This turns out to be about ~10.5, which now becomes the new default.

---

## 2026-01-19 to 2026-01-22: Optimizer Hyperparameter Sweep

Ran ~320 experiments across 6 rounds, scaling from d12→d16→d20 to find optimal optimizer hyperparameters. Added granular per-component control to `setup_optimizers()` — separate LRs and betas for embedding, unembedding, value_embeds, resid_lambdas, x0_lambdas, and Muon matrix params.

### What We Swept
- Learning rates for all 6 parameter groups
- Beta1/beta2 for all 5 AdamW groups
- Muon momentum (start/end), weight decay
- Hundreds of combinations (2-way, 3-way, 4-way, etc.)

### The Journey

**At d12**, found two independent improvement routes:
- **Route A:** emb_lr↑ (0.3→0.4), weight_decay↑ (0.1→0.15), matrix_lr↑ (0.02→0.025)
- **Route B:** x0_lr↓ (0.5→0.2), x0_beta1↑ (0.8→0.9+)

Both gave ~0.002 improvement, but combining them caused conflicts. Fine-tuning found wd=0.13, matrix_lr=0.027, emb_lr=0.38 helped slightly. Best d12 config: Route A + x0_beta1=0.95.

**At d16**, Route B became competitive with Route A. The routes still conflicted when combined.

**At d20** (target scale), everything changed:
- Fine-tuned values from d12 **actively hurt** performance
- Routes no longer conflicted
- Just `x0_beta1=0.96` alone captured nearly all the gains

### Final x0_beta1 Sweep at d20

| x0_beta1 | val/bpb | Δ vs baseline |
|----------|---------|---------------|
| **0.96** | **0.7971** | **-0.0007** |
| 0.94 | 0.7972 | -0.0006 |
| 0.90 | 0.7972 | -0.0006 |
| 0.97 | 0.7977 | -0.0001 |
| 0.98 | 0.8011 | +0.0033 💀 |

Flat plateau from 0.90-0.96, then sharp cliff at 0.97+.

### Key Learnings

1. **Hyperparameters are scale-dependent.** What works at d12 doesn't transfer to d20. The elaborate fine-tuning that won at d12 actively hurts at d20.

2. **Improvement magnitude shrinks with scale.** ~0.002 at d12 → ~0.0007 at d20. The baseline is already better-tuned for larger models.

3. **Sharp cliffs exist.** x0_beta1=0.98 is catastrophic while 0.96 is optimal.

4. **Don't over-tune on small proxies.** Validate at target scale before shipping.

### Final Recommendation

For production d20 runs, add one flag:
```
--x0-lambdas-beta1=0.96
```

Skip everything else discovered at smaller scales.

---

## 2026-01-18: More various experiments

- Tried Muon custom kernels for XXT and all the others. The improvement was there for targeted tests (~20%) but washed out completely to noise in an actual training run, especially because the Muon compute is split across all the workers. Abandoned due to complexity bloat.
- Fuse Q,K,V,O nn.Linear layers into a single QKVO Linear layer. ~Zero impact
- Tried the `sa_lambdas` that gate QKV and O. Slightly confused because of the use of rmsnorm, which erases the effect of any scalar multiplier. Helped a tiny bit (~1e-4 of loss), abandoned to control complexity.

---

## 2026-01-17: Various experiments

Modded-nanogpt uses [Value Embeddings](https://arxiv.org/abs/2410.17897) (VEs) in a funny U-shaped structure, 3 of them in total and with gates. I tried a large number of tweaks on this today:

- VEs at every layer, at alternating layers, U shaped, front and back. Alternating layers worked best, i.e. we end up with *a lot* more VEs than modded-nanogpt, at every other layer. It works better.
- Many parameters sharing ideas to reduce new parameter count, nothing here worked. All failed.
- Many ideas to reduce parameter count, the LLM hates all of them: low rank decompositions, projections. All failed.
- Gated yes or no and how much. Gate helps.

Long story short is that the models *love* Value Embeddings. It is a way to add a huge amount of capacity (parameters) to the model at almost zero cost of FLOPs, because these embeddings are simply added to the Values tensor. Any attempt to reduce the capacity of value embeddings (param sharing, low rank, projections) fail. The model wants many of them, and with all the capacity, and doing so wins across all x axes of steps, flops and wall clock. I re-ran the scaling laws and, because the models are now very parameter bloated, the optimal ratio has halved from 8 to 4! Way down lower than Chinchilla's 20 at this point.

Other experiments, looking at val/bpb as a function of all of steps, flops and wall clock time:

- Aspect ratio of 128 is worse than 64, I tried a sweep fixing FLOPs == 1e18 and 64 outperforms. The LLM prefers to be slightly thinner and longer.
- Head dim definitely prefers to be 128 instead of 64, i.e. fewer bigger heads
- Bunch of other random stuff like that.

Keeping all of this work on a private branch for now but hope to push shortly.

---

## 2026-01-17: Modded-nanogpt Ideas Sweep (Continued)

Continued testing ideas from modded-nanogpt.

| Idea | Result | Notes |
|------|--------|-------|
| Attention gates | No improvement | Per-head learnable gates on attention output. +1GB memory, decreased efficiency. |
| Batch size schedule | Abandoned | 8→16→24 with LR scaling. Made training script too bloated/complex, not worth cognitive overhead. |
| Value embeddings | Helps a lot | Experiments still ongoing, more on this later. |

---

## 2026-01-16: Flash Attention 3 Fallback to SDPA

Added automatic fallback from Flash Attention 3 to PyTorch's `scaled_dot_product_attention` (SDPA) for users without Hopper GPUs. This enables nanochat to run on older CUDA GPUs, CPU, and MPS (Apple Silicon).

### Implementation

Created `nanochat/flash_attention.py` - a unified interface that:
- Detects FA3 availability at import time (requires sm90+ / Hopper)
- Exports a `flash_attn` object matching FA3's API exactly (`flash_attn.flash_attn_func`, `flash_attn.flash_attn_with_kvcache`)
- Automatically routes to FA3 or SDPA based on hardware
- Handles tensor layout differences: FA3 uses (B, T, H, D), SDPA uses (B, H, T, D)
- Implements sliding window attention via explicit masks for SDPA
- Manages KV cache manually for SDPA (FA3 does it in-place)

### Changes to Existing Files

Changes to existing code were intentionally kept extremely minimal.

**gpt.py**: Only the import line changed and a comment

**engine.py**: Zero changes needed

**base_train.py**: Added status print and warnings:
- Prints whether FA3 or SDPA fallback is being used
- Warns about efficiency loss without FA3
- Warns about sliding window support if `--window-pattern` is not "L"

### Testing

Tests are split into two classes due to dtype/device constraints:

1. **TestFA3VsSDPA**: Comparison tests requiring Hopper GPU + bfloat16. Run both implementations on identical inputs and verify outputs match (max diff typically 0, at most ~0.004 for sliding window).

2. **TestSDPAOnly**: SDPA-only tests that run on any device with appropriate dtype. Verify forward pass, backward pass, and KV cache work correctly.

Added `_override_impl` mechanism for testing - can force 'fa3' or 'sdpa' to directly compare implementations.

### Notes

- SDPA fallback is significantly slower than FA3 especially in that it lacks the sliding window attention support
- Recommend `--window-pattern L` (full context) when using SDPA fallback

---

## 2026-01-16: Modded-nanogpt Ideas Sweep (Mostly Negative)

Tested several architectural ideas from modded-nanogpt to see if they transfer to nanochat. All of these did not help:

| Idea | Result | Notes |
|------|--------|-------|
| Half-truncated RoPE | No improvement | Only first half of head dims get RoPE (base 1024, linspace). Second half "stationary". |
| Asymmetric softcap | Slightly worse | `23 * sigmoid((x+5)/7.5)` vs our symmetric `15 * tanh(x/15)`. May only help with FP8. |
| Smear gate | Negligible | Blend each token with predecessor via learned gate. Tiny improvement not worth n_embd² params. |
| Backout | No improvement | Save activations at ~60% through network, subtract scaled version at end. |
| Skip connection | Slightly worse | Save at layer ~25%, add at layer ~50%. Also +2GB memory from storing activations. |

Value Embeddings do show promise. I need a more elaborate exploration of a few related ideas, which I leave for tomorrow.

---

## 2026-01-15: Olmo pretraining mix (Negative result)

I attempted to train on the Olmo 3 pretraining dataset [allenai/dolma3_mix-6T](https://huggingface.co/datasets/allenai/dolma3_mix-6T) instead of FineWeb-edu. I ran into a number of [errors and issues](https://huggingface.co/datasets/allenai/dolma3_mix-6T/discussions/2) trying to both download and process the dataset and then noticed some quality issues (e.g. some documents seem to be extremely short, like "5".). I managed to work around these with some sensible hacks (e.g. reject documents less than 100 characters in length) and tried to process the dataset exactly as FineWeb, re-trained the tokenizer and trained a d16 model. The CORE score decreased from 15.5 to 13.8, i.e. the result is quite a bit worse.

I am still looking to try the [DCLM dataset](https://arxiv.org/abs/2406.11794), which according to the paper should be better that FineWeb-edu. I do have some concerns that the same group both prepared the DCLM dataset *and* introduced the CORE score so I'm a bit hesitant in case there was some overfitting to CORE score adjacent data distribution.

Classifying as negative result and reverting back to FineWeb-edu for now.

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

**2. NorMuon Variance Reduction**
- Added per-neuron/column adaptive learning rate from NorMuon ([arxiv.org/pdf/2510.05491](https://arxiv.org/pdf/2510.05491))
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

Muon was changed to use Polar Express, added NorMuon variance reduction, and cautious weight decay with schedule that ramps linearly to zero. All of these changes follow modded-nanogpt repo, but all of them were also validated piece by piece to yield improvements in nanochat with the exception of the Polar Express change which was in the noise. This is default on and configurable with `--weight_decay`, using simply 0.2 and ∝ 1/width² scaling. The kwarg `--weight_decay` is therefore changing as of this change. It used to configure AdamW via standard weight decay and now it becomes exclusively used in Muon (AdamW is hardcoded to 0.0), and it is scaled based on depth.

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
