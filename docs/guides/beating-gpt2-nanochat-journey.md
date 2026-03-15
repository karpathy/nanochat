---
title: "Beating GPT-2 for <<$100: the nanochat journey"
summary: ""
status: active
source: https://github.com/karpathy/nanochat/discussions/481
source_date: 2026-01-31
last_updated: 2026-01-31
---

# Beating GPT-2 for <<$100: the nanochat journey

> Source: [https://github.com/karpathy/nanochat/discussions/481](https://github.com/karpathy/nanochat/discussions/481)

When OpenAI released GPT-2 in February 2019, training the largest model (1.5B parameters) required serious compute:

- **Hardware:** 32 TPU v3 chips (256 TPU v3 cores, 8 cores per chip)
- **Training time:** "A bit over a week" (~168 hours)
- **Cloud cost:** At $8/hour per TPU v3, that's `32 × 168 × $8 = `**$43,000**

Sources: [Reddit thread from 2019](https://www.reddit.com/r/MachineLearning/comments/aqlzde/comment/eghhhyj/), [HuggingFace model card](https://huggingface.co/openai-community/gpt2-xl).

Beating GPT-2 for <$100 from scratch has been a bit of an odd obsession for me but finally here we are. Seven years later, we can beat GPT-2's performance in nanochat ~1000 lines of code running on a single 8XH100 GPU node for ~3 hours. At ~$24/hour for an 8×H100 node, that's **$73**, i.e. **~600× cost reduction**. That is, each year the cost to train GPT-2 is falling to approximately 40% of the previous year. (I think this is an understimate and that further improvements are still quite possible). The gains come from everywhere: better hardware (H100 vs TPU v3), better software (Flash Attention 3, torch.compile), better algorithms (Muon optimizer, architectural improvements), and better data (FineWeb-edu).

<img width="1789" height="270" alt="image" src="https://github.com/user-attachments/assets/c58d589f-fba8-4bb6-97fe-0f964e59969c" />

*Above: a nicely uneventful run of training a GPT-2 capability model, this one even a little bit better after tuning the warmdown ratio slightly from 0.4 to 0.5. The training time on x axis appears a bit longer on wandb because it includes inline evaluation.*

## The Goal

Our target is the [CORE metric](https://arxiv.org/abs/2406.11794) from the DCLM paper—a comprehensive evaluation across 22 high-quality benchmarks. GPT-2's CORE score is **0.256525**. I introduced a new leaderboard to track how long it takes to reach this performance:

| # | Record time | Description | Date | Commit | Contributors |
|---|-------------|-------------|------|--------|--------------|
| 1 | 3.04 hours | d24 baseline, slightly overtrained | Jan 29 2026 | 348fbb3 | @karpathy |

The leaderboard tracks wall-clock training time (excluding eval/logging) to beat GPT-2's CORE score on 8×H100. The leaderboard is very much inspired by the one in [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) repo, except our target is CORE score instead of validation loss, and our goal is GPT-2 specifically. Contributions to improve on this are welcome! Most of your work will probably be in only one of 3 files: `base_train.py` (main driver), `gpt.py` (arch), and `optim.py` (optimizer), though it's possible that gains can be made by tuning the dataset or the tokenizer as well.

## The Jan 29 Model: Architecture Deep Dive

A few words on the current record-holding model.

### Model Architecture (`nanochat/gpt.py`)

**The basics:**
- 24 layers, 1536 channels (depth × 64 aspect ratio), 12 heads with 128 head dim
- Parameter counts:

```
wte                     : 50,331,648
value_embeds            : 603,979,776
lm_head                 : 50,331,648
transformer_matrices    : 679,481,856
scalars                 : 48
total                   : 1,384,124,976
```

(wte are token embeddings, transformer_matrices are projections inside the transformer (MLP and attention)).

**Departures from vanilla Transformer:**

1. **RoPE instead of learned positional embeddings.** Standard now, but worth noting. Base theta 10,000, computed once and cached.

2. **RMSNorm everywhere, no learnable params.** Just `F.rms_norm(x, (x.size(-1),))`. No gamma/beta. Applied after embedding, before each attention/MLP, and before lm_head.

3. **QK normalization.** After applying RoPE to Q and K, we normalize them: `q, k = norm(q), norm(k)`. Stabilizes attention without softcapping the attention weights.

4. **Untied embedding/unembedding.** `wte` and `lm_head` are separate parameters with different initializations and learning rates.

5. **ReLU² activation.** `F.relu(x).square()` instead of GELU. Sparse and cheap.

6. **Logit softcapping.** `15 * tanh(logits / 15)` bounds logits to [-15, 15]. Computed in float32.

7. **Sliding window attention.** Pattern `SSSL` = 3 short-window layers (1024 tokens), 1 long-window layer (2048 tokens), tiled across depth. Final layer always full context. I've first seen this in the GPT-3 paper. Flash Attention 3 makes this very efficient with their support for `window_size` kwarg.

8. **Value Embeddings (VE).** At alternating layers, we add a gated value embedding to the V tensor:
   ```python
   ve = value_embeds[layer_idx](token_ids)  # (B, T, kv_dim)
   gate = 2 * sigmoid(ve_gate(x[:, :, :32]))  # range (0, 2)
   v = v + gate * ve
   ```
   These add massive parameter count (~150M for d24) at near-zero FLOPs.

9. **Per-layer residual scalars.** Two learnable scalars per layer:
   ```python
   x = resid_lambdas[i] * x + x0_lambdas[i] * x0
   ```
   Where `x0` is the initial normalized embedding. `resid_lambdas` init to 1.0, `x0_lambdas` init to 0.1.

10. **Flash Attention 3.** Native `(B, T, H, D)` layout. Falls back to PyTorch SDPA on non-Hopper GPUs.

### Optimizer (`nanochat/optim.py`)

**Split optimizer design:** AdamW for embeddings/scalars, Muon for weight matrices.

**AdamW groups:**
- `lm_head`: lr=0.004, scaled by 1/√(dim/768)
- `wte` + `value_embeds`: lr=0.3, same scaling
- `resid_lambdas`: lr=0.005 (scalar_lr × 0.01)
- `x0_lambdas`: lr=0.5, beta1=0.96 (higher than default 0.8)

**Muon for matrix params:**
- All attention projections (Q, K, V, O) and MLP weights
- Grouped by shape, stacked for efficient batched updates

**Muon internals (see `muon_step_fused`):**
1. **Nesterov momentum** with warmup 0.85→0.95 over first 300 steps
2. **Polar Express orthogonalization** (5 iterations) instead of Newton-Schulz
3. **Factored variance reduction** (Adafactor-style): maintains low-rank second moment buffer
4. **Cautious weight decay**: only decays where `grad * param >= 0`, linear schedule to zero

**Distributed optimizer (`DistMuonAdamW`):**
- ZeRO-2 style sharding: each rank owns a slice of optimizer state
- 3-phase async: launch reduce → compute updates → gather results
- No DDP—gradient sync happens in the optimizer step

### Training Script (`scripts/base_train.py`)

**Key hyperparameters:**
- Batch size: 524,288 tokens (32 × 2048 × 8 GPUs)
- Warmdown: 50% of training (linear LR decay to 0)
- Weight decay: 0.2 at d12, scaled by (12/depth)²
- Tokens:params ratio: 10.5 (compute-optimal), or 12 for speedrun (slight overtrain)

**Data pipeline:**
- BOS-aligned dataloader: every sequence starts with `<|bos|>`
- BestFit-Crop packing: 100% utilization, ~35% token waste from cropping
- FineWeb-edu, ~8.8B tokens in total needed

**Scaling via depth:** The `--depth` flag is the single knob. Everything else derives from it:
- `model_dim = depth × 64`
- `num_heads = model_dim / 128`
- Optimal token budget scales with depth
- Weight decay scales with 1/depth²

The majority of these optimizations have been cherry picked and adapted from the [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) repo. Not all of the things in the modded-nanogpt worked for nanochat, and based on some recent chatter - vice versa :)

---

## The Optimization Journey

We started with a vanilla Transformer (learned positional embeddings, LayerNorm, GELU, AdamW, Flash Attention 2). Here's what changed.

### What Worked

- **Flash Attention 3** — ~9% tok/sec improvement. Native tensor layout, single API for training and inference.
- **Sliding window attention** — `SSSL` pattern. Compute savings without quality loss.
- **Muon optimizer overhaul** — Polar Express, NorMuon variance reduction, cautious weight decay with linear schedule to zero. The cautious WD was a clear win. I tried to delete Muon and couldn't.
- **Per-layer residual scalars** — `x = λ_resid * x + λ_x0 * x0`. Consistent improvement across all model sizes (0.003-0.01 bpb).
- **Value Embeddings at alternating layers** — Models love the value embeddings capacity. Any attempt to reduce it (low-rank, sharing, projections) hurt. We tried U-shaped placement, every layer, alternating—alternating won.
- **BOS-aligned dataloader** — Every row starts with BOS. Made midtraining unnecessary (deleted it). BestFit-Crop packing reduces waste vs naive cropping.
- **Hyperparameter sweep at scale** — 320 experiments to find that `x0_beta1=0.96` is optimal at d20. Key lesson: small-scale tuning doesn't transfer. Validate at target scale.
- **Scaling law discovery** — We empirically measured the optimal tokens:params ratio to be ~10. It's important to do the actual experiment on your own network.

### What Didn't Work

- **Multi-token prediction (MTP)** — +13GB memory, no improvement
- **Varlen attention** — BOS-aligned dataloader already handles this to some extent. Attending across BOS document boundaries does not seem to make things much worse.
- **FP8 for lm_head** — Works, but +2GB memory (!), only 1% speedup, todo to look into more.
- **Half-truncated RoPE** — No improvement
- **Asymmetric softcap** — Slightly worse
- **Skip connections / backout** — No improvement, +2GB memory
- **Smear gate, attention gates** — Negligible improvement, not worth complexity
- **Batch size schedule** — Deemed a little too complex
- **Bigram embeddings (Engram-lite)** — Works, but not by too much, and it bloats complexity and parameter count by a lot, so it was skipped in the end.
- **Hyperball/MuonH** — Intriguing idea, didn't work out of the box

See `dev/LOG.md` for detailed experiment notes on each. Note that it is very difficult (/impossible) to rule out an idea. Sometimes you have to try multiple times. I'm only chronicling some of the things that worked and didn't work out of the box, trying with at most medium amount of effort.

---

## Reproduce

Here is how I trained the Jan29 model on commit `348fbb3`. Boot up your 8XH100 node (e.g. from Lambda or etc.), run the setup (see `runs/speedrun.sh`, you can just run the commands individually one by one to set up the environment, download the data shards and train the tokenizer), then run pretraining like this:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m nanochat.scripts.base_train -- \
    --depth=24 \
    --run=d24-jan29 \
    --model-tag=d24_jan29 \
    --device-batch-size=16 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=3000 \
    --target-param-data-ratio=12
```

Wait 3 hours to see:

```
wandb: Run summary:
wandb:          core_metric 0.25851
wandb:                 step 16704
wandb: total_training_flops 4.330784131228946e+19
wandb:  total_training_time 10949.46713
```

- **CORE Score: 0.25851** (GPT-2: 0.256525) ✓
- **Training Time: 3.04 hours** (10,949 seconds)
- **Cost: ~$73** (at ~$24/hour for 8×H100)

See `runs/speedrun.sh` script for more detailed reference.

If you don't have hundreds of hours to spend on training GPT-2, you can experiment and find improvements on much smaller scales, e.g. just use `--depth=12` to train a d12 (it trains in only ~5 minutes), or try a d16. A lot of my iteration is on a smaller scale and many (but not all!) ideas that work there transfer to the bigger models.

## Discord

Come talk about further improvements on `#nanochat` on our [Discord](https://discord.com/channels/1020383067459821711/1427295580895314031), or [alternative link to try](https://discord.gg/3zy8kqD9Cp).

## Acknowledgements

This work builds heavily on [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt). Many winning ideas originated there: Muon improvements, Value Embeddings, per-layer scalars, etc. Thanks to HuggingFace for FineWeb-edu, to Tri Dao and friends for FA3 kernels, Lambda for compute, and everyone who contributed.

## Samples fun

For fun, here are some of the samples from the model, printed by `base_eval.py`.

First, conditional samples. Prompts are:

```
prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
```

The samples become:

```
--------------------------------------------------------------------------------
<|bos|>The capital of France is Paris. It is the largest city in France and the second largest city in Europe
--------------------------------------------------------------------------------
<|bos|>The chemical symbol of gold is Au. It is a soft, malleable, ductile, and lust
--------------------------------------------------------------------------------
<|bos|>If yesterday was Friday, then tomorrow will be Saturday. If today is Tuesday, then tomorrow will be Wednesday. If today is
--------------------------------------------------------------------------------
<|bos|>The opposite of hot is cold. The opposite of hot is cold. The opposite of hot is cold.
--------------------------------------------------------------------------------
<|bos|>The planets of the solar system are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune
--------------------------------------------------------------------------------
<|bos|>My favorite color is blue. I love the sky, the ocean, the blue sky. I love
--------------------------------------------------------------------------------
<|bos|>If 5*x + 3 = 13, then x is a factor of 5.
```

So the model has pretty decent knowledge! Unconditional samples:

```
--------------------------------------------------------------------------------
<|bos|>The announcement in July 2020 was further our plans to stop the spread of COVID-19 throughout our service by making and using disposable personal protective equipment in the Volvo VT 360 Sprint bioreversion and plug-in/mobile.
Young people, older people, and people with disabilities hesitate to join technology advantages programs. It is critical to discuss technology with the participants and be sure to challenge the ideas and stances that they will research before signing up on an organization. Remember that a student may have a diverse social circle outside of their country and be in a more favorable position to speak up about other parts of their truth by
--------------------------------------------------------------------------------
<|bos|>New York Times, Monday, September 22, 2012
From the Editor:
by Kenneth Koike, Follow him on Twitter, at KNOCKforlifestyle.blogspot.com, and email@example.com
Tax revolt in Alaska has just begun; it has arrived but been scattered with such force that a full state revolt will not likely take place this year. The surprise came again as the result of two consecutive rejections of this year’s election—thanks to two separate populations. It began at a vote in June that protected Alaska’s oil business until Congress made the law require it to limit its support for U
--------------------------------------------------------------------------------
<|bos|>Inquiry Based Projects in the Science Classroom: Bringing Foil Can Biology to Life
Because of my professional training, I am often asked about how you can bring biology and core features of Inquiry Based Science (IBS) to K-12 science students. I would like to share with other teachers and students what I've learned about tools to bring the inquiry process into the classroom and in the life of students. I will also share other projects I have participated in that yield tremendous results.
At the onset, I built a strong foundation of science teaching, such as classroom observation and TajidarK (Horwitz, D. &
--------------------------------------------------------------------------------
<|bos|>Civil war over the state flower of West Virginia never materialized. But the story of West Virginia’s Forgotten Flower stands as a reminder that weathering misfortune with grace and courage is exactly what many of the characters in my historical novels don’t have. But on a brighter happier note, the Forgotten Flower makes a special appearance: when the movie Kum 19 (1920) started filming in the early 1960s, it included a tribute to the Seneca Indians.
The Seneca, a North American Indian tribe, were historically Casimir Pulaski and they rose to prominence in the Revolutionary War after their loyalty
--------------------------------------------------------------------------------
<|bos|>Dialysis is becoming increasingly popular as a treatment alternative for patients with end-stage renal disease (ESRD) and diabetes. The United States of America (USA) is one of the leading dialysis centers worldwide. The demand for dialysis in Baltimore, Maryland is strong. This study assessed the willingness of patients to receive outpatient dialysis during the 2015 dialysis service season (March -June 2015). Non-sectional population-based study across community dialysis centers in Baltimore, Maryland, USA.
All registered dialysis centers in Baltimore, Maryland, USA (males 3.0%; females 6.60%) during the 2015 dialysis
--------------------------------------------------------------------------------
<|bos|>Freedom in the World
Freedom Rating (1 = best, 7 = worst)
Civil Liberties (1 = best, 7 = worst)
Political Rights (1 = best, 7 = worst)
Administrative changes in 2015 led to constitutional reforms that resulted in a new state legislative assembly. Earlier proposals by lawmakers to broaden the law’s authority to remove press-censorship-related laws were blocked, and opposition parties boycotted the legislative process.
After a decade of relative social and economic stability, Fiji was rocked by financial collapse and political coups, often blamed on the lack of
--------------------------------------------------------------------------------
<|bos|>SANEYLETH - Every few months we get evocative statements from our cities — we are on fire, we are drifting away from our Ṭula-, we have too much water and we can't get enough.
According to Sneennyards, sea level is rising! Camelond Bay is threatened with erosion!
But does the rise in sea level really tell anything about the water level in the PNF tankers that ply our urban waterways, our canals, our rivers?
“Most of the waterways are tidal with half a metre of ‘mean’ high tide,” goes todays Live Science headline. So
--------------------------------------------------------------------------------
<|bos|>The bladder is located in the lower part of the abdomen [Fig.1 a-d.]. The bladder stores urine until it leaves the body through a tube (urethra) that is passed through the opening (sphincter) of the vagina. The sphincter is under strong muscular and skeletal control. It keeps the urethra from leaking.
The author introduced this cadaver to describe the bladder:
The bladder is a hollow organ with a circular opening on the outside [Fig.2]. When we describe the bladder, we refer collectively to this organ and its different chambers as the "cylinder" [Fig.3
```
