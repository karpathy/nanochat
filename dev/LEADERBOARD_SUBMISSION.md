# Run 7 candidate — d22 + MuonClip + warmdown=0.85

**Result**: 95.7 min training (3.3% faster than Run 6's 99.0 min), val_bpb **0.72106**, CORE **0.26656**.

```
core_metric            0.26656
val_bpb                0.72106
total_training_time    5743.4   (= 95.7 min)
step                   6517
```

vs Run 6 leaderboard SOTA (`a825e63`):

| | Run 6 | Run 7 candidate | Δ |
|---|---|---|---|
| total_training_time | 5934 s (99.0 min) | **5743 s (95.7 min)** | **−3.3%** |
| val_bpb | 0.71808 (Run 5 ref); 0.7190 (Run 6 our repro) | 0.72106 | +0.43% (within tolerance) |
| CORE | 0.262634 | **0.26656** | **+1.5%** |

CORE clears the 0.2626 reference by 1.5% — comfortably beyond run-to-run noise. val_bpb sits 0.43% above the 0.71800 reference (the Run 5 number, achieved with `ratio=8.7` at extra wall-clock cost; Run 6 itself sits at 0.7190).

## Launch (mirrors `runs/speedrun.sh` style — no hardcoded iterations)

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=22 \
    --target-param-data-ratio=12 \
    --total-batch-size=1048576 \
    --device-batch-size=16 \
    --warmdown-ratio=0.85 \
    --muon-qk-clip-tau=100 \
    --fp8 \
    --run=$WANDB_RUN
```

## What changed (4 things)

### 1. `--depth=22 --target-param-data-ratio=12`
Run 6 uses `d24 + ratio=8` ("undertrain a slightly-too-big model"). I take the dual: **`d22 + ratio=12`** ("overtrain a slightly-too-small model"). At d22 the same compute budget approaches compute-optimal (10.5) from above, and the per-iter wall-clock is meaningfully cheaper.

Generalizes: drop in for any depth — overtrain when below GPT-2 capability, undertrain when above. Run 6's doc explicitly suggests this as the principled lever.

### 2. `--total-batch-size=1048576`
Explicit, mirrors Run 3's [Auto Batch Size Scaling](dev/LOG.md). Locks the d24-tuned 1 M batch in for d22 deterministically across hardware.

### 3. `--warmdown-ratio=0.85` (Run 6 default 0.65)
**Critical**: warmdown=0.85 *alone* at d22 regresses to CORE 0.2489 (below GPT-2 floor). Only combined with MuonClip does it net +0.005 CORE over default 0.65. The longer low-LR tail amplifies whatever attention-side stability MuonClip provides.

Inspired by trapezoidal-schedule findings (DeepSeek-V2/V3, Qwen2). At d22 I tested 0.50/0.65/0.75/0.85 — 0.85 is the peak with MuonClip; the rest regress with or without it.

### 4. `--muon-qk-clip-tau=100` (NEW flag, single small code change)
Kimi K2 § A QK-Clip ([arXiv:2507.20534](https://arxiv.org/abs/2507.20534)). After each Muon step, rescales `c_q`/`c_k` so the Frobenius/√(min_dim) spectral-norm estimate ≤ √τ. Caps max attention logit ≈ τ; defends Muon's repeated orthogonalization against logit blowup over long warmdown tails.

Implementation: 66 LOC across 3 files; default τ=0 leaves Run 6 behavior bit-identical. Sharp τ-peak at 100 (verified 1500-iter sweep at d22: τ=50→CORE 0.1953, **τ=100→0.2005**, τ=200→0.1917).

| file | LOC | purpose |
|---|---|---|
| `nanochat/optim.py` | +44 | `_apply_qk_clip()` helper, called after `MuonAdamW.step()` and `DistMuonAdamW.step()` |
| `nanochat/gpt.py` | +20 | `setup_optimizer(muon_qk_clip_tau=0.0, …)`; pulls `c_q`/`c_k` into a dedicated Muon group with `is_qk=True, qk_tau=tau` when `tau > 0` |
| `scripts/base_train.py` | +2 | `--muon-qk-clip-tau` arg, threaded to `setup_optimizer` |

## Ablation map — what doesn't work

The recipe above is the **only configuration in the sweep that comfortably crosses both leaderboard thresholds in less wall-clock than Run 6**; every other combination of the same knobs regresses on at least one axis.

| run | recipe | val_bpb | CORE | ttt min | verdict |
|---|---|---|---|---|---|
| **v213 (this submission)** | **d22 r=12 + wd=0.85 + muonclip** | **0.7211** | **0.2666** | **95.7** | **submission** |
| v206 | d24 r=8 + muonclip | 0.7188 | 0.2646 | 99.0 | tied with Run 6 wall-clock |
| v208 | d22 6000 + wd=0.85 + muonclip | 0.7241 | 0.2646 | 88.2 | val too high (sub-90 attempt) |
| v209 | d22 6000 default | 0.7242 | 0.2610 | 87.9 | CORE thin |
| v210 | d22 + wd=0.85, no clip | 0.7241 | **0.2489** | 87.9 | warmdown alone fails GPT-2 |
| v211 | d22 + muonclip, default wd | 0.7241 | 0.2569 | 88.1 | clip alone marginal |
| v214 | d24 r=7.5 + lr=0.025 + wd=0.85 + clip | 0.7209 | **0.2558** | 92.9 | ratio reduction breaks CORE |
| v215 | d24 r=8 + clip + lr=0.025 | 0.7189 | **0.2585** | 99.0 | matrix-lr=0.025 hurts CORE at d24 |
| v216 | d22 r=11 + wd=0.85 + clip | 0.7242 | **0.2564** | 87.7 | sharp CORE cliff at r=11 |
| v217 | d22 r=11.5 + wd=0.85 + clip | 0.7226 | 0.2596 | 91.8 | between cliffs |

Earlier private exploration (separate fork; pre-Run 6 code) also covered:
- **MLA — DeepSeek-V2 latent attention** ([arXiv:2405.04434](https://arxiv.org/abs/2405.04434)): implemented; lost CORE at d22.
- **GQA / MQA via head-divisor knob**: d22 has prime n_head=11 with default head_dim=128, so GQA collapses to MQA which regressed CORE by ~0.016. head_dim=64 + GQA 2:1 was iso-wallclock-positive at 2000-iter but saturated below v73 at 6000-iter.
- **NoPE** ([Haviv et al. 2022, arXiv:2203.16634](https://arxiv.org/abs/2203.16634)): −0.015 CORE at d22.
- **Chunked cross-entropy**: bit-identical loss, no wall-clock savings at d22 (logits not the bottleneck).
- **Qwen3.6-style attention-output gate** ([config](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json)): best val_bpb of any d22 run (0.7211), but failed CORE; gate adds n_embd² params/block and ate the wall-clock budget.
- **Rephrased pretraining (WRAP, [arXiv:2401.16380](https://arxiv.org/abs/2401.16380)); MATES reweighting ([arXiv:2402.09739](https://arxiv.org/abs/2402.09739))**: out of scope; both need an offline data-gen pipeline.

The takeaway is the same one autoresearch round 2 found and Run 6 already encodes: at this compute scale, **architecture-side novelty is mostly dead headroom** — you're either fighting tightly-tuned interactions or not paying for what you add. The remaining gains live in **optimizer-level fixes** (MuonClip) and **schedule shape** (warmdown tail). Both are small, principled, and compose with everything else in the recipe.

## Generalization to a depth miniseries

The four changes are either independent of depth (`muon-qk-clip-tau`, `warmdown-ratio`, `total-batch-size`) or scale predictably with it (`depth/ratio` is the same lever Run 6 uses, just from the other side):

- d12 / d16 / d20 / d22 / d24 / d26 — set `--target-param-data-ratio` so the side below GPT-2 capability gets `ratio > 10.5` and the side above gets `ratio < 10.5`.
- Keep `--muon-qk-clip-tau=100` and `--warmdown-ratio=0.85` constant — both are recipe-level invariants, not depth-tuned.

## References

- **Kimi K2** technical report (MuonClip / QK-Clip), [arXiv:2507.20534](https://arxiv.org/abs/2507.20534) §A
- **Muon optimizer** baseline ([Jordan et al. 2024](https://kellerjordan.github.io/posts/muon/), incorporated into nanochat from modded-nanogpt)
- **Karpathy's nanochat** repo and Runs 1–6 (this PR builds directly on Run 6, commit `a825e63`)
- **Karpathy autoresearch round 2** writeup: [tweet](https://x.com/karpathy/status/2031135152349524125) and [Run 5 commit](https://github.com/karpathy/nanochat/commit/6ed7d1d82cee16c2e26f45d559ad3338447a6c1b)
- **DeepSeek-V2** MLA (evaluated, abandoned), [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- **Qwen3.6-27B** gated attention (evaluated, abandoned), [HF model card](https://huggingface.co/Qwen/Qwen3.6-27B)
- **NoPE** (Haviv et al., evaluated, abandoned), [arXiv:2203.16634](https://arxiv.org/abs/2203.16634)
- **WRAP** rephrased pretraining (out of scope), [arXiv:2401.16380](https://arxiv.org/abs/2401.16380)

## Reproduction

Branch [`upstream-run6-muonclip`](https://github.com/giovannizinzi/nanochat-gio/tree/upstream-run6-muonclip) on this fork — `upstream/master` + the 3-file MuonClip patch:

```bash
git clone -b upstream-run6-muonclip https://github.com/giovannizinzi/nanochat-gio.git
cd nanochat-gio
# follow runs/speedrun.sh for venv/tokenizer/data setup, then use the launch above
```
