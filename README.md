# Clarinet

Clarinet is a research fork of [nanochat](https://github.com/karpathy/nanochat)
that imports the econometric notion of **instrumental variables (IV)** into the
autoregressive token generation of an LLM. The core idea: use a document's
**data source** — formal-reasoning corpora (math) vs. general web text — as an
instrument, condition the model on it during training, and then steer generation
toward the reasoning-induced component of next-token transitions at inference,
classifier-free-guidance (CFG) style.

In one sentence: Clarinet teaches the model a *source-conditioning channel* (a
marker token at the start of every document) and then exploits that channel at
decode time by running the model twice — conditional and unconditional — and
extrapolating along the direction the marker induces.

> New to the idea? Read **[docs/iv_conditioning.md](docs/iv_conditioning.md)**
> for a precise, pedagogical walk-through of the whole pipeline.

## Relationship to nanochat

Clarinet keeps the upstream `nanochat/` package largely unchanged so that
upstream improvements rebase cleanly. Clarinet's own contributions live in the
sibling `clarinet/` package plus a handful of new `scripts/` and `runs/` files:

| Concern | Upstream nanochat | Clarinet addition |
|---|---|---|
| Datasets | `nanochat/dataset.py` (climbmix) | `clarinet/dataset.py` (climbmix **+** reasoning, source-labelled) |
| Data loading | `nanochat/dataloader.py` | `clarinet/dataloader.py` (source markers, CFG dropout, target masking) |
| Inference | `nanochat/engine.py` | `clarinet/engine.py` (dual-pass IV guidance + L1 adaptive scale) |
| Reasoning corpus | — | `clarinet/prepare_reasoning_data.py` (FineMath shards) |
| Training entry | `scripts/base_train.py` | `scripts/clarinet_train.py` (wraps base_train) |
| Chat CLI | `scripts/chat_cli.py` | `scripts/clarinet_cli.py` (guidance flags) |
| Guidance sweep | — | `scripts/iv_eval.py` |

Three special tokens are added to the tokenizer (`nanochat/tokenizer.py`):
`<|src_reasoning|>`, `<|src_general|>`, `<|src_unknown|>`.

## How it works (short version)

1. **Two corpora, one source flag.** `clarinet/dataset.py` lists climbmix
   (general) and reasoning (FineMath) shards, tagging each with `is_reasoning`.
2. **Conditioning installed at the dataloader.** Every document is laid out as
   `[BOS, marker, ...tokens]`. With probability `p_uncond` (default 0.1) the true
   marker is dropped to `<|src_unknown|>` — the CFG unconditional dropout that
   makes the model also learn the neutral distribution. The target at each
   marker position is masked to `-1`, so the model never learns to *predict* the
   marker — it is purely an input conditioner.
3. **Dual-pass generation.** `ClarinetEngine` runs two lock-step passes per
   step (reasoning marker vs. unknown marker) and combines logits as
   `logit_iv = logit_uncond + w · s · (logit_cond − logit_uncond)`, where
   `w = iv_weight` and `s` is the scale factor.
4. **L1 adaptive scale (opt-in).** `s` can adapt per step to the L1
   (total-variation) distance between the two distributions — spend guidance
   where the marker actually moves the prediction, back off where it doesn't.
   The default (`scale_lo == scale_hi == 1.0`) reproduces vanilla CFG exactly.

## Getting started

### Setup

Clarinet uses [uv](https://docs.astral.sh/uv/) for dependency management:

```bash
uv sync --extra gpu    # CUDA (A100/H100/etc.)
uv sync --extra cpu    # (or) CPU-only / MPS
source .venv/bin/activate
```

For development (adds pytest, matplotlib, ipykernel, transformers, etc.):

```bash
uv sync --extra gpu --group dev
```

> Note: adding the three source-marker special tokens changes the vocabulary, so
> the tokenizer **must be retrained** before training a model. The run scripts
> below do this for you.

### Run the full Clarinet pipeline

Two reference scripts wire together data prep, tokenizer training, pretraining,
SFT, and the IV guidance sweep:

```bash
bash runs/clarinet_speedrun.sh     # 8×H100 (Hopper fp8, large batch)
bash runs/clarinet_local_run.sh    # single-GPU (e.g. ROCm in WSL2), smaller batch
```

The key Clarinet-specific steps inside those scripts are:

```bash
# 1. Build the reasoning corpus (FineMath shards)
python -m clarinet.prepare_reasoning_data -n 50

# 2. Retrain the tokenizer (new special tokens) and the model
python -m scripts.tok_train
python -m scripts.clarinet_train \
    --reasoning-mix-ratio=0.3 --p-uncond=0.1 \
    --depth=24 --target-param-data-ratio=8 \
    --run=$WANDB_RUN

# 3. Sweep the IV guidance weight (the actual experiment)
python -m scripts.iv_eval -i sft --weights 0,1.0,1.5,2.0,3.0
```

`clarinet_train.py` adds `--reasoning-mix-ratio` (default 0.3), `--p-uncond`
(default 0.1), and `--clarinet-seed`; everything else forwards to
`base_train.py` unchanged (notably `--depth`, the single complexity dial — see
below). Validation always uses `p_uncond=0.0` (true marker).

### Talk to the model with IV guidance

```bash
# vanilla CFG (default)
python -m scripts.clarinet_cli --iv-weight 2.0 --wald-scale 1.0

# L1 content-adaptive scale (opt-in; recommended starting point)
python -m scripts.clarinet_cli --iv-weight 2.0 --scale-lo 0.5 --scale-hi 2.0
```

`iv_weight=0` recovers the unconditional distribution; `iv_weight=1, s=1` the
conditional distribution exactly; `iv_weight` in ~[1.5, 2.5] is the expected
useful range. Only the **product `w·s`** affects sampling, so tune them as a
pair. The adaptive schedule's *direction* is an unvalidated hypothesis — compare
it against constant `s` on held-out reasoning likelihood / GSM8K before trusting
it.

## Inherited nanochat foundation

The pieces below come from upstream nanochat and are unchanged by Clarinet.

### One complexity dial: `--depth`

nanochat is configured around a single integer, `--depth` (the number of
transformer layers). It automatically determines width, number of heads, learning
rate, training horizon, weight decay, etc., so models come out compute-optimal.
GPT-2 capability sits around depth 24–26. For fast experimentation a 12-layer
model trains in ~5 minutes; any change to the repo should be principled enough to
work across depths.

### Precision / dtype

nanochat manages precision explicitly via a global `COMPUTE_DTYPE` (in
`nanochat/common.py`), auto-detected from hardware:

| Hardware | Default dtype |
|----------|--------------|
| CUDA SM 80+ (A100, H100) | `bfloat16` |
| CUDA SM < 80 (V100, T4) | `float32` (fp16 via `NANOCHAT_DTYPE=float16`) |
| CPU / MPS | `float32` |

Override with `NANOCHAT_DTYPE`. Weights are stored in fp32; the custom `Linear`
casts to `COMPUTE_DTYPE` in the forward pass. `float16` training auto-enables a
`GradScaler` in `base_train.py`.

### CPU / MPS

[runs/runcpu.sh](runs/runcpu.sh) (upstream) and
[runs/clarinet_local_run.sh](runs/clarinet_local_run.sh) (Clarinet) show small
runs that fit in a few tens of minutes. These are educational; you will not get
strong results.

### Speedrun leaderboard

Upstream nanochat maintains a "time-to-GPT-2" speedrun leaderboard; see
[dev/LEADERBOARD.md](dev/LEADERBOARD.md) and
[runs/speedrun.sh](runs/speedrun.sh). Clarinet does not track its own leaderboard
— its metric of interest is reasoning-task performance as a function of the IV
guidance weight, measured by `scripts/iv_eval.py`.

## File structure

```
.
├── README.md
├── NOTICE.md                       # Clarinet ↔ nanochat relationship
├── docs
│   └── iv_conditioning.md          # Technical note: how conditioning works end to end
├── clarinet                        # Clarinet's contributions
│   ├── dataset.py                  # Source-labelled climbmix + reasoning shards
│   ├── dataloader.py               # BOS best-fit loader + markers + CFG dropout + target masking
│   ├── engine.py                   # Dual-pass IV engine + L1 adaptive scale
│   └── prepare_reasoning_data.py   # Build the FineMath reasoning corpus
├── nanochat                        # Upstream nanochat (kept ~unchanged)
│   ├── dataloader.py / dataset.py  # Base loader / pretraining data
│   ├── engine.py                   # KV-cache inference (Clarinet subclasses this)
│   ├── gpt.py                      # GPT transformer (cross-entropy ignore_index=-1, softcap)
│   ├── tokenizer.py                # BPE tokenizer + the 3 new source markers
│   └── ...                         # checkpoint_manager, optim, core_eval, report, ...
├── scripts
│   ├── clarinet_train.py           # Train wrapper (mix ratio, p_uncond)
│   ├── clarinet_cli.py             # Chat CLI with IV guidance flags
│   ├── iv_eval.py                  # IV guidance-weight sweep
│   ├── base_train.py / chat_sft.py / chat_cli.py / ...   # upstream stages
│   └── ...
├── runs
│   ├── clarinet_speedrun.sh        # Full Clarinet pipeline, 8×H100
│   ├── clarinet_local_run.sh       # Full Clarinet pipeline, single GPU
│   └── speedrun.sh / runcpu.sh / ...                     # upstream runs
├── tasks                           # arc, gsm8k, mmlu, humaneval, smoltalk, ...
├── tests
│   ├── test_clarinet_engine.py     # Dual-pass + combine + L1 adaptive-scale tests
│   ├── test_clarinet_dataloader.py # Interleaving, markers, masking
│   └── test_engine.py              # upstream engine tests
├── pyproject.toml
└── uv.lock
```

## Acknowledgements

- Clarinet is a research project by Vemund Rundberget.
- It is a fork of [nanochat](https://github.com/karpathy/nanochat) by Andrej
  Karpathy; the entire training/inference/eval foundation is his. nanochat is in
  turn inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and
  [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt).
- Thank you to [HuggingFace](https://huggingface.co/) for FineWeb, SmolTalk, and
  FineMath (`HuggingFaceTB/finemath`), used as the reasoning corpus.

See [NOTICE.md](NOTICE.md) for the precise upstream relationship; both copyrights
are preserved in [LICENSE](LICENSE) per the MIT License.

## Cite

Clarinet builds directly on nanochat; please cite the upstream project:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
