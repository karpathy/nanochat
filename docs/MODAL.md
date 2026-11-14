# Training on Modal (Cloud GPU)

Train a GPT-2-grade LLM on Modal's 8×H100 GPU cluster. This guide covers the quickest path from zero to a working chat model.

## Prerequisites

You need these **once per account**:

1. **Modal account** — sign up at [modal.com](https://modal.com)
2. **8×H100 entitlement** — Modal needs to grant this (request in the dashboard or contact support; new accounts may default to smaller GPU pools)
3. **Local `modal` CLI** — install with `pip install modal` or `uv tool install modal`, then run `modal setup` (this authenticates your CLI to your Modal account)

You do not need a local GPU to run training on Modal, but you do need enough local disk and RAM to download the trained model afterward (see [Chat locally](#chat-locally) below).

## Cost and time

- **Cost:** ~$40–60 for a full run (~$3/GPU/hr × 8 GPUs × ~1.5–2h training)
  - First run is higher: Modal builds a Docker image and compiles PyTorch, adding ~0.5–1h and proportional cost
  - Later runs reuse the image and cost less
- **Wall-clock time:** ~1.5–2h training, plus ~30 min prefetch (CPU) and ~15 min image build (first run only)
  - Keep your laptop awake; the job is attached to your terminal
  - Use `screen` or `tmux` if you expect network drops (see [RUNNING.md](RUNNING.md) for tmux tips)

The script shows cost and time estimates before charging anything and asks for confirmation.

## One command

```bash
bash runs/runmodal.sh
```

The script:
1. Checks that `modal` is installed and configured
2. Shows a cost/time banner and asks `Proceed? [y/N]`
3. Runs the training pipeline on Modal:
   - **Prefetch** (CPU on Modal): downloads 170 shards of ClimbMix data to a Modal Volume (~17 GB)
   - **Train** (8×H100): runs speedrun.sh with GPU training (~1.5–2h)
   - **Download**: pulls the trained tokenizer and SFT checkpoints back to your laptop under `~/.cache/nanochat-modal`
   - **Print local chat tip**: shows the command to chat with your model locally

For scripted/automated runs, skip the confirmation with:
```bash
RUNMODAL_YES=1 bash runs/runmodal.sh
```

## What happens inside

The job is split into stages, each marked in a persistent Modal Volume so re-runs can skip finished work:

| Stage | What it does | Artifact(s) |
|-------|--------------|-----------|
| Prefetch | Download 170 shards of ClimbMix (CPU, ~30 min) | `base_data_climbmix/` on Volume |
| Tokenizer | Train a BPE tokenizer (~5 min, GPU) | `tokenizer/tokenizer.pkl` |
| Base train | Pre-train the LLM (~45 min, 8×H100) | `base_checkpoints/` |
| Base eval | Measure CORE metric (~10 min) | `.eval_ok` marker |
| SFT | Fine-tune for chat (~15 min) | `chatsft_checkpoints/` |
| Chat eval | Evaluate on MMLU/GSM8K (~10 min, optional) | `.eval_ok` marker |

Each stage's marker file lives on the Volume, so a re-run of `runmodal.sh` skips finished stages and only redoes stages that failed or were forced (`FORCE_STAGES=stage_name`).

## Chat locally

After a successful run, your trained model is on your laptop at `~/.cache/nanochat-modal/`. To chat with it:

### Prerequisites for chatting
- Local Python environment with PyTorch: run `uv sync --extra cpu` in the repo (same as for `runcpu.sh`)
- Downloaded model: `tokenizer/` and `chatsft_checkpoints/` (the script downloads these automatically on success)
- ~8 GB of free disk space and ~4 GB of free RAM (to load the checkpoint)

### Chat command
```bash
NANOCHAT_BASE_DIR=~/.cache/nanochat-modal python3 -m scripts.chat_cli -p "Why is the sky blue?"
```

Start an interactive session with:
```bash
NANOCHAT_BASE_DIR=~/.cache/nanochat-modal python3 -m scripts.chat_cli
```

The model is a ~1.6B parameter GPT-2-grade transformer. It is not as capable as GPT-3.5, but it makes reasonable guesses and can write stories, poems, and answer factual questions (sometimes incorrectly). Have fun.

If the training failed and the model files are missing, the script exits with an error and does not print a success tip.

## Idempotent re-runs

You can safely re-run `bash runs/runmodal.sh` multiple times:

- **Finished stages SKIP** — their markers exist on the Volume, so the data/checkpoint is not re-computed
- **Killed mid-stage** — that stage reruns on the next call (no mid-training resume; you start that stage over)
- **Data caches do not collide** — Modal stores artifacts on the Volume; your laptop uses `~/.cache/nanochat-modal/`, separate from the local `~/.cache/nanochat` (if you run `runcpu.sh` or `speedrun.sh` locally)

To force a stage to rerun (useful if you made code changes):
```bash
FORCE_STAGES=base_train,base_eval bash runs/runmodal.sh
```

## Stopping a run

If you need to stop training:

1. **Stop the local client:** press `Ctrl-C` in your terminal
2. **Verify the remote job stopped:** run `modal app list` to see running jobs, and `modal app stop <app>` if needed

Marked as "verify on first real run" — the exact behavior depends on Modal's scheduler. After confirming the app stopped, billing should cease.

## Storage map

Where artifacts live:

| Artifact | Location | Size |
|----------|----------|------|
| ClimbMix shards (pretraining data) | Modal Volume `/data/nanochat-modal/base_data_climbmix/` | ~17 GB |
| Tokenizer | Modal Volume `/data/nanochat-modal/tokenizer/` | ~100 MB |
| Base checkpoints (intermediate) | Modal Volume `/data/nanochat-modal/base_checkpoints/` | ~2–3 GB |
| SFT checkpoints (final model) | Modal Volume `/data/nanochat-modal/chatsft_checkpoints/` | ~3–5 GB |
| Local chat copy (after download) | `~/.cache/nanochat-modal/tokenizer/` + `/chatsft_checkpoints/` | ~3–5 GB |
| HF caches (transformers, tokenizers) | Modal Volume `/data/nanochat-modal/hf_home/` | ~2 GB |

The Volume persists across runs, so stages can skip if their marker exists. Your laptop's `~/.cache/nanochat-modal/` is populated only after a successful full run.

## Which scripts apply to Modal

In the `runs/` directory:

| Script | What it does | Modal? |
|--------|--------------|--------|
| `speedrun.sh` | Full 8×H100 training (the MVP) | **Yes** — this is what Modal wraps |
| `runcpu.sh` | Tiny demo (depth 6) for laptops | No — local CPU/MPS only |
| `miniseries.sh` | Research sweep over multiple depths | Later (not in MVP) |
| `scaling_laws.sh` | FLOPs×depth grid | Later (not in MVP) |

Modal always uses `speedrun.sh` via `runmodal.sh`. Do not try to run `runcpu.sh` on Modal; it uses a different compute profile and will not produce a useful model.

## How Modal optionality works

When you run locally (without Modal env vars set):
- `bash runs/runcpu.sh` — uses `~/.cache/nanochat`, full CPU/MPS bootstrap, no Modal
- `bash runs/speedrun.sh` — uses `~/.cache/nanochat`, full GPU bootstrap, no Modal

When Modal runs `speedrun.sh` internally:
- Modal sets env vars (`NANOCHAT_BASE_DIR`, `SKIP_SETUP`, `NANOCHAT_POST_STAGE_HOOK`)
- Same script reuses the volume path and skips venv setup (the image has deps pre-installed)
- No code duplication; the shell scripts adapt via environment variables

Your local runs are never affected by Modal's env vars, and Modal artifacts never collide with local caches.

## Troubleshooting

**"Modal not found on PATH"**
- Run `pip install modal` or `uv tool install modal`, then `modal setup`
- Verify with `modal --version`

**"No 8×H100 entitlement"**
- Contact Modal support or check your account dashboard; new accounts may need an upgrade

**"Job timed out / killed"**
- Modal has default timeouts for long-running jobs. The script sets explicit timeouts for prefetch (2h) and GPU train (4h), so this should not happen unless there is a network issue
- Re-run; skipped stages cost nothing, and only the killed stage reruns

**"Download failed"**
- If `modal volume get` fails partway through, re-run the script; it will skip earlier stages and retry the download

**"Missing tokenizer or chatsft_checkpoints after a 'successful' run"**
- The training job failed partway through, and the script correctly did not print a success tip
- Check the job logs for the failure (see Modal dashboard or `modal logs`)
- Fix the issue and re-run with the relevant `FORCE_STAGES=…` if needed

For more details on stage markers, artifacts, re-runs, and failure diagnosis, see [docs/RUNNING.md](RUNNING.md).
