# Training on Modal (Cloud CPU or GPU)

Train a small LLM on Modal (`MODE=cpu`, the default) or a GPT-2-grade LLM on Modal's 8×H100 GPU cluster (`MODE=gpu`). This guide covers the quickest path from zero to a working chat model.

| Mode | Runs | Machine | Needs |
|------|------|---------|-------|
| `cpu` (default) | `runs/runcpu.sh` (small profile, depth 6) | Up to 64 cores / 128 GiB, no GPU | Modal account, `modal` CLI |
| `gpu` | `runs/speedrun.sh` (GPT-2-grade, depth 24) | 8×H100 | The above, plus an 8×H100 entitlement and a Modal payment method |

## Prerequisites

You need these **once per account**:

1. **Modal account** — sign up at [modal.com](https://modal.com)
2. **8×H100 entitlement** (GPU mode only) — Modal needs to grant this (request in the dashboard or contact support; new accounts may default to smaller GPU pools)
3. **Local `modal` CLI** — install with `pip install modal` or `uv tool install modal`, then run `modal setup` (this authenticates your CLI to your Modal account)

You do not need a local GPU to run training on Modal, but you do need enough local disk and RAM to download the trained model afterward (see [Chat locally](#chat-locally) below).

## Runtime config and environment

### Your machine (where you run `runmodal.sh`)

- **Python 3.11 or newer** for the local `modal` CLI. The CLI warns that 3.10 is end-of-life.
- **`bash`, `git`, and the `modal` CLI** on `PATH`. The script stops if `modal` is missing.
- **Network access** during the run. The laptop stays attached to the job, so keep it awake and use `screen` or `tmux`.
- **Disk:** room under `~/.cache/nanochat-modal` for the downloaded tokenizer and chat checkpoint.

You do not need a local GPU or the training dependencies. Modal builds them into its own image. For the local install paths and the torch build per platform, see the README [Setup](../README.md#setup) section; for dtype per device, see [Precision / dtype](../README.md#precision--dtype).

### Modal side

| | CPU mode (`MODE=cpu`) | GPU mode (`MODE=gpu`) |
|---|---|---|
| Machine | 64 cores, 128 GiB RAM, no GPU | 8×H100 |
| Entry script in container | `runs/runcpu.sh` | `runs/speedrun.sh` |
| Python dependency group | `cpu` extra | `gpu` extra |
| Account needs | Nothing extra | GPU entitlement on the account |
| Train timeout | 12 h (billed by actual run time) | 4 h |
| Prefetched data shards | 8 | 170 |

The container sets `SKIP_SETUP=1`, so it uses the prebuilt `.venv` and does not reinstall. It also sets `NANOCHAT_BASE_DIR=/data/nanochat-modal` and `HF_HOME` on the shared Volume, and runs `runs/modal_hook.sh` after each stage.

### Hyperparameters

The values below are what the scripts actually use. Profile values live in [`nanochat/defaults.py`](../nanochat/defaults.py) (`PROFILE_RUNCPU`, `PROFILE_SPEEDRUN`), and `runs/profiles.sh` is generated from it. Do not hand-edit `profiles.sh`. Modal-only overrides live at the top of [`runs/modal_run.py`](../runs/modal_run.py).

**Tokenizer (both modes):** vocabulary 32768 (2^15), trained on about 2 billion characters.

**CPU mode (`runcpu.sh`, base model and SFT):**

| Setting | Value | Source |
|---|---|---|
| Depth / head dim / window | 6 / 64 / `L` | `PROFILE_RUNCPU` |
| Max sequence length | 512 | `PROFILE_RUNCPU` |
| Total batch size (tokens) | 16384 | `PROFILE_RUNCPU` |
| Device batch size, train | 32 on Modal; 8 locally | `modal_run.py`; `PROFILE_RUNCPU` |
| Device batch size, eval | 16 on Modal; 1 locally | `modal_run.py`; `PROFILE_RUNCPU` |
| Iterations, base | 5000 | `PROFILE_RUNCPU` (Modal sets the same value) |
| Eval every / eval tokens | 100 steps / 524288 | `PROFILE_RUNCPU` |
| SFT iterations / eval every | 1500 / 200 | hardcoded in `runcpu.sh` |

**GPU mode (`speedrun.sh`):**

| Setting | Value | Source |
|---|---|---|
| Depth | 24 | `PROFILE_SPEEDRUN` |
| Device batch size, train / eval | 16 / 16 | `PROFILE_SPEEDRUN` |
| Target param-to-data ratio | 8 | `PROFILE_SPEEDRUN` |
| Max sequence length | 2048 | `speedrun.sh` default |
| Processes | 8 (one per GPU), FP8 on | hardcoded in `speedrun.sh` |
| Threads | `OMP_NUM_THREADS=1` | `speedrun.sh` |

### Environment variables you can set

| Variable | Default | Effect |
|---|---|---|
| `MODE` | `cpu` | `cpu` or `gpu` |
| `RUNMODAL_YES` | unset | Skip the spend prompt |
| `RUNMODAL_ALLOW_CONCURRENT` | unset | Launch even when another nanochat run is live (see [One command](#one-command)) |
| `NANOCHAT_MODAL_ARCHIVE` | `~/.cache/nanochat-modal/runs` | Where local run logs are kept |
| `FORCE_STAGES` | unset | Redo named stages, e.g. `FORCE_STAGES=base_train,base_eval` |
| `WANDB_RUN` | `dummy` | Weights & Biases run name; `dummy` disables logging |

The runner sets the rest (`SKIP_SETUP`, `NANOCHAT_BASE_DIR`, `NANOCHAT_RUNCPU_*`, `NANOCHAT_POST_STAGE_HOOK`). You should not need to change them.

## Cost and time

**CPU mode (`MODE=cpu`):** cost scales with cores. Check Modal's current CPU price with `modal billing rates` before the first run. Time and cost are not measured yet; the first run records both in its run record.

**GPU mode (`MODE=gpu`):**
- **Cost:** ~$40–60 for a full run (~$3/GPU/hr × 8 GPUs × ~1.5–2h training)
  - First run is higher: Modal builds a Docker image and compiles PyTorch, adding ~0.5–1h and proportional cost
  - Later runs reuse the image and cost less
- **Wall-clock time:** ~1.5–2h training, plus ~30 min prefetch (CPU) and ~15 min image build (first run only)

For both modes:
  - Keep your laptop awake; the job is attached to your terminal
  - Use `screen` or `tmux` if you expect network drops (see [RUNNING.md](RUNNING.md) for tmux tips)

The script shows cost and time estimates before charging anything and asks for confirmation.

## One command

**Pre-flight check — run this first:**

```bash
modal app list
```

Confirm all `nanochat-*` apps show a non-null `stopped_at`. The run guard refuses to launch if another nanochat run is live (to avoid accidental double-billing). If you see a `nanochat-*` app with `stopped_at` empty, stop it first:

```bash
modal app stop <app-id>
```

Then launch:

```bash
bash runs/runmodal.sh                 # CPU mode (default)
MODE=gpu bash runs/runmodal.sh        # GPU mode
```

The script:
1. Checks that `modal` is installed and configured
2. Shows a mode banner with cost and time estimates and asks `Proceed? [y/N]`
3. Runs the pipeline on Modal:
   - **Prefetch** (CPU on Modal): downloads the training data shards to a Modal Volume (8 shards in CPU mode, 170 in GPU mode)
   - **Train**: `runcpu.sh` on the CPU machine (CPU mode), or `speedrun.sh` on 8×H100 (GPU mode)
   - **Download**: pulls the trained tokenizer and SFT checkpoints back to your laptop under `~/.cache/nanochat-modal`
   - **Print local chat tip**: shows the command to chat with your model locally

For scripted/automated runs, skip the confirmation with:
```bash
RUNMODAL_YES=1 bash runs/runmodal.sh
```

Each run gets an id (`<UTC time>-<mode>`). Its archive and run record are on the Volume under `runs/<run-id>/`. To check a run from the terminal without changing it, use `bash runs/modal_peek.sh` (commands are listed in the script's header: `status`, `apps`, `past`, `report`, `compare`, `pull`, `info`, `logs`, `container`, `cost`).

## Watching the run

Keep your terminal visible (or use `tmux attach`). Key signals to look for in the log:

| Signal | What it means |
|--------|--------------|
| `NANOCHAT_BASE_DIR=/data/nanochat-modal` | Volume path is set correctly in the container |
| `OMP_NUM_THREADS=<N>` | Actual core count the container received; confirms the CPU spec was accepted |
| Per-step loss lines from `base_train` | Training is progressing; note the step/sec rate for PROGRESS.md |
| `.train_ok` written | Base train finished and saved its marker; eval stage can start |
| `SKIP` on a stage line | That stage's marker already exists; re-run is picking up where it left off |
| `modal volume get …` progress bar | Download of tokenizer and SFT checkpoint to your laptop underway |

**If the machine spec is rejected** (Modal returns a resource error for 64 cores / 128 GiB), lower `cpu=64` and `memory=131072` in `runs/modal_run.py` and re-run.

**If the run exceeds the timeout** and is killed mid-stage, re-run as normal — finished stages SKIP and only the interrupted stage reruns. Mid-stage resume is implemented for `base_train` and `chat_sft`, but its Modal path is not verified, so treat the stage as restarting from its beginning.

After the run completes, record the actual core count, wall-clock time per stage, and total cost in `docs/MODAL.md` (the "Cost and time" section above) and in `docs/plan/PROGRESS.md`.

## What happens inside

The job is split into stages, each marked in a persistent Modal Volume so re-runs can skip finished work. The table shows the GPU profile; the CPU profile runs the same stages with smaller settings.

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
- **Killed mid-stage** — that stage reruns on the next call (mid-stage resume is not verified on Modal; treat that stage as restarting from its beginning)
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
| `speedrun.sh` | Full 8×H100 training (the MVP) | **Yes** — `MODE=gpu` |
| `runcpu.sh` | Small demo (depth 6) on a CPU machine | **Yes** — `MODE=cpu` (default); also runs locally on a laptop |
| `miniseries.sh` | Research sweep over multiple depths | Later (not in MVP) |
| `scaling_laws.sh` | FLOPs×depth grid | Later (not in MVP) |

`runmodal.sh` picks the script from `MODE`. Modal's CPU mode runs on a larger machine than a laptop. Its speed against the laptop has not been measured yet. The model is still small and is not a GPT-2-grade model.

## How Modal optionality works

When you run locally (without Modal env vars set):
- `bash runs/runcpu.sh` — uses `~/.cache/nanochat`, full CPU/MPS bootstrap, no Modal
- `bash runs/speedrun.sh` — uses `~/.cache/nanochat`, full GPU bootstrap, no Modal

When Modal runs `runcpu.sh` or `speedrun.sh` internally:
- Modal sets env vars (`NANOCHAT_BASE_DIR`, `SKIP_SETUP`, `NANOCHAT_POST_STAGE_HOOK`, and `NANOCHAT_RUN_ID` / `NANOCHAT_RUN_DIR` for the run record)
- Same script reuses the volume path and skips venv setup (the image has deps pre-installed)
- CPU-only tuning (`NANOCHAT_RUNCPU_*`) applies to CPU mode and does not reach the GPU job
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
