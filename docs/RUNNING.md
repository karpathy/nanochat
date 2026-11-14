# Running nanochat scripts (Mac / CPU fix-forward guide)

Level 2 guide. The short version lives in the [README](../README.md#running-on-cpu--mps). Install and `uv sync` steps are in the README [Setup](../README.md#setup) section and are not repeated here.

Design source: [docs/plan/2026-11-13-fix-forward-mac.md](plan/2026-11-13-fix-forward-mac.md).

**Contents**
[1 Pick a script](#1-pick-a-script) · [2 First run](#2-first-run-expectations) · [3 Fix-forward loop](#3-the-fix-forward-loop) · [4 Re-run map](#4-what-a-re-run-keeps-and-skips) · [5 Echo legend](#5-reading-the-logs-echo-legend) · [6 Glossary](#6-enrichment-glossary) · [7 Mac / Intel torch](#7-mac--intel-torch) · [8 NaN](#8-nan-troubleshooting) · [9 Controls](#9-controls) · [10 Failure shapes](#10-common-failure-shapes) · [11 Memory / OOM](#11-memory--oom) · [12 Defaults and profiles](#12-defaults-and-profiles)

How-to first (§1–4, §8–10), then reference (§5–6, §9). Kept in one L2 file on purpose so a stuck runner can stay on one page (plan doc mapping); use the TOC to jump.

How-to first (§1–4, §8–10), then reference (§5–6, §9). Kept in one L2 file on purpose so a stuck runner can stay on one page (plan doc mapping); use the TOC to jump.

---

## 1. Pick a script

- **`bash runs/runcpu.sh`** — small demo (depth 6) for Mac / CPU / MPS; use this on a laptop.
- **`bash runs/speedrun.sh`** — the GPT-2-grade run, built for a blank 8xH100 CUDA node; **exits with ERROR** if no CUDA is found (use `runcpu.sh` on a laptop — a Mac speedrun can thrash memory hard enough to reboot).
- **`runs/miniseries.sh` / `runs/scaling_laws.sh`** — multi-run sweeps for researchers; they are not part of the fix-forward flow described here.

## 2. First run expectations

- Run from the repo root. The script creates `.venv` and runs `uv sync` if needed.
- It prints a `pipeline:` map first, then one `START` … `DONE` block per stage.
- `runcpu.sh` was tuned to take roughly an hour on a MacBook Pro M3 Max (tokenizer ~minutes, base training ~30 min, SFT ~10 min). Slower machines take longer. Results are a demo, not a useful chatbot.
- ETA reads `not yet knowable` at `START`. That is deliberate: nothing is estimated before the first train step.
- Everything is written under `~/.cache/nanochat` (override: [`NANOCHAT_BASE_DIR`](#9-controls)).

## 3. The fix-forward loop

The scripts use `set -euo pipefail` and stop at the **first** failing stage. You never need to read past it.

```text
run → first FAIL → fix → re-run the same command → repeat
```

1. Run `bash runs/runcpu.sh`.
2. Find the first `FAIL stage=...` line. The `root_stage=` line under it names the stage that broke; later stages are not run, so there is no cascade to wade through.
3. Fix the cause (see [common failure shapes](#10-common-failure-shapes)).
4. Run the exact same command again. Finished stages print `SKIP` and cost nothing.
5. Repeat until `ALL DONE`.

## 4. What a re-run keeps and skips

A stage is skipped when its artifact already exists. Paths are relative to `$NANOCHAT_BASE_DIR` (default `~/.cache/nanochat`).

| Stage | Skipped when this exists | Written by |
|-------|--------------------------|------------|
| `tok_train` | `tokenizer/tokenizer.pkl` | `scripts/tok_train.py` |
| `tok_eval` | `tokenizer/.tok_eval_ok` | end of `scripts/tok_eval.py` (and touched by the shell on success) |
| `base_train` | `base_checkpoints/.train_ok` | end of `scripts/base_train.py` (rank 0) |
| `base_eval` | `base_checkpoints/.eval_ok` | end of `scripts/base_eval.py` |
| `chat_sft` | `chatsft_checkpoints/.train_ok` | end of `scripts/chat_sft.py` (rank 0) |
| `chat_eval` (speedrun only) | `chatsft_checkpoints/.eval_ok` | end of `scripts/chat_eval.py` |

Notes:

- Markers are written **last**, after a clean finish. A crashed train/eval leaves no marker (or the shell clears it at the next `START`), so the stage re-runs. A stray `model_*.pt` from a probe or interrupt is **not** enough to SKIP.
- Upgrading from older skip-on-`model_*.pt` behavior: if you already finished train and want to keep SKIP without redoing it, `touch "$NANOCHAT_BASE_DIR/base_checkpoints/.train_ok"` (and the chatsft equivalent if needed).
- Mid-train resume is still manual (`--resume-from-step=N` on `scripts/base_train.py`); the shell pipeline resumes at the **stage** boundary only.
- `nanochat.dataset -n 8` runs every time; it only downloads missing shards.
- Per-stage timings are saved in `$NANOCHAT_BASE_DIR/.stage_elapsed/` so a later `SKIP` can report time saved.

**To redo stages on purpose**, use `FORCE_STAGES` (see [Controls](#9-controls)):

```bash
FORCE_STAGES=base_train,base_eval bash runs/runcpu.sh   # comma-list
FORCE_STAGES=all bash runs/runcpu.sh                    # everything
```

Never use `FORCE_STAGES=1`. The scripts reject it (exit 2) and list the valid names: `tok_train,tok_eval,base_train,base_eval,chat_sft,chat_eval`.

## 5. Reading the logs (echo legend)

Shell lines start with `[HH:MM:SS]`. This is the frozen echo contract from the plan. Extra lines like `pipeline:`, `purpose:`, `health:`, or `tip:` are enrichments (see [section 6](#6-enrichment-glossary)), not contract shapes — do not treat them as START/DONE/SKIP/FAIL.

| Kind | Shape | Meaning |
|------|-------|---------|
| START | `[HH:MM:SS] START stage=base_train profile=runcpu depth=6 device_batch_size=8 ETA: not yet knowable` | Stage begins. Profile and key knobs are shown. |
| DONE | `[HH:MM:SS] DONE stage=base_train elapsed=71m cumulative=2h45m` | Stage finished; time for this stage and for the whole script run so far. |
| SKIP | `[HH:MM:SS] SKIP stage=tok_train reason=artifact exists path=... (FORCE_STAGES=tok_train to redo)` | Artifact found; nothing ran. |
| FORCE | `[HH:MM:SS] FORCE stage=tok_train reason=FORCE_STAGES=tok_train` | Artifact exists but you asked to redo it. |
| FAIL | `[HH:MM:SS] FAIL stage=base_train elapsed=… hint=…` | Stage exited non-zero; the script stops here. |
| rms_norm | `[norm] original=F.rms_norm → adjusted=manual_rsqrt (reason: missing on torch 2.2.2)` | Fallback in use. See [section 7](#7-mac--intel-torch). |
| compile | `[compile] original=torch.compile → adjusted=eager (reason: Inductor has no MPS backend)` | Model stays eager on MPS (Inductor unsupported). CUDA/CPU still compile. See [section 7](#7-mac--intel-torch). |
| auto-fit | `[auto-fit] device_batch_size: original/default=8 → adjusted=2 (reason: est_peak=12.4GiB > margin*avail=0.6*18.0GiB)` | Batch shrunk so the estimate fits under the memory margin. See [section 11](#11-memory--oom). |
| unchanged | `[auto-fit] device_batch_size=8 (unchanged; estimate fits)` | Auto-fit ran; requested batch already fits. |

**ETA rules** (one of three, never invented):

| Form | When |
|------|------|
| `ETA: not yet knowable` | At `START`. Becomes knowable only after the first train step. |
| `ETA: ~Xm remaining (step i/N, avg dt=…)` | Planned for the train loop once timing exists. |
| `ETA: n/a (skipped)` | On `SKIP` lines. |

**Emission order** within a stage: shell `START` → Python startup (torch version, device, paths) → fallback/auto-fit lines if any → shell `DONE` / `SKIP` / `FAIL` / `FORCE`.

Rule of thumb: no knob is ever changed silently. Any change is printed as `original → adjusted` with a reason.

## 6. Enrichment glossary

One extra line at most per event. These are hints, not part of the strict contract.

| Term | Where | What it tells you |
|------|-------|-------------------|
| `pipeline:` map | Script start | Journey at a glance: `✓` done, `▶` next to run, `○` later. Example: `pipeline: [tok✓] [base▶] [eval○] [sft○]` |
| `changed:` / `same:` | Script start (after `pipeline:`) | Knob diff vs `$NANOCHAT_BASE_DIR/last_run.json` from the last successful full run. Skipped silently on first run (no file yet). Example: `changed: device_batch_size 32→8; same: depth=6,profile=runcpu`. |
| `tip:` | Script start (once per invocation) | Exactly one educational tip for the whole `runcpu.sh` / `speedrun.sh` run — not per stage. Mentions a real knob (e.g. `--device-batch-size` / grad accumulation). |
| `purpose:` · `health:` | After `START` (one packed line) | Why the stage exists, then health (mem · ckpt · shards). Packed together so START still has only one enrichment line. Example: `purpose: teach raw next-token prediction · health: ~12.4GiB free-ish · ckpt=1.2GiB · shards=8/8`. Best-effort; never fails the run. |
| `next:` | After `DONE` | What the script runs next (or that it will skip it). Packed on one line with `wrote:` / `mem:` when those apply. |
| `skip-saved:` | After `SKIP` | Time saved, from the last recorded run of that stage; `unknown` if there is no timing yet. |
| `root_stage=` | After `FAIL` | The stage that actually broke. `cascade_suppressed=yes` means later stages were not run. |
| `device:` line | Python startup | Torch version, device and world size, plus the base dir, e.g. `torch 2.2.2 \| device: mps (…) \| world size: 1`. |
| `wrote:` | After `tok_train` / `base_train` / `chat_sft` `DONE` | Artifact path and approximate disk size, e.g. `wrote: ~/.cache/nanochat/base_checkpoints/d6 (1.2GiB)`. |
| `mem:` | After `DONE` | Best-effort free/available memory snapshot (macOS `vm_stat` or Linux `MemAvailable`). |
| `memory:` | After auto-fit | Plain-language fit, e.g. `memory: aiming to use ≤60% of ~18GiB; batch 8→2 so logits fit`. Complements the `[auto-fit]` contract line; does not replace it. |
| `trend:` | After a new `val_bpb` / CORE metric | Compares this run’s metric to the prior value in `$NANOCHAT_BASE_DIR/last_metrics.json` (global last-seen keys — mid-train eval can update before `base_eval`). Printed only when a prior exists. Fail-soft: never aborts train/eval. Example: `trend: val_bpb 7.47 → better than prior 8.1`. |
| `val bpb:` progress | During val eval on non-CUDA | Sparse cadence (~10 lines) so Mac/CPU do not look hung; off by default on CUDA. |
| `heartbeat:` | During training | Planned: optional one-line progress summary (not every-step pre-announce). |

## 7. Mac / Intel torch

<a id="rms-norm"></a>

- The README [torch table](../README.md#setup) pins **torch 2.2.2** on macOS Intel (`x86_64`), the last version available there. Apple Silicon gets 2.9.1.
- `F.rms_norm` does not exist on torch 2.2.2. `nanochat/gpt.py` now picks the norm once at import time: native `F.rms_norm` if present, otherwise a manual `rsqrt` version. The log says which:
  - `[norm] using native F.rms_norm`
  - `[norm] original=F.rms_norm → adjusted=manual_rsqrt (reason: missing on torch 2.2.2)`
- The fallback is covered by `tests/test_norm.py` (eager mode and `torch.compile`). Check it yourself with `pytest tests/test_norm.py`.
- **`torch.compile` / Inductor on MPS:** TorchInductor has no MPS backend (`AssertionError: Device mps not supported`). On `device_type=mps`, training keeps the model and optimizer steps in **eager** mode and prints `[compile] original=torch.compile → adjusted=eager …`. CUDA/CPU still compile. Covered by `tests/test_compile_device.py`.
- **Why a laptop + `speedrun.sh` looked like many failures:** `speedrun.sh` targets 8xH100. On a laptop, base training could not run because of the missing `rms_norm` (and the GPU-only setup), and the old script had no `set -e`, so it kept going into eval and SFT, each of which failed again on the missing checkpoint. One root cause produced a wall of errors. Now the script stops at the first `FAIL`, and the checkpoint error points back to the prior stage. On a Mac, use `runs/runcpu.sh`.

## 8. NaN troubleshooting

In `base_train`:

- **First NaN loss** prints a warning: `WARNING: loss is NaN at step N (numerical instability, NOT a memory problem) …`. Training continues.
- **5 consecutive NaN steps** abort the run with a non-zero exit, which the script reports as `FAIL stage=base_train`. A good step resets the counter.
- NaN means the numbers blew up (learning rate, dtype, data), **not** that you ran out of memory. Lowering batch size is not the fix.

What to try, cheapest first:

1. Note the step of the first warning in the log; it tells you how early the run went bad.
2. Force fp32: `NANOCHAT_DTYPE=float32` (the default on CPU/MPS; see README [Precision](../README.md#precision--dtype)).
3. Lower the learning rate flags in `scripts/base_train.py`, then force a fresh start: `FORCE_STAGES=base_train`.

Root-causing NaN beyond warn/abort is out of scope for Phase 1.

## 9. Controls

| Variable | Effect | Notes |
|----------|--------|-------|
| `FORCE_STAGES` | Redo stages even if artifacts exist. `all` or a comma-list of stage names. Never `=1`. | Example: `FORCE_STAGES=tok_train,base_train bash runs/runcpu.sh` |
| `NANOCHAT_BASE_DIR` | Where artifacts live for the Python scripts and checkpoint lookups. Default `~/.cache/nanochat`. | Can be pre-set before the script. Modal sets this to `/data/nanochat-modal` internally; see [docs/MODAL.md](MODAL.md) for cloud training. Example: `NANOCHAT_BASE_DIR=/tmp/nc python -m scripts.chat_cli -p "hi"` |
| `SKIP_SETUP` | When set to `1`, skip `uv install`/venv create/`uv sync`. Always `source .venv/bin/activate` (the venv must exist). | Used by Modal (image provides `.venv`); also useful for CI or reproducible runs. Unset = full bootstrap (default local behavior). |
| `NANOCHAT_POST_STAGE_HOOK` | Bash script/command run after each successful stage. No-op when unset (local runs unchanged). | Used by Modal to coordinate Volume commits via wait-for-ack. Example: `NANOCHAT_POST_STAGE_HOOK="bash /path/to/hook.sh" bash runs/speedrun.sh`. The hook must be an executable path or a simple command (not a bare string with quotes). |
| `NANOCHAT_MEMORY_FIT_MARGIN` | Fraction of available device memory the auto-fit estimate may use (default `0.6`). See [section 11](#11-memory--oom). | Example: `NANOCHAT_MEMORY_FIT_MARGIN=0.5 bash runs/runcpu.sh` |
| `WANDB_RUN` | Weights & Biases run name. Unset means `dummy` (no logging). | Example: `WANDB_RUN=mytest bash runs/runcpu.sh` |

**Key invariants:**

- `runcpu.sh` and `speedrun.sh` set `NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"` themselves when unset, so pre-setting has no effect on those scripts directly. Pre-set it for running individual Python scripts by hand. If you change the dir by hand, keep it the same across all stages.
- `SKIP_SETUP=1` always activates `.venv` (fails loudly if missing). The Modal image provides `.venv` with GPU deps pre-installed; local GPU training with `SKIP_SETUP=1` requires you to build `.venv` first.
- `NANOCHAT_POST_STAGE_HOOK` is a no-op when unset (existing local runs are unchanged). Modal uses this internally for Volume persistence.
- **Modal training:** see [docs/MODAL.md](MODAL.md) for the full cloud training guide, including how these variables interact on remote jobs.

## 10. Common failure shapes

| You see | Usually means | Do this |
|---------|---------------|---------|
| `Checkpoint directory not found: …/base_checkpoints` (with `Expected source: 'base'`) | `base_train` never finished, or you're pointing at a different base dir | Re-run the script; it resumes at the failing stage. Check `NANOCHAT_BASE_DIR`. |
| Same error for `chatsft_checkpoints` (`'sft'`) | `chat_sft` never finished | Re-run; fix the `FAIL` above it first. |
| `ERROR: FORCE_STAGES=… has unknown stage` | Typo, or `FORCE_STAGES=1` | Use `all` or names from the list in [section 4](#4-what-a-re-run-keeps-and-skips). |
| `FAIL stage=base_train … exit code 1` right after startup on Intel Mac | torch 2.2.2 issue | Check for the `[norm]` line, see [section 7](#7-mac--intel-torch). |
| `ERROR: 5 consecutive NaN losses` | Numerical blow-up | See [section 8](#8-nan-troubleshooting). |
| Out-of-memory / MPS allocator failure during train | Batch too large for device RAM | Check for an `[auto-fit]` line; see [section 11](#11-memory--oom). Do not treat this as NaN. |
| `Device mps not supported` / `BackendCompilerFailed` (Inductor) | `torch.compile` ran on MPS (old build) | Re-run on a build that skips compile on MPS ([section 7](#7-mac--intel-torch)); you should see `[compile] … adjusted=eager`. |
| `ERROR no CUDA device detected` (speedrun, exit 1) | Wrong script for this machine | Use `bash runs/runcpu.sh`. Do not override; speedrun is 8xH100-only. |
| Train stage re-runs despite `model_*.pt` on disk | No `.train_ok` marker (partial/probe/interrupted run, or pre-marker upgrade) | Let it retrain, or `touch …/base_checkpoints/.train_ok` only if you know train finished cleanly. |
| `AssertionError` / crash mentioning rotary length (e.g. `2596 > 2560`) | Few-shot CORE prompts exceeded the old static RoPE cache | Current builds grow the RoPE cache on demand (2× buffer). Re-run; no knob change needed. |

The missing-checkpoint error is almost always a **symptom**: scroll up to the first `FAIL`, because the checkpoint was never produced.

## 11. Memory / OOM

Think of training memory like packing a suitcase: most of the bulk is the **logits** tensor — shape roughly `B × T × V` (batch × sequence length × vocab). Weights and optimizer state matter too, but when you OOM on a laptop it is usually because `B` was too big for that product. Auto-fit (below) is the current Mac mitigator; a deeper softcap/logits redesign is parked backlog, not in this slice.

**What auto-fit does**

1. Before heavy allocation, estimate peak use (params + optim + logits, with a documented fudge factor of **1.5**).
2. If the estimate exceeds `margin × available` memory, **halve** `device_batch_size` until it fits (floor **1**).
3. Print an `[auto-fit]` contract line (`original → adjusted`, or unchanged), then a plain-language `memory:` enrichment on every fit (including when the batch did not change).
4. If **`base_train`** still OOMs after that fit: empty the cache, **halve once more**, fall back to the **eager** (uncompiled) model, and **retry that step once**. If it still OOMs after the retry, the stage `FAIL`s with a clear message (batch size + `RUNNING.md` pointer) — not only when the batch is already 1. (`base_eval` / `chat_sft` do preflight auto-fit only — they do not re-halve mid-run.)

**Knobs**

| Knob | Default | Notes |
|------|---------|-------|
| `--auto-fit-batch` | **True on `mps`**, **False on `cuda`** | CUDA/speedrun stays hands-off unless you opt in. CPU skips auto-fit (or treats capacity as effectively unlimited). |
| `NANOCHAT_MEMORY_FIT_MARGIN` | `0.6` | Aim to use ≤60% of available device memory so the OS and fragmentation have headroom. |

Gradient accumulation keeps the *total* batch size the same when the micro-batch shrinks — you just take more steps per update.

**NaN is not a memory problem.** A `loss: nan` warning means the numbers blew up (learning rate, dtype, data). Lowering batch size will not fix it. See [section 8](#8-nan-troubleshooting).

## 12. Defaults and profiles

One place to turn the dials — no hunting through shell scripts and argparse for the same number.

| Layer | File | Role |
|-------|------|------|
| Canonical | [`nanochat/defaults.py`](../nanochat/defaults.py) | Shared constants (`VOCAB_SIZE`, `LOGIT_SOFTCAP`, `MEMORY_FIT_MARGIN`, …) and profiles `PROFILE_RUNCPU` / `PROFILE_SPEEDRUN`. Python wins on conflict. |
| Emitted shell | [`runs/profiles.sh`](../runs/profiles.sh) | Generated/emitted from Python for `runcpu.sh` / `speedrun.sh` to source. Do not hand-edit as a second source of truth. |

**Batch sizes (easy to mix up)**

| Source | `device_batch_size` | Why |
|--------|---------------------|-----|
| CLI / argparse default | **32** | Sensible for a big GPU when you call `python -m scripts.base_train` with no flags. |
| `PROFILE_RUNCPU` | **8** | Safer educational default on a laptop; auto-fit may shrink further. |
| `PROFILE_SPEEDRUN` | script/profile value (e.g. 16 on 8xH100) | CUDA speedrun; auto-fit off unless you pass `--auto-fit-batch`. |

**How to change a run**

- **Educational / Mac / CPU** — edit `PROFILE_RUNCPU` in `nanochat/defaults.py`, re-emit `runs/profiles.sh` if your workflow checks that file in, then re-run `bash runs/runcpu.sh` (use `FORCE_STAGES=…` if you need to redo a finished stage).
- **GPU / speedrun** — edit `PROFILE_SPEEDRUN` the same way, or pass flags on the `torchrun` line; leave `--auto-fit-batch` off unless you want MPS-style fitting on CUDA.

Sync is checked by `tests/test_defaults.py` (emitted shell vars must match `PROFILE_*`). Memory math is covered by `tests/test_memory.py`.
