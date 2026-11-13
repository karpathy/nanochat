# Running nanochat scripts (Mac / CPU fix-forward guide)

Level 2 guide. The short version lives in the [README](../README.md#running-on-cpu--mps). Install and `uv sync` steps are in the README [Setup](../README.md#setup) section and are not repeated here.

Design source: [docs/plan/2026-11-13-fix-forward-mac.md](plan/2026-11-13-fix-forward-mac.md).

**Contents**
[1 Pick a script](#1-pick-a-script) · [2 First run](#2-first-run-expectations) · [3 Fix-forward loop](#3-the-fix-forward-loop) · [4 Re-run map](#4-what-a-re-run-keeps-and-skips) · [5 Echo legend](#5-reading-the-logs-echo-legend) · [6 Glossary](#6-enrichment-glossary) · [7 Mac / Intel torch](#7-mac--intel-torch) · [8 NaN](#8-nan-troubleshooting) · [9 Controls](#9-controls) · [10 Failure shapes](#10-common-failure-shapes) · [11 Phase 2 stubs](#11-coming-in-phase-2)

---

## 1. Pick a script

- **`bash runs/runcpu.sh`** — small demo (depth 6) for Mac / CPU / MPS; use this on a laptop.
- **`bash runs/speedrun.sh`** — the GPT-2-grade run, built for a blank 8xH100 CUDA node; it warns (but continues) if no CUDA is found.
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
| `base_train` | any `base_checkpoints/**/model_*.pt` | `scripts/base_train.py` checkpoint saves |
| `base_eval` | `base_checkpoints/.eval_ok` | end of `scripts/base_eval.py` |
| `chat_sft` | any `chatsft_checkpoints/**/model_*.pt` | `scripts/chat_sft.py` checkpoint saves |
| `chat_eval` (speedrun only) | `chatsft_checkpoints/.eval_ok` | end of `scripts/chat_eval.py` |

Notes:

- Markers are written **last**, after a clean finish. A crashed eval leaves no marker, so it re-runs.
- `base_train` skips as soon as *any* `model_*.pt` exists, including a partial one from an interrupted run. To retrain, force it.
- `nanochat.dataset -n 8` runs every time; it only downloads missing shards.
- Per-stage timings are saved in `$NANOCHAT_BASE_DIR/.stage_elapsed/` so a later `SKIP` can report time saved.

**To redo stages on purpose**, use `FORCE_STAGES` (see [Controls](#9-controls)):

```bash
FORCE_STAGES=base_train,base_eval bash runs/runcpu.sh   # comma-list
FORCE_STAGES=all bash runs/runcpu.sh                    # everything
```

Never use `FORCE_STAGES=1`. The scripts reject it (exit 2) and list the valid names: `tok_train,tok_eval,base_train,base_eval,chat_sft,chat_eval`.

## 5. Reading the logs (echo legend)

Shell lines start with `[HH:MM:SS]`. This is the frozen echo contract from the plan.

| Kind | Shape | Meaning |
|------|-------|---------|
| START | `[HH:MM:SS] START stage=base_train profile=runcpu depth=6 device_batch_size=32 ETA: not yet knowable` | Stage begins. Profile and key knobs are shown. |
| DONE | `[HH:MM:SS] DONE stage=base_train elapsed=71m cumulative=2h45m` | Stage finished; time for this stage and for the whole script run so far. |
| SKIP | `[HH:MM:SS] SKIP stage=tok_train reason=artifact exists path=... (FORCE_STAGES=tok_train to redo)` | Artifact found; nothing ran. |
| FORCE | `[HH:MM:SS] FORCE stage=tok_train reason=FORCE_STAGES=tok_train` | Artifact exists but you asked to redo it. |
| FAIL | `[HH:MM:SS] FAIL stage=base_train elapsed=… hint=…` | Stage exited non-zero; the script stops here. |
| rms_norm | `[norm] original=F.rms_norm → adjusted=manual_rsqrt (reason: missing on torch 2.2.2)` | Fallback in use. See [section 7](#7-mac--intel-torch). |
| auto-fit | `[auto-fit] device_batch_size: original/default=8 → adjusted=2 (reason: …)` | **Phase 2** — not emitted yet. See [section 11](#11-coming-in-phase-2). |

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
| `purpose:` | After `START` | One plain sentence on why the stage exists. |
| `next:` | After `DONE` | What the script runs next (or that it will skip it). |
| `skip-saved:` | After `SKIP` | Time saved, from the last recorded run of that stage; `unknown` if there is no timing yet. |
| `root_stage=` | After `FAIL` | The stage that actually broke. `cascade_suppressed=yes` means later stages were not run. |
| `device:` line | Python startup | Torch version, device and world size, plus the base dir, e.g. `torch 2.2.2 \| device: mps (…) \| world size: 1`. |
| `wrote:` | After a checkpoint `DONE` | Planned: artifact path and size on disk. |
| `heartbeat:` | During training | Planned: optional one-line progress summary. |

## 7. Mac / Intel torch

<a id="rms-norm"></a>

- The README [torch table](../README.md#setup) pins **torch 2.2.2** on macOS Intel (`x86_64`), the last version available there. Apple Silicon gets 2.9.1.
- `F.rms_norm` does not exist on torch 2.2.2. `nanochat/gpt.py` now picks the norm once at import time: native `F.rms_norm` if present, otherwise a manual `rsqrt` version. The log says which:
  - `[norm] using native F.rms_norm`
  - `[norm] original=F.rms_norm → adjusted=manual_rsqrt (reason: missing on torch 2.2.2)`
- The fallback is covered by `tests/test_norm.py` (eager mode and `torch.compile`). Check it yourself with `pytest tests/test_norm.py`.
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

| Variable | Effect | Example |
|----------|--------|---------|
| `FORCE_STAGES` | Redo stages even if artifacts exist. `all` or a comma-list of stage names. Never `=1`. | `FORCE_STAGES=tok_train,base_train bash runs/runcpu.sh` |
| `NANOCHAT_BASE_DIR` | Where artifacts live for the Python scripts and checkpoint lookups. Default `~/.cache/nanochat`. | `NANOCHAT_BASE_DIR=/tmp/nc python -m scripts.chat_cli -p "hi"` |
| `WANDB_RUN` | Weights & Biases run name. Unset means `dummy` (no logging). | `WANDB_RUN=mytest bash runs/runcpu.sh` |

Caveat: `runcpu.sh` and `speedrun.sh` set `NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"` themselves, so setting it before them has no effect. It is for running individual Python scripts by hand. If you change the dir by hand, keep it the same across all stages.

## 10. Common failure shapes

| You see | Usually means | Do this |
|---------|---------------|---------|
| `Checkpoint directory not found: …/base_checkpoints` (with `Expected source: 'base'`) | `base_train` never finished, or you're pointing at a different base dir | Re-run the script; it resumes at the failing stage. Check `NANOCHAT_BASE_DIR`. |
| Same error for `chatsft_checkpoints` (`'sft'`) | `chat_sft` never finished | Re-run; fix the `FAIL` above it first. |
| `ERROR: FORCE_STAGES=… has unknown stage` | Typo, or `FORCE_STAGES=1` | Use `all` or names from the list in [section 4](#4-what-a-re-run-keeps-and-skips). |
| `FAIL stage=base_train … exit code 1` right after startup on Intel Mac | torch 2.2.2 issue | Check for the `[norm]` line, see [section 7](#7-mac--intel-torch). |
| `ERROR: 5 consecutive NaN losses` | Numerical blow-up | See [section 8](#8-nan-troubleshooting). |
| `WARN no CUDA device detected` (speedrun) | Wrong script for this machine | Use `runs/runcpu.sh`. |

The missing-checkpoint error is almost always a **symptom**: scroll up to the first `FAIL`, because the checkpoint was never produced.

## 11. Coming in Phase 2

Stubs only, so you know what is planned. None of this is in the current scripts.

- **Memory / OOM** — an auto-fit step that shrinks `device_batch_size` on MPS so it fits in memory, prints `[auto-fit] original → adjusted`, and retries once after an out-of-memory error. See plan Phase 2, section 2.1–2.2.
- **Defaults / profiles** — one canonical source of defaults (`nanochat/defaults.py`) feeding both Python and shell, with a `runcpu` profile. Today the `runcpu.sh` flags are written directly in the script. See plan Phase 2, section 2.0.

Until then, if you hit an out-of-memory error on MPS, lower `--device-batch-size` in `runs/runcpu.sh` and re-run; finished stages skip.
