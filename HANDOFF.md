# HANDOFF — modal-nanochat-minimum

> **STALE (2026-11-15).** This handoff is from before PR #14–#45. The current backlog is `docs/plan/TRAIL.md`; the current plan is `docs/plan/2026-11-15-eval-resume-and-tracker-cleanup.md`. Do not follow the steps below without checking TRAIL first.

Branch: `feat/modal-nanochat-minimum` (pushed, head `3c6ba61`)
Plan (SSOT): `docs/plan/2026-11-14-modal-nanochat-minimum.md` (§1a = CPU phase)
Tracker: `docs/plan/PROGRESS.md` and `docs/plan/TRAIL.md`

## Phase goal (current): full pipeline on Modal CPU

GPU is blocked: Modal needs a payment method before any H100 function starts. This phase runs the whole `runcpu` pipeline (tokenizer → base train → base eval → SFT → chat) on a heavy CPU machine. GPU comes later.

## What is done

- `9eed864` image fix: `.env()` before the local-source layer (Modal rejects build steps after local files).
- `31974eb` `runs/runcpu.sh` honours `NANOCHAT_BASE_DIR` and `SKIP_SETUP`. Unset = local behaviour.
- `3c6ba61` Modal CPU mode:
  - `runs/modal_run.py --mode cpu|gpu` (default cpu). CPU: `prefetch_cpu` (8 shards) + `train_cpu` (`runcpu.sh`, 64 cores, 128 GiB, `OMP_NUM_THREADS` = cores the container gets).
  - `runs/runmodal.sh` takes `MODE=cpu|gpu`, prints a mode banner, then runs `modal run runs/modal_run.py --mode $MODE`.
  - `tests/test_runmodal_mode.py` (3 tests, fake `modal` on PATH, no Modal calls).
- Local checks: 143 passed / 18 skipped. Ruff clean. `bash -n` and `py_compile` clean. `modal run runs/modal_run.py --help` loads and shows `--mode`. `uv export --extra cpu` resolves Linux torch to `2.9.1+cpu`.
- Plan updated (§1a, goals, FAQ, verification 16–20). Earlier history: `PROGRESS.md` says "MERGED — live run pending".

## Not verified yet (needs one billable run)

- The CPU image builds on Modal (`uv sync --extra cpu --frozen` remote).
- Real core cap for `cpu=64` and `memory=131072` on this account. Modal may reject it. If so, lower and re-run.
- Per-step time and total time of `runcpu` on that machine (plan §1a step 5). If total > 3 h, the 3 h timeout kills it.
- `modal volume get` pulls the chat checkpoint; chat tip gives a non-empty answer.
- `modal app stop` ends billing (Ctrl-C only stops the local client).

## Next steps, in order

1. **Get explicit go-ahead for a CPU run.** Cost scales with cores; check Modal's current CPU price first. Nothing billable runs without it.
2. **Run:** `RUNMODAL_YES=1 MODE=cpu bash runs/runmodal.sh` (or drop `RUNMODAL_YES` to confirm interactively). Keep the laptop awake; use screen/tmux.
3. **Watch the log for:** `NANOCHAT_BASE_DIR=/data/nanochat-modal`, `OMP_NUM_THREADS` set to the core count, per-step timing from `base_train`, and `.train_ok` on the Volume after base train.
4. **If the machine is rejected or too slow:** lower `CPU_CORES` / `CPU_MEMORY_MIB` / timeout in `runs/modal_run.py`, re-run.
5. **Check billing stops:** `modal app list`, then `modal app stop <app>` after any early stop. Record the result in `docs/MODAL.md`.
6. **Chat tip on the Mac:** `NANOCHAT_BASE_DIR=~/.cache/nanochat-modal python3 -m scripts.chat_cli -p "What is the capital of France?"` returns a non-empty answer (quality is not a gate).
7. **Second run:** `modal volume get --force` refreshes the chat dir; shards and finished stages SKIP.
8. **Update trackers** with real numbers (`PROGRESS.md`, `TRAIL.md`), then open a PR from this branch. Do not commit to `master`.
9. **Review:** nWave reviewers have not seen `03d0c4d`, `9eed864`, `31974eb`, `3c6ba61`. Ask for a review.

## Later (GPU phase)

- Add a payment method at Modal (Settings → Billing) for the `neomatrix369` workspace.
- Re-run `MODE=gpu bash runs/runmodal.sh` after an explicit go-ahead (~$40–60, ~2 h). Prefetch 170 shards is CPU but still billable.

## Known gaps (not blockers)

- Docs not yet updated for the CPU mode: `docs/MODAL.md`, `README` Modal pointer. The runmodal banner is the source of truth until then.
- Run pytest as `uv run python3 -m pytest`. Ruff: `uvx ruff check runs/ tests/`.
- Sourcery suggests `contextlib.suppress` in `modal_run.py` (cosmetic).
