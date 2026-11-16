# Eval resume and tracker cleanup

Date: 2026-11-15
Status: APPROVED 2026-11-15. S0 DONE (uncommitted, branch `docs/tracker-cleanup-2026-11-15`). S1 and S2 NOT STARTED (S1 is test-writing, so it needs its own test-planning step first).
Decisions: (1) per task granularity, because tasks are the reported unit and per-problem adds code for little gain. (2) only full eval runs set the marker.
Scope: only the critical and important items. Ranks 4–7 (Could), SFT stale-`.tmp` cleanup, and the duplicate-eval-on-resume nit are out of scope here and stay in TRAIL.

## Why

1. **Eval cannot resume.** `scripts/base_eval.py` and `scripts/chat_eval.py` keep all results in memory and write only at the end. A crash part-way through restarts the whole eval.
2. **The eval stage marker can lie.** The runner writes the stage marker when the command exits 0 (`runs/_stage_lib.sh` around lines 405–412). A partial run (one task, or `--eval bpb` only) therefore counts as finished, and the full eval is skipped next time.
3. **The tracker is wrong.** TRAIL says ranks 2, 3, 3a and 3b are not started, but the code and tests show them done. Anyone reading TRAIL would redo finished work or misjudge risk.

## Slices

### S0 — Reconcile trackers (Must, docs only, first)

- Mark done with evidence (88 targeted tests passed on 2026-11-15: `test_stage_markers`, `test_modal_run_paths`, `test_run_lock`, `test_sft_resume`, `test_checkpoint_errors`):
  - Rank 2 (B2): depth-qualified stage markers, `runs/_stage_lib.sh:144`.
  - Rank 3 (B3): heartbeat lock, `runs/_run_lock.sh`, `tests/test_run_lock.py`.
  - Rank 3a: `.python-version` = 3.12, `pyproject.toml` `requires-python >=3.12`.
  - Rank 3b: `base_train` calls `assert_resume_config_matches` (`scripts/base_train.py:382`), tested in `tests/test_sft_resume.py`.
- Rank 1 (B1): code and mocked test done; **live download still unverified**. Keep it open until a real download runs.
- Fix the `docs/DIAGRAMS.md` line that says `base_train` has no resume guard.
- Replace or retire `HANDOFF.md` (it names branch `feat/modal-nanochat-minimum`).

**Done when:** TRAIL, PROGRESS and DIAGRAMS agree with the code; no finished item is listed as open.

### S1 — Per-task eval results (Must)

- Each finished task writes one small JSON result file, keyed by model step, task name and config.
- On rerun, a task with an existing result file is skipped and its saved number is reused.
- Writes are atomic (same pattern as `nanochat/checkpoint_manager.py:63-67`). Reuse that helper; do not copy it.
- Applies to `scripts/base_eval.py` and `scripts/chat_eval.py`.
- Tests first: a finished task is not re-run; a killed run keeps finished tasks; a half-written file is ignored.

**Done when:** a local eval killed mid-run and restarted re-runs only the unfinished tasks, and the final numbers match an uninterrupted run. Save the logs under `docs/evidence/` in the same style as the SFT resume evidence.

### S2 — Honest eval marker (Should)

- Only a run that covers every requested task and mode may write the eval stage marker.
- A partial run exits without writing the marker, and the runner reports it as incomplete.
- Tests: a subset run leaves no marker; a full run writes it.

**Done when:** `--eval bpb` or `-a <one task>` never causes a later skip of the full eval.

### Blocked (not started; needs your explicit go-ahead)

- TRAIL rank 14: live Modal CPU run. Costs money.
- TRAIL rank 13: Modal preemption resume. Needs rank 14.
- TRAIL rank 8: CPU timing docs. Needs a finished rank-14 run.
- TRAIL rank 11: GPU run. Needs a payment method and your go-ahead.
- B1 live download check. Needs a Modal Volume.

## Order

S0 → S1 → S2. Each slice is its own branch and PR, and nothing is committed to `master`.

## Verification (every slice)

- `uv run python3 -m pytest` for the touched test files, then the full suite.
- `uvx ruff check runs/ tests/` and `nanochat/` / `scripts/` as touched.
- Docs changed in the same PR as the code they describe.

## Open decisions

1. **Eval resume granularity:** per task (recommended) or per problem (finer, more code)?
2. **Marker rule:** only full runs write the marker (recommended), or keep current behaviour?
