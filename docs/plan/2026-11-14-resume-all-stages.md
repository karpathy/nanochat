# Plan: resume after interruption for every stage

Status: `base_train` resume verified end to end (Step A, 2026-11-15). Kill after the step-7 checkpoint, resume from it: 193 overlapping steps (7-199) match (pq and rg exact, loss within 1e-6). Evidence: `docs/evidence/resume-step-a-pass-2026-11-15/`. The earlier 2026-11-14 "189 steps match" claim came from a run made before the data-position fix and is withdrawn. `chat_sft` resume is implemented (`docs/plan/2026-11-14-local-resume-and-guard.md`); its data-order check has not been run. Branch: `docs/trail-priority-backlog`. Owner: Mani Sarkar.

## Goal

A run that is interrupted at any point (local kill, Modal preemption, crash) continues
from the last safe point with no flags. Finished stages are still skipped by their markers.
A forced stage (`FORCE_STAGES=<stage>`) always starts that stage clean.

Today only `base_train` resumes mid-stage. This plan covers the rest.

## Current state (checked in code, 2026-11-14)

| Stage | Interruption today | Resume point | Gap |
|---|---|---|---|
| `dataset` (shard download) | Skips shards that already exist; `.tmp` then rename per shard | Shard granularity | Verify a killed `.tmp` is re-fetched, not kept |
| `tok_train` | Restarts from zero | None | Tokenizer saved only at the end. Check whether a killed `tokenizer.save` can leave a partial `tokenizer.pkl` that `stage_done` treats as finished |
| `tok_eval` | Restarts from zero | None | Writes `.tok_eval_ok` at end. Cheap to rerun |
| `base_train` | Resumes | Last complete checkpoint (`find_safe_resume_step`) | **Verified e2e 2026-11-15** (Step A): steps 7-199 match after resume from step 7 (pq, rg exact; loss within 1e-6) |
| `base_eval` | Restarts from zero | None | Writes `.eval_ok` at end. CORE metric is slow to rerun |
| `chat_sft` | Resumes | Last complete checkpoint (`find_safe_resume_step`, auto-detect or `--resume-from-step`) | **Implemented, not verified.** Config guard, model and optimizer restore, and dataloader state are in place. Data-order check still open |
| `chat_eval` | Restarts from zero | None | Writes `.eval_ok` at end. Cheap to rerun |
| Modal wrapper (`modal_run.py`) | SIGTERM stops the whole process group, then final commit | Via the stage that is resumed | Only tested locally, not on a real preemption |

## Principles

1. Each stage owns its own resume. Shared logic lives in `_stage_lib.sh` (`resume_flag`)
   and `nanochat/checkpoint_manager.py` (`find_safe_resume_step`). No copies per stage.
2. A stage's completion marker is written only after its last output is complete.
   Partial output must never look finished.
3. Writes are atomic: write to `*.tmp`, then `os.replace`.
4. Each phase ships on its own and works without the later phases.

## Phases

### Phase 1: prove what exists (no new resume code)

- Run scenarios 1 to 5 from the PR test plan on CPU (`runcpu.sh`, scratch base dir).
- Kill `dataset` download midway. Restart. Confirm no corrupt shard is used.
- Kill `tok_train` midway. Restart. Confirm `tokenizer.pkl` is absent or whole, and
  that `stage_done tok_train` is false until it is.
- Kill `base_eval` and `chat_eval` midway. Restart. Confirm they rerun cleanly and write markers only at the end.

Exit: a table of observed results. Gaps found here go into Phase 2 or 3.

### Phase 2: make the cheap stages atomic

- `tok_train`: write the tokenizer files via `*.tmp` and rename, so the marker's
  artifact appears only when complete. Same for the `tokenizer.pkl` written by
  `scripts/tok_train.py`.
- Add a unit test: an interrupted tokenizer save leaves no `tokenizer.pkl`.
- `base_eval`, `chat_eval`, `tok_eval`: no change beyond confirming they write the
  marker last.

Exit: no stage can leave a partial artifact that its `stage_done` check accepts.

### Phase 3: resume `chat_sft` (implemented, not verified)

Status (checked 2026-11-15 against code): implemented.

- Checkpoints: `scripts/chat_sft.py` saves with `--save-every-secs` (default 300 s) and `--save-every` (default 0, disabled). Both are in `scripts/chat_sft.py:77-78`.
- Resume: `--resume-from-step` (`scripts/chat_sft.py:79`). Auto-detect via `find_safe_resume_step`. Config guard `assert_resume_config_matches`, model and optimizer restore `apply_sft_resume_state`, dataloader state saved in `meta` (`scripts/chat_sft.py:170-203`, `:497-504`).
- Wired: `resume_flag chat_sft` in `runs/runcpu.sh:85` and `runs/speedrun.sh:113`.
- Tests: `tests/test_sft_resume.py`.

Still open (the only outstanding exit criterion):
- **Data-order check:** run an interrupted `chat_sft` and an uninterrupted one. Confirm the conversation stream matches after resume, the same way Step A checked `base_train`.
- **Continuity, not step-count equality:** `chat_sft` has no fixed total step count (`scripts/chat_sft.py:246`). The exit criterion is continuity of the conversation stream and of the learning-rate progress.

Exit: an interrupted `chat_sft` resumes with the same conversation stream and learning-rate progress as an uninterrupted run.

### Phase 4: `dataset` and Modal

- `dataset`: confirm the `.tmp` rename is atomic and that a killed `.tmp` is removed
  or overwritten on restart. Add a test for both cases.
- Modal: run one real preemption (or simulate with `kill -TERM` on the container) and
  confirm that the last `base_train` checkpoint is committed, and that the next run resumes it.
  Record the observed grace period.
- Modal: confirm a crash-loop cannot happen. The current wrapper raises `RuntimeError`,
  so there is no automatic retry. Keep it that way unless a retry policy is added on purpose.

Exit: each stage's interruption behavior is recorded from a real run.

## Verification (per stage, before a stage counts as covered)

- Unit test for each new function or file-writing path.
- One CPU run killed mid-stage and restarted. Record the resumed step, or "restarted from zero"
  for stages that do not resume.
- `FORCE_STAGES=<stage>` restarts that stage from zero.
- The stage marker is absent while the stage is incomplete.

## Prioritized work order (2026-11-14)

Work through these in sequence. Items 1 and 2 are the top-priority resume fix.

### Do first

**1. Fix resume data position (Must).** New branch from `master`. The loader must restore
`rg_idx`/`pq_idx` on resume so the resumed run's epoch/pq/rg matches the uninterrupted run at
each step. Then run the **Step A control**: the check passes only if those fields match and losses
stay within a stated tolerance. This is a shared path — get a review before merging.

### Do after item 1 passes

**2. Correct resume wording** in `docs/RUNNING.md` and this file (`resume-all-stages.md`). Do not
write "verified" in either doc until the Step A check passes.

### Then the rest of the Must/Should list

**3. Step B — `chat_sft` resume (Phase 3 above).** Requires a test interview (one question at a
time) before any test code is written.

**4. Step C — local run lock (Phase 4 / `docs/plan/2026-11-14-local-resume-and-guard.md`).** Same
interview requirement before tests.

**5. Modal daemon cleanup (Could).** Last, and only if time allows.

### Housekeeping (any time)

- Push `3f1b67f` and open a PR for the plan branch — currently local only.
- Delete stray branch `docs/resume-priority-plan` if agreed.
- Decision pending: revert or force-push `70ce9e2` on `master` (do not touch `master` until
  decided).
- Graphiti: once it comes back online, save the Step A outcome there.

---

## Decisions (2026-11-14)

- Save interval is time-based (default 300 s), decided by rank 0 and broadcast to all ranks, so every rank saves the same step. Implemented in PR #26.
- `chat_sft` resume is a follow-up PR (Phase 3).
- Resuming on a different GPU count: out of scope.
- Corrupt-but-present checkpoint files: out of scope for now.
- Modal preemption: Modal docs say a preempted function is restarted on the same input automatically (https://modal.com/docs/guide/preemption). The grace period is not documented, so it is measured in Phase 4.

## Open questions

1. Is `tokenizer.save` atomic in practice? Phase 1 answers this before we change code.
