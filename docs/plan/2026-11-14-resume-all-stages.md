# Plan: resume after interruption for every stage

Status: proposed. Branch: `feat/checkpoint-mid-train-resume` (PR #26). Owner: Mani Sarkar.

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
| `base_train` | Resumes | Last complete checkpoint (`find_safe_resume_step`) | Covered by PR #26, not yet e2e-tested |
| `base_eval` | Restarts from zero | None | Writes `.eval_ok` at end. CORE metric is slow to rerun |
| `chat_sft` | Restarts from zero | None | Saves checkpoints, but no `--resume-from-step` is wired and no dataloader state is saved for resume |
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

### Phase 3: resume `chat_sft`

- Save a complete checkpoint every N steps (same `save_checkpoint` contract as `base_train`:
  model + meta + optim shards, atomic writes).
- Save the dataloader state in `meta`, as `base_train` already does, so a resumed run
  does not repeat or skip conversations.
- Add `--resume-from-step` to `scripts/chat_sft.py`.
- Wire `resume_flag chat_sft <dir> <nproc>` into `speedrun.sh` and `runcpu.sh`.
  Use the same `--save-every` default as base training.
- Tests: `find_safe_resume_step` reads the `chatsft_checkpoints` directory.
  A CPU run that is killed after step N resumes at N.

Exit: `chat_sft` resumes after a kill, with the same guarantees as `base_train`.

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

## Out of scope

- Resuming on a different number of GPUs than the run started with. Requires shard re-mapping; document as unsupported and fail loudly if detected.
- Time-based save interval. Reviewers suggested it; decision pending (see open questions).
- Automatic retry on preemption. Not added by this plan.

## Open questions

1. Save interval: keep steps (`--save-every=500`), or switch to time (every 5 minutes)?
   Reviewers recommended time, because CPU steps are much slower than GPU steps.
2. Should Phase 3 (`chat_sft`) ship in this PR, or as a follow-up PR? Recommendation: follow-up,
   so PR #26 can merge once `base_train` is verified.
3. Is `tokenizer.save` atomic in practice? Phase 1 answers this before we change code.
