# Plan: check Modal-side changes reach local CPU/GPU runs

## Context

Work on PR #26 (`feat/checkpoint-mid-train-resume`, not yet merged to `master`) and the Modal
fixes merged on `master` (PRs #21–#25) was built mostly for Modal. The question: is that code
shared, so local `runs/runcpu.sh` (Mac CPU/MPS) and `runs/speedrun.sh` (8×GPU) get it too,
rather than a second copy?

Note: `gh` is not installed, so PR #26's GitHub page was not read. Graphiti MCP is down, so no
prior-decision lookup was done. Findings come from `git diff master...HEAD` and the files.

## Findings

**Shared (local and Modal both use it, single owner):**

| Change | Owner file | Local use |
|---|---|---|
| Atomic writes (`*.tmp` then `os.replace`) | `nanochat/checkpoint_manager.py` | Every checkpoint save |
| `find_safe_resume_step` | `nanochat/checkpoint_manager.py` | Via `resume_flag` |
| `resume_flag` helper | `runs/_stage_lib.sh` | `runcpu.sh` and `speedrun.sh` |
| `--save-every-secs` (time-based save) | `scripts/base_train.py` | `runcpu.sh` (300 s default), `speedrun.sh` |
| `NANOCHAT_SAVE_EVERY_SECS` knob | `runs/runcpu.sh`, `runs/speedrun.sh` | Local only (env var) |
| CLI colour, plaques, learn line, `NO_COLOR` (#21–#23) | `nanochat/*`, `runs/_stage_lib.sh` | Local terminal output |

Verdict: the resume and checkpoint logic is already one shared owner. Modal does not copy it.

**Modal-only (correct to stay Modal-only):**

- Commit daemon and periodic Volume commit every 300 s (`runs/modal_run.py`). A Modal Volume
  does not exist locally.
- SIGTERM handler that stops the process group, then commits (`runs/modal_run.py`).
- Launch concurrency guard for the shared Volume (PRs #24/#25, in `runs/runmodal.sh` lines 27–42,
  not in `modal_run.py`). Local runs do not share the Volume.
- `commit_every_secs` and `num_iterations` overrides on `train_cpu` (bbd7e48). These are Modal
  function arguments. Local runs use the env vars `NANOCHAT_RUNCPU_NUM_ITERATIONS` and
  `NANOCHAT_SAVE_EVERY_SECS` instead.

**Gaps that affect local runs too (not Modal-specific):**

1. `chat_sft` has no resume wiring (`scripts/chat_sft.py`, no `--resume-from-step`). Local runs
   restart SFT from zero after a kill. The plan's Phase 3 covers it; the comment in `runcpu.sh`
   and `speedrun.sh` says it's intentional.
2. `tok_train` atomicity (plan Phase 2). Still open. `dataset.py` already writes `*.tmp` then renames (lines 105–111), so it is not a gap.
3. Mid-train resume is marked "not yet e2e-tested" in `docs/plan/2026-11-14-resume-all-stages.md`.
   Local CPU is the cheapest place to test it.
4. A local "two runs at once" guard does not exist. Two local runs on one base dir can still collide.
   Only a concern if you run two locally at once.

## Recommended approach

Filter rule: change code only if it adds real value or impact. No refactor for its own sake.

**No code change is needed for the reuse question.** The shared pieces are already shared, and
Modal-only pieces should stay Modal-only.

Only two actions pass the filter:

1. **Verify local resume (no code change, free):** kill `runs/runcpu.sh` mid `base_train`, restart,
   confirm it resumes from the last complete checkpoint. This closes gap 3 and proves value that
   local users already get.
2. **Chat SFT resume (gap 1), only if you want local SFT crash-safety:** it is the one real gap that
   local runs hit, and the plan already marks it as a follow-up PR. Skip it if SFT restarts are
   acceptable, since SFT is about 10 minutes locally.


## Step A: end-to-end mid-train resume test (base_train, local)

Mid-train resume has never been tested end to end. Do this before any SFT work.

1. Run `runs/runcpu.sh` with a scratch `NANOCHAT_BASE_DIR`, a small `NANOCHAT_RUNCPU_NUM_ITERATIONS`
   (so it takes minutes, not 30), and a small `NANOCHAT_SAVE_EVERY_SECS`.
2. Kill the `base_train` process group (`kill -TERM -- -<pgid>`) after at least one checkpoint is saved.
3. Confirm the checkpoint is complete with `python -m nanochat.checkpoint_manager find-resume-step <dir>`,
   which must print a step number, not -1.
4. Restart `runs/runcpu.sh`. Confirm the log says "Resuming optimization from step N", and that
   the first printed step line is `step N` (checkpoint N holds the state before step N runs; `scripts/base_train.py:604`).
5. Run a control on CPU in its own `NANOCHAT_BASE_DIR`, uninterrupted, with the fixed seed (`nanochat/common.py:211`).
   Compare step N loss and `epoch/pq/rg` with a stated tolerance. MPS and some kernels are not bit-deterministic
   (`nanochat/common.py:211-215`), so exact equality is not expected off CPU. These prove optimizer and
   dataloader state were restored, not only that a file was found.
6. Let it finish. Confirm `.train_ok` is written once, after the last step.

Exit: a recorded log excerpt of the kill, the resume line, the matching control line, and the final marker.

## Step B: chat SFT resume (only if you approve, see question below)

Design, mirroring `base_train` and reusing the shared helpers:

- **Save:** add `--save-every-secs` to `scripts/chat_sft.py`, same logic as `base_train.py` (rank 0 decides, all ranks follow).
- **Save extra state:** the SFT data generator keeps its position in local variables (`cursor`, `consumed`,
  `epoch`, `it`, and `conv_buffer`). Store these in `meta` so a resumed run continues the same
  conversation order without repeats or skips. Also store loop state (`progress`, `min_val_bpb`,
  `smooth_train_loss`, `total_training_time`), as `base_train` does.
- **Size budget:** `conv_buffer` holds up to 100 conversations of up to 2049 token ids plus masks. That adds roughly
  1–2 MB to each `meta_*.json`. It is fine for the 300 s save interval, but it is a deliberate cost to accept.
- **Serialization:** the buffer holds plain Python lists (`nanochat/tokenizer.py:148, 224`), so `json` can write them
  as-is. No tensor conversion is needed.
- **Resume:** add `--resume-from-step` to `scripts/chat_sft.py`, with the same `load_checkpoint` path.
- **Config check (new guard, not a reuse: `base_train` has no equivalent today):** on resume, refuse to continue if `device_batch_size`, `max_seq_len`, `total_batch_size`,
  or `num_iterations` differ from the saved `meta`. A changed batch shape would silently break the saved position.
- **Shell:** wire `resume_flag chat_sft <dir> <nproc>` and `NANOCHAT_SAVE_EVERY_SECS` into
  `runs/runcpu.sh` and `runs/speedrun.sh`, and remove the "no resume wiring intentionally" comments.
- **Reuse:** `save_checkpoint`, `load_checkpoint`, `find_safe_resume_step`, `resume_flag`. No new resume helper.
- **No change to `find_safe_resume_step`:** it takes any directory. `runs/runcpu.sh` and `runs/speedrun.sh` pass the
  `chatsft_checkpoints/<tag>` path to `resume_flag`, so the function needs no edit.
  Run the local kill-and-resume test from Step A on the SFT stage.
- **Open check before coding:** confirm `TaskMixture` order is deterministic (no shuffle), so restoring the cursor gives the same conversations.

**Tests (decided in an interview, not here):** the user's global rule (`~/.claude/CLAUDE.md`, Testing section) requires
an interview, one question at a time, before any test is written. The test list is decided there. Candidates it should cover:

- Round-trip: save then load SFT state; the next N batches after restore equal the uninterrupted run's batches. `TaskMixture` is deterministic (`tasks/common.py:146-148`, seeded `random.Random(42)`), so pin this.
- Config-mismatch refusal: resume refuses if `device_batch_size`, `max_seq_len`, `total_batch_size`, or `num_iterations` differ.
- Order: extract the generator state (currently closure variables, `scripts/chat_sft.py` ~198-234) into a plain object first, then write the round-trip test against it.
- Cheap base checkpoint for the test: a tiny `--depth` and few iterations, not the 30-minute run.

**Save-timing test:** extract the `--save-every-secs` due-time check (`scripts/base_train.py:596-604`) into a pure function and test it. Keep the DDP broadcast (`:600-603`) in the caller. The function must also cover the `step > 0` and resume-step exclusion from `:604`.

**Shell tests:** `resume_flag` and the `find-resume-step` CLI have no tests today. Add a fixture-dir test for
`--resume-from-step=N`, and for `FORCE_STAGES` returning nothing. `tests/test_stage_completion_markers.sh` is not run by pytest, so decide whether to run it in CI or by hand.

## Step C: local concurrency guard (keep, per user)

Two local runs on the same `NANOCHAT_BASE_DIR` can collide. Add a lock so a second run refuses to start.

- Portable lock: `flock` is not installed on stock macOS, so use an atomic `mkdir "$BASE_DIR/.run.lock"`
  with a PID file inside. On start, if the directory exists and its PID is dead, reclaim it by atomically
  renaming the stale directory first, so two runs cannot both reclaim it. Otherwise refuse.
  Remove the lock on exit with a `trap` (EXIT and INT/TERM). Check PID liveness with `kill -0`.
- Put the helper in its own file, `runs/_run_lock.sh`, sourced by `runcpu.sh` and `speedrun.sh` after
  `_stage_lib.sh` is sourced (`runs/runcpu.sh:23`) and before the dataset download (`runs/runcpu.sh:54`), so a
  refused second run downloads nothing. `_run_lock.sh` must not source `_stage_lib.sh`.
- A second run prints a clear message and exits non-zero.
- Verify with a cheap test, no data download: a background holder takes the lock (the test uses the holder's
  `$!` PID, not the test shell's `$$`), the second acquire must fail, kill the holder, the acquire must then
  succeed, and a stale PID-file lock must be reclaimed.
- Decision changed: this guard was dropped earlier and is now kept, per user.

## Modal refactor (keep, per user)

Scope (user choice after design review: daemon cleanup + guard test; the shared lock is dropped):

- **Corrected (design review):** the Modal guard and the local lock are different mechanisms. The Modal guard
  checks cloud apps with `modal app list --json` (`runs/runmodal.sh:27-42`). The local lock is a file lock
  (Step C). Moving the Modal check into a shared helper would make the `modal` CLI a requirement for every local
  run. So the Modal guard stays in `runs/runmodal.sh`, and the two do not share one implementation.
- Test the guard in `runs/runmodal.sh` (`:26-42`). `tests/test_runmodal_mode.py` never sends `app list --json`,
  so the guard's refusal branch is never run. Add a fake `modal` that prints JSON such as
  `[{"app_id":"ap-1","description":"nanochat-speedrun","stopped_at":null}]`. Assert a non-zero exit and that no `run`
  call happens. Add cases for `RUNMODAL_ALLOW_CONCURRENT=1` and for invalid JSON.
- Before the daemon cleanup, add a characterization test for the SIGTERM handler (`runs/modal_run.py:290-296`),
  since `tests/test_periodic_commit.py` does not touch signals. The cleanup must keep it green.
- The "shared lock" part of the original scope was based on the wrong premise and is removed. Scope is settled.
- **Daemon cleanup:** tidy `runs/modal_run.py` commit daemon and SIGTERM handling. No behaviour change.
  Verify with the existing `tests/test_periodic_commit.py` and `tests/test_runmodal_mode.py`.

## Dropped

- `tok_train` atomicity: low impact, left alone (user decision).

## Verification

- Step A end to end (local, CPU, no GPU cost).
- Step B: local kill-and-resume on the SFT stage, plus the existing shared-helper tests
  (`tests/test_find_safe_resume_step.py`, `tests/test_periodic_commit.py`).
- Read-only checks so far: `git diff master...HEAD --stat`, and `git diff` on the shared files.

## Next-iteration priority (MoSCoW)

**Done:** reconcile the resume verification claim. Local kill and SIGTERM checks are recorded in
`docs/RUNNING.md` and the trackers (commit `f14c355`). Loss is close but not bit-identical, and the
Modal path is not covered.

**Must**
- **Step B: `chat_sft` resume.** Requested explicitly: idempotent and recoverable. The one real local gap.
  Needs the test interview (see Tests section) before any test is written.

**Should**
- **Step A: control-run comparison.** Same-seed control run, to settle the loss gap. Records evidence for PR #26.
- **Step C: local run lock (`runs/_run_lock.sh`).** Requested to keep. Prevents two local runs sharing a base dir.
- **Modal guard test.** Covers the launch-guard refusal branch in `runs/runmodal.sh` (see Modal refactor section).

**Could**
- **Modal daemon cleanup.** Refactor with no behaviour change. Do it after the Should items, and only if time allows.

**Won't (this iteration)**
- **Live Modal run (about $40–60).** Needs an explicit go-ahead from you.
- **`tok_train` atomicity.** Dropped as low impact.
