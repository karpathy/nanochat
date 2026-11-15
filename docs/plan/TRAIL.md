# TRAIL

## Needs your decision

None right now. The `nan` Modal run (`ap-1uTQ0hJAC2gQgVwK3YRLVR`) was stopped on 2026-11-15 after it failed. See rank 14.

## Backlog (order agreed 2026-11-15)

Work top to bottom. MoSCoW from the plans. A row moves to the history table below once merged or closed.

| rank | slice_id | MoSCoW | plan | state | notes |
|------|----------|--------|------|-------|-------|
| 1 | modal-fix-b1-download | **Must** | [modal-lifecycle-bugs](2026-11-15-modal-lifecycle-bugs.md) B1 | CODE FIXED, NOT VERIFIED | Commit `829d5f7` changed `_download_artifacts()` (`runs/modal_run.py:503-524`) to walk `chatsft_checkpoints/` recursively. No test yet and no download run yet. Next: add a test with a mocked Volume, then mark DONE. Docs callout in `docs/MODAL.md` updated 2026-11-15. |
| 2 | modal-fix-b2-depth-markers | **Must** | [modal-lifecycle-bugs](2026-11-15-modal-lifecycle-bugs.md) B2 | PARKED until the GPU run is near (user, 2026-11-15) | Still open: stage markers in `runs/_stage_lib.sh` carry no profile or depth. A GPU run can skip on a CPU depth-6 checkpoint. Workaround: `FORCE_STAGES=all`. Must land before any GPU run. |
| 3 | modal-fix-b3-heartbeat-lock | **Should** | [modal-lifecycle-bugs](2026-11-15-modal-lifecycle-bugs.md) B3 | NOT STARTED | Lock still uses PID + `kill -0` (`runs/_run_lock.sh:87`, `:132`). Replace with run ID + heartbeat (600 s stale threshold). Do before the next Modal run can be cancelled. Workaround: `modal volume rm -r nanochat-modal .run.lock`. |
| 3a | env-test-import-python-pin | **Should** | PR #41 caveat | NOT STARTED | Plain `uv run pytest` fails at collection (`No module named 'nanochat'`); it passes with `PYTHONPATH=.`. Fix the import path (project install or a pytest `pythonpath` setting). `.python-version` pins 3.10, which Modal flags as end-of-life (its pytest-run warning: `Python 3.10 is EOL and Modal will not support it in version 1.7.0`). Choose the pin and update it. |
| 3b | base-train-resume-guard | **Should** | [DIAGRAMS.md](../DIAGRAMS.md#checkpoint-and-resume) | NOT STARTED | `base_train` resumes with no config check, so changed settings resume silently. `chat_sft` refuses on the same kind of change. Add the check to `base_train` using the existing `assert_resume_config_matches`, with a test. |
| 4 | run-record-replay | **Could** | Session plan `~/.claude/plans/with-our-current-setup-tranquil-rossum.md` (not in repo) | NOT STARTED | Add `python3 -m nanochat.run_record replay <run-dir>`. Print `git checkout <code_rev>`, `uv sync --extra cpu` for local runs, and `MODE=<mode> bash runs/runmodal.sh`. The old plan said `--mode`, but `runmodal.sh` reads `MODE` from the environment. Only `nanochat/run_record.py` changes. |
| 5 | cli-learn-color-could-items | **Could** | [cli-learn-color-ux](2026-11-14-cli-learn-color-ux.md) "Could" | NOT STARTED | Colour and define lines for `tok_train`, `tok_eval`, `chat_sft`, `chat_eval`; `chat_cli` welcome plaque. Decide whether `base_train` keeps its `CORE` define call. |
| 6 | train-heartbeat-eta | **Could** | [GROWTH.md](GROWTH.md) fix-forward-mac-phase15 "Deferred" | NOT STARTED | Heartbeat and train-loop ETA. Planned in phase 1.5, never built. Not the same as B3 (lock heartbeat). |
| 7 | review-fix-could-suggestions | **Could** | [pr27-29-review-fixes](2026-11-14-pr27-29-review-fixes.md) S3,S4,N1-N2 | NOT STARTED | S3 (trap-ownership comment in `_run_lock.sh:34-36`), S4 (cap serialised `doc_buffer` in `dataloader.py`). Nits: N1 (explicit `time_due` param in `should_save_checkpoint`), N2 (`conv_buffer` type annotation in `sft_loader.py`). |
| 13 | modal-preemption-resume | **Should** | [resume-all-stages](2026-11-14-resume-all-stages.md) "Modal path not covered" | REOPENED: NOT VERIFIED on Modal. Unblocked 2026-11-15 (user) | Local resume is verified (Step A; SIGTERM on MPS). The Modal path is not. Run `20261114-125440-cpu` was preempted (`run.log:4440`) and restarted `base_train` at step 0 with `ckpt=n/a` (`run.log:4459`), so it did not resume. The user then stopped it from the CLI (`run.log:4839`). The traceback after that is the KeyboardInterrupt from that stop, not a training bug. The earlier DONE claim is withdrawn. |
| 14 | modal-cpu-live-run | **Should** | [HANDOFF.md](../../HANDOFF.md) steps 1 to 8 | REOPENED: NOT COMPLETE. Unblocked 2026-11-15 (user). A new paid Modal CPU run still needs a separate go-ahead | No archived run prints `ALL DONE`. `20261114-125440`: preempted, restarted from step 0, then stopped by the user (`run.log:4839`). `20261115-004549`: build aborted because `.omc/state/idle-notif-cooldown.json` changed during the image build. `20261115-004651` and `20261115-125031`: only the Initialized line, outcome not recorded. `20261115-010057`: ConnectionError (DNS). `20261115-131340`: lock conflict, another run active on PID 5 (`run.log:25`). `20261115-132517`: base_train and base_eval DONE; `chat_sft` loss `nan` from step 4 (`run.log:2966`). Then the container was preempted and restarted, the restart hit a stale lock (`another run is already active ... PID 5`), and `runcpu.sh` exited 1. The Modal app was stopped on 2026-11-15 (state `stopped`). Before the next run, remove the stale lock if it is still on the volume: `rm -rf /data/nanochat-modal/.run.lock`. Measured numbers: none. |
| 8 | modal-cpu-timing-docs | **Should** | [MODAL.md](../MODAL.md), [README.md](../../README.md) "Cloud CPU" row | BLOCKED on rank 14 | README and MODAL.md say CPU timing is "Unmeasured". That is accurate today because no run has finished. Once a CPU run prints `ALL DONE`, measure wall time, stage times and tok/s with `run_record compare` on the pulled run record, and put the numbers in both docs. Blocked until a CPU run completes successfully. |
| 9 | modal-pipeline-diagram | **Could** | [MODAL.md](../MODAL.md) "How the pieces fit"; [DIAGRAMS.md](../DIAGRAMS.md) | DONE (aafb840) | Modal data flow and commit handshake added to MODAL.md; training pipeline and skip-decision diagrams in DIAGRAMS.md. Checked against the runners. Checkpoint and resume diagram still planned (DIAGRAMS.md). |
| 10 | agents-md-decision | **Could** | global `~/.claude/CLAUDE.md` (imports `@AGENTS.md`) | DECISION NEEDED | Repo has no `AGENTS.md` or `CLAUDE.md`, so the sync-docs agent rows are not applicable. Decide: add a short `AGENTS.md` pointing to this `docs/plan/` tracker, or leave it out. Leaving it out is the default. |
| 11 | modal-gpu-phase | **Won't (for now)** | [HANDOFF.md](../../HANDOFF.md) | BLOCKED | 8×H100 run (~$40–60, ~2 h). No payment method on the `neomatrix369` Modal workspace. Needs a card and your explicit go-ahead. Do only after rank 2 (B2) lands. |

### Won't (this iteration)

| slice_id | reason |
|----------|--------|
| tip-rotation-knob-diff-ui | Deferred in GROWTH phase 1.5. Only one line of tip text ships. |
| tok-train-atomicity | Low impact. User decision. |
| softcap-logits-memory | Parked in the 2026-11-14 backlog decision. No design chosen. |
| base-eval-headers-colour | Would change existing script output. |

### Closed (reference, moved from the old backlog)

| rank (old) | slice_id | MoSCoW | state | notes |
|------------|----------|--------|-------|-------|
| 1 | review-fix-b1-nameerror | Must | DONE (on fix/resume-review-blockers) | `del _saved_meta` fix in `chat_sft.py` (commit `cc5fab1`). |
| 2 | review-fix-s2-base-dir | Must | DONE | `${NANOCHAT_BASE_DIR:?must be set}` in `_run_lock.sh` (commit `aee798f`). |
| 3 | review-fix-b4-guard-extract | Must | DONE | `assert_resume_config_matches` in `nanochat/checkpoint_manager.py`. |
| 4 | review-fix-b2-sft-weights | Must | DONE | `apply_sft_resume_state` in `checkpoint_manager.py`; tested in `test_sft_resume.py`. |
| 5 | review-fix-b3-ddp-sharding | Must | DONE | `ddp_rank_position` in `nanochat/common.py`. |
| 6 | review-fix-b5-lock-race | Must | DONE | `mkdir .reclaim` gate + ownership check (commit `aee798f`). |
| 7 | resume-data-position | Must | PASSED 2026-11-15 | Commit `97ec941`. Step A: steps 7–199 match (evidence `docs/evidence/resume-step-a-pass-2026-11-15/`). |
| 8 | resume-wording-correction | Must | DONE 2026-11-15 | Wording fixed in RUNNING.md and resume-all-stages.md. Data order checked only through rank 7. |
| 9 | master-warning-70ce9e2 | Must | DONE (user decision: leave it) | Docs-only commit `70ce9e2` on `master`. |
| 10 | live-modal-apps-check | Must | DONE 2026-11-15 | `modal app list`: earlier `nanochat-gpu-probe` apps are stopped. |
| 11 | review-fix-should-suggestions | Should | DONE (PRs #33, #34, #35) | S7 in PR #33; S1 in [PR #34](https://github.com/neomatrix369/nanochat/pull/34); S5+S6 in [PR #35](https://github.com/neomatrix369/nanochat/pull/35). |
| 12 | modal-guard-tests | Should | DONE (PR #30) | Cases a–f in `tests/test_runmodal_mode.py` (commit `dbd6445`). |
| 15 | modal-review-and-cpu-docs | Should | DONE (PR #37) | Lifecycle gaps in `docs/MODAL.md`, `docs/RUNNING.md` §13, `README.md`. |
| 16 | graphiti-stepA-save | Should | DONE 2026-11-15 (see PROGRESS.md) | Graphiti up; save written. Local fallback not needed. |
| 17 | modal-daemon-cleanup | Could | DONE (PR #30) | `runs/modal_run.py` daemon + SIGTERM cleanup (commit `651d789`). |

## History

| slice_id | plan | state | notes |
|----------|------|-------|-------|
| fix-forward-mac-phase1 | [2026-11-13-fix-forward-mac.md](2026-11-13-fix-forward-mac.md) Phase 1 | PASSED (merged) | [PR #4](https://github.com/neomatrix369/nanochat/pull/4) + follow-up [PR #5](https://github.com/neomatrix369/nanochat/pull/5) |
| fix-forward-mac-phase2 | same plan Phase 2 | PASSED (merged) | [PR #6](https://github.com/neomatrix369/nanochat/pull/6) merged `f8d7174` |
| fix-forward-mac-phase15 | [2026-11-13-fix-forward-mac-phase15.md](2026-11-13-fix-forward-mac-phase15.md) | PASSED (merged) | [PR #8](https://github.com/neomatrix369/nanochat/pull/8) merged `3054e4a` |
| fix-mps-skip-inductor | (hotfix; see GROWTH / RUNNING §7) | PASSED (merged) | [PR #10](https://github.com/neomatrix369/nanochat/pull/10) merged `f6ef779` |
| modal-nanochat-minimum | [2026-11-14-modal-nanochat-minimum.md](2026-11-14-modal-nanochat-minimum.md) | MERGED; CPU run NOT complete (see rank 14) | [PR #14](https://github.com/neomatrix369/nanochat/pull/14) merged `5225a3d`. Modal image build and 8×H100 run not executed. |
| resume-all-stages | [2026-11-14-resume-all-stages.md](2026-11-14-resume-all-stages.md) | MERGED; Step A PASSED 2026-11-15 | [PR #26](https://github.com/neomatrix369/nanochat/pull/26); fix in [PR #33](https://github.com/neomatrix369/nanochat/pull/33). Modal path covered only by rank 13 CPU checks. |
| local-resume-and-guard | [2026-11-14-local-resume-and-guard.md](2026-11-14-local-resume-and-guard.md) | SUPERSEDED (MERGED) | Do not rebuild. |
| modal-guard-hardware-capture | [2026-11-14-modal-guard-hardware-capture.md](2026-11-14-modal-guard-hardware-capture.md) | MERGED (guard); accelerator open | [PR #24](https://github.com/neomatrix369/nanochat/pull/24), [PR #25](https://github.com/neomatrix369/nanochat/pull/25). |
| cli-learn-color-ux | [2026-11-14-cli-learn-color-ux.md](2026-11-14-cli-learn-color-ux.md) | MERGED | PRs [#20](https://github.com/neomatrix369/nanochat/pull/20) to [#23](https://github.com/neomatrix369/nanochat/pull/23). Could items at rank 5. |
| docs-sync-audit-2026-11-15 | `/sync-docs` audit (this session) | DONE on PR #41, pending merge | Fixed `docs/MODAL.md`: B1 callout now matches code; `finish` recovery command added. Reconciled TRAIL and PROGRESS with the code and logs. Decisions are in the 2026-11-15 rows of PROGRESS.md. |

### Housekeeping done

- Stray branch `docs/resume-priority-plan` is gone.
- Plan commit `3f1b67f` reached master through PR #27 and PR #28.
- PR #30 covered ranks 12 and 17. Merged.
- `nw-software-crafter-reviewer` review of PRs #27–#29 completed 2026-11-14: CHANGES_REQUESTED, 5 blockers. Fixes tracked at ranks 1–6, 11. Plan: [pr27-29-review-fixes](2026-11-14-pr27-29-review-fixes.md).
- 4-reviewer pass on PR #31 completed 2026-11-14: B5 promoted to Must; B1+B2 must ship together; rank 11 split into Should and Could (rank 22 at that time, now rank 7).
