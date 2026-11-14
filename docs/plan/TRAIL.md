# TRAIL

## Backlog (priority order, 2026-11-14)

Work top to bottom. MoSCoW from the plans. A row moves to the history table below once merged or closed.

| rank | slice_id | MoSCoW | plan | state | notes |
|------|----------|--------|------|-------|-------|
| 1 | resume-data-position | **Must** | [resume-all-stages](2026-11-14-resume-all-stages.md) "Prioritized work order" #1 | CODE COMMITTED (`65705f7`), STEP A CHECK NOT RUN | Fix commit `65705f7` "fix(dataloader): resume the train stream at the exact batch" is on `docs/trail-priority-backlog`. The old `fix/resume-data-position` branch was deleted (it had no unique commits). Review this commit on the docs branch, or move it to its own branch before review. Owner not recorded; peer session `nanochat-b0` was busy and may own it. Restore `rg_idx`/`pq_idx` on resume. Pass = Step A control (`rg` and `epoch`/`pq` match the uninterrupted run; loss within a stated tolerance). Shared path: needs review before merge. Evidence: `docs/evidence/resume-step-a-2026-11-14/`. |
| 2 | resume-wording-correction | **Must** | [resume-all-stages](2026-11-14-resume-all-stages.md) #2 | BLOCKED by rank 1 | Fix "verified" wording in `docs/RUNNING.md` and `resume-all-stages.md`. Say the restart picks the right checkpoint and LR schedule; data order is not restored until rank 1 passes. |
| 3 | master-warning-70ce9e2 | **Must** | [resume-all-stages](2026-11-14-resume-all-stages.md) "Housekeeping" | DECISION PENDING (user) | Commit `70ce9e2` (MoSCoW plan) landed on `master` and `origin/master` by mistake. Choose: revert commit (safer) or force-push. Do not touch `master` until chosen. |
| 4 | live-modal-apps-check | **Must** | [modal-guard-hardware-capture](2026-11-14-modal-guard-hardware-capture.md) "Open blocker" | UNVERIFIED | Two live `nanochat-speedrun` apps (`stopped_at: null`) were seen. Confirm they are the user's and stop them, or the guard refuses launches and they keep billing. Modal CLI was not available in the last check, so run `modal app list` first. |
| 5 | chat-sft-resume | **Must** | [local-resume-and-guard](2026-11-14-local-resume-and-guard.md) Step B | CODE COMMITTED on `fix/modal-daemon-cleanup` (`805e1e0`) | `SftLoaderState` dataclass, `should_save_checkpoint()`, `find_safe_resume_step` CLI, config-mismatch guard. 27 tests in `tests/test_sft_resume.py`. Pending PR and merge. |
| 6 | local-run-lock | **Should** | [local-resume-and-guard](2026-11-14-local-resume-and-guard.md) Step C | CODE COMMITTED on `fix/modal-daemon-cleanup` (`f5d6548`) | `runs/_run_lock.sh`: atomic `mkdir` + PID file (not `flock`). EXIT/INT/TERM traps. Stale PID reclaimed via atomic `mv`. 5 tests in `tests/test_run_lock.py`. Pending PR and merge. |
| 7 | modal-guard-tests | **Should** | [modal-guard-hardware-capture](2026-11-14-modal-guard-hardware-capture.md) Workstream A | CODE COMMITTED on `fix/modal-daemon-cleanup` (`dbd6445`) | 3 guard tests added to `tests/test_runmodal_mode.py`: refuse when app running, bypass with `RUNMODAL_ALLOW_CONCURRENT=1`, warn and proceed when modal unavailable. Total 7 tests pass. Pending PR and merge. |
| 8 | modal-preemption-resume | **Should** | [resume-all-stages](2026-11-14-resume-all-stages.md) "Modal path not covered" | UNCLAIMED | Modal preemption path not verified. Needed before any Modal resume claim. Includes the commit gap (SIGTERM before `vol.commit()`). |
| 9 | modal-cpu-live-run | **Should** (needs your go-ahead, billable) | [HANDOFF.md](../../HANDOFF.md) steps 1 to 8 | READY, NOT RUN | Full `runcpu` pipeline on Modal CPU (64 cores, 128 GiB, 3 h timeout). Code is on master (`9eed864`, `31974eb`, `3c6ba61`). Not verified: CPU image build, the core and memory cap being accepted, per-step timing (3 h limit), `modal volume get` of the chat checkpoint, a non-empty chat answer on the Mac, and `modal app stop` ending billing. Record real numbers in `PROGRESS.md` and `docs/MODAL.md`. |
| 10 | modal-review-and-cpu-docs | **Should** | [HANDOFF.md](../../HANDOFF.md) "Known gaps" and step 9 | UNCLAIMED | (a) Reviewer pass on `9eed864`, `31974eb`, `3c6ba61` and `03d0c4d`; none has been reviewed. (b) `docs/MODAL.md` and the README Modal pointer have no CPU-mode section yet. The runtime-config section is already on master (`fe8b8ba`), so only the CPU-mode part is open. |
| 11 | graphiti-stepA-save | **Should** | [resume-all-stages](2026-11-14-resume-all-stages.md) "Housekeeping" | WAITING (Graphiti down) | Save the Step A outcome to Graphiti once it is back online. Fallback: `~/.claude/knowledge-export.md`. |
| 12 | modal-daemon-cleanup | **Could** | [local-resume-and-guard](2026-11-14-local-resume-and-guard.md) "Modal refactor" | CODE COMMITTED on `fix/modal-daemon-cleanup` (`651d789`) | `threading.Event` stop signal wired; inner `commit_thread` fixed to `daemon=True`; `stop_event.set()` after `proc.wait()`. 1 new test in `tests/test_periodic_commit.py` (5 total). Pending PR and merge. |
| 13 | accelerator-capture | **Could** | [modal-guard-hardware-capture](2026-11-14-modal-guard-hardware-capture.md) Workstream C | IDEA ONLY | Record accelerator and RAM in `run.json`. C1: pin `system_ram_bytes` behaviour with a test, then extract `nanochat/hardware.py`. C2: add `ram_total_bytes` to `host_info()` (zero-arg). Device fields come from the training entry, not the launcher mode. |
| 14 | run-record-replay | **Could** | Session plan `~/.claude/plans/with-our-current-setup-tranquil-rossum.md` (not in repo) | UNCLAIMED | Add `python3 -m nanochat.run_record replay <run-dir>` that prints `git checkout`, `uv sync`, and the `runmodal.sh --mode` line. Only `nanochat/run_record.py` changes. |
| 15 | cli-learn-color-could-items | **Could** | [cli-learn-color-ux](2026-11-14-cli-learn-color-ux.md) "Could" | UNCLAIMED | Colour and define lines for `tok_train`, `tok_eval`, `chat_sft`, `chat_eval`; `chat_cli` welcome plaque. Decide whether `base_train` keeps its `CORE` define call. |
| 16 | train-heartbeat-eta | **Could** | [GROWTH.md](GROWTH.md) fix-forward-mac-phase15 "Deferred" | UNCLAIMED | Heartbeat and train-loop ETA. Planned in phase 1.5, never built. |

### Won't (this iteration)

| slice_id | reason |
|----------|--------|
| modal-gpu-phase | 8×H100 run (~$40–60, ~2 h). Blocked: no payment method on the `neomatrix369` Modal workspace. Needs a card and an explicit go-ahead. |
| tip-rotation-knob-diff-ui | Deferred in GROWTH phase 1.5. Only one line of tip text ships. |
| tok-train-atomicity | Low impact. User decision. |
| softcap-logits-memory | Parked in the 2026-11-14 backlog decision. No design chosen. |
| base-eval-headers-colour | Would change existing script output. |

## History

| slice_id | plan | state | notes |
|----------|------|-------|-------|
| fix-forward-mac-phase1 | [2026-11-13-fix-forward-mac.md](2026-11-13-fix-forward-mac.md) Phase 1 | PASSED (merged) | [PR #4](https://github.com/neomatrix369/nanochat/pull/4) + follow-up [PR #5](https://github.com/neomatrix369/nanochat/pull/5) |
| fix-forward-mac-phase2 | same plan Phase 2 | PASSED (merged) | [PR #6](https://github.com/neomatrix369/nanochat/pull/6) merged `f8d7174` |
| fix-forward-mac-phase15 | [2026-11-13-fix-forward-mac-phase15.md](2026-11-13-fix-forward-mac-phase15.md) | PASSED (merged) | [PR #8](https://github.com/neomatrix369/nanochat/pull/8) merged `3054e4a` |
| fix-mps-skip-inductor | (hotfix; see GROWTH / RUNNING §7) | PASSED (merged) | [PR #10](https://github.com/neomatrix369/nanochat/pull/10) merged `f6ef779`: skip Inductor on MPS, MPS AdamW/Muon, progress flush, `.train_ok` + CUDA hard-fail, RoPE grow-on-demand |
| modal-nanochat-minimum | [2026-11-14-modal-nanochat-minimum.md](2026-11-14-modal-nanochat-minimum.md) | MERGED; live run pending (see rank 9) | [PR #14](https://github.com/neomatrix369/nanochat/pull/14) merged `5225a3d`. CPU-mode commits `9eed864`, `31974eb`, `3c6ba61` are on master. Live Modal run not executed. |
| resume-all-stages | [2026-11-14-resume-all-stages.md](2026-11-14-resume-all-stages.md) | MERGED (`c9cf80e`); resume NOT equivalent yet | [PR #26](https://github.com/neomatrix369/nanochat/pull/26) merged to master. `base_train` mid-stage resume, time-based save, atomic writes. Step A done 2026-11-14: data position wrong (`rg` 3 vs 1). Fix is rank 1. |
| local-resume-and-guard | [2026-11-14-local-resume-and-guard.md](2026-11-14-local-resume-and-guard.md) | DECIDED (not started) | Plan approved 2026-11-14. Steps B, C and guard tests are ranks 5, 6, 7. |
| modal-guard-hardware-capture | [2026-11-14-modal-guard-hardware-capture.md](2026-11-14-modal-guard-hardware-capture.md) | MERGED (guard); tests, runtime docs on master; accelerator open | [PR #24](https://github.com/neomatrix369/nanochat/pull/24) and [PR #25](https://github.com/neomatrix369/nanochat/pull/25) merged (`282c84c`, `ffea067`). Runtime-config docs on master (`fe8b8ba`). Ranks 7, 10, 13. |
| cli-learn-color-ux | [2026-11-14-cli-learn-color-ux.md](2026-11-14-cli-learn-color-ux.md) | MERGED | PRs [#20](https://github.com/neomatrix369/nanochat/pull/20) to [#23](https://github.com/neomatrix369/nanochat/pull/23) merged to master (`edfb3d3`, `9153d28`, `cf49fab`, `993962f`). Could items at rank 15. |

### Housekeeping done

- Stray branch `docs/resume-priority-plan` is gone (not in `git branch`).
- Plan commit `3f1b67f` reached master through PR #27 and PR #28.
