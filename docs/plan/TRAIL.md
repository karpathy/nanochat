# TRAIL

## Backlog (priority order, 2026-11-14)

Work top to bottom. MoSCoW from the plans. A row moves to the history table below once merged or closed.

| rank | slice_id | MoSCoW | plan | state | notes |
|------|----------|--------|------|-------|-------|
| 1 | review-fix-b1-nameerror | **Must** | [pr27-29-review-fixes](2026-11-14-pr27-29-review-fixes.md) B1 | DONE (on fix/resume-review-blockers) | `del _saved_meta` fix in `chat_sft.py` (commit `cc5fab1`). Ships with ranks 2–6. |
| 2 | review-fix-s2-base-dir | **Must** | [pr27-29-review-fixes](2026-11-14-pr27-29-review-fixes.md) S2 | DONE (on fix/resume-review-blockers) | `${NANOCHAT_BASE_DIR:?must be set}` in `_run_lock.sh` (commit `aee798f`). |
| 3 | review-fix-b4-guard-extract | **Must** | [pr27-29-review-fixes](2026-11-14-pr27-29-review-fixes.md) B4 | DONE (on fix/resume-review-blockers) | `assert_resume_config_matches` extracted to `nanochat/checkpoint_manager.py:202`; `chat_sft.py` imports and calls it; `tests/test_sft_resume.py` imports and tests from `nanochat.checkpoint_manager`. Ships with this branch. |
| 4 | review-fix-b2-sft-weights | **Must** | [pr27-29-review-fixes](2026-11-14-pr27-29-review-fixes.md) B2 | DONE (on fix/resume-review-blockers) | `apply_sft_resume_state` extracted to `checkpoint_manager.py`; called from `chat_sft.py` on resume; tested in `test_sft_resume.py::TestApplySftResumeState` (commit `cc5fab1` + review fixes). |
| 5 | review-fix-b3-ddp-sharding | **Must** | [pr27-29-review-fixes](2026-11-14-pr27-29-review-fixes.md) B3 | DONE (on fix/resume-review-blockers) | `ddp_rank_position` extracted to `nanochat/common.py`; used in `dataloader.py` and `sft_loader.py` (commits `97ec941`, `2096327`, review fixes). |
| 6 | review-fix-b5-lock-race | **Must** | [pr27-29-review-fixes](2026-11-14-pr27-29-review-fixes.md) B5 | DONE (on fix/resume-review-blockers) | `mkdir .reclaim` gate + ownership check (commit `aee798f`); review fixes close mid-acquire window (empty PID = live-unknown). |
| 7 | resume-data-position | **Must** | [resume-all-stages](2026-11-14-resume-all-stages.md) "Prioritized work order" #1 | PASSED 2026-11-15 (Step A: steps 7-199 match; evidence `docs/evidence/resume-step-a-pass-2026-11-15/`) | Fix is commit `97ec941` on `fix/resume-review-blockers`. Step A verified 2026-11-15 (evidence in `docs/evidence/resume-step-a-pass-2026-11-15/`). Ships with this branch. |
| 8 | resume-wording-correction | **Must** | [resume-all-stages](2026-11-14-resume-all-stages.md) #2 | DONE 2026-11-15 (wording fixed in RUNNING.md and resume-all-stages.md; `chat_sft` data order still unchecked) | Fix "verified" wording in `docs/RUNNING.md` and `resume-all-stages.md`. Say the restart picks the right checkpoint and LR schedule; data order is not restored until rank 7 passes. |
| 9 | master-warning-70ce9e2 | **Must** | [resume-all-stages](2026-11-14-resume-all-stages.md) "Housekeeping" | DONE (user decision 2026-11-15: leave it) | Docs-only commit `70ce9e2` (MoSCoW plan) on `master` — left in place. No runtime impact. |
| 10 | live-modal-apps-check | **Must** | [modal-guard-hardware-capture](2026-11-14-modal-guard-hardware-capture.md) "Open blocker" | DONE 2026-11-15 | `modal app list` checked: all prior `nanochat-gpu-probe` apps show `stopped` state. No live apps. Modal CPU run (rank 14) launched. |
| 11 | review-fix-should-suggestions | **Should** | [pr27-29-review-fixes](2026-11-14-pr27-29-review-fixes.md) S1,S5-S7 | DONE (PRs #33, #34, #35 merged 2026-11-15) | S7 merged in PR #33. S1 (PID+start-time in lock): [PR #34](https://github.com/neomatrix369/nanochat/pull/34). S5+S6 (`SftLoaderState.initial` + guard from `user_config`, with tests for the `user_config` subdict path): [PR #35](https://github.com/neomatrix369/nanochat/pull/35). |
| 12 | modal-guard-tests | **Should** | [modal-guard-hardware-capture](2026-11-14-modal-guard-hardware-capture.md) Workstream A | DONE (PR #30 merged 2026-11-14) | Cases a, b, c1, c2, d, e, f in `tests/test_runmodal_mode.py` (commit `dbd6445`). |
| 13 | modal-preemption-resume | **Should** | [resume-all-stages](2026-11-14-resume-all-stages.md) "Modal path not covered" | DONE (verified 2026-11-15, multiple runs) | Modal preemption (SIGTERM) + commit gap verified across multiple CPU runs. Resume picks up correctly after preemption. |
| 14 | modal-cpu-live-run | **Should** | [HANDOFF.md](../../HANDOFF.md) steps 1 to 8 | DONE (completed 2026-11-15) | Full `runcpu` pipeline on Modal CPU completed. Record real numbers in `PROGRESS.md` and `docs/MODAL.md` (rank 15 covers docs). |
| 15 | modal-review-and-cpu-docs | **Should** | [HANDOFF.md](../../HANDOFF.md) "Known gaps" and step 9 | IN PROGRESS (another session 2026-11-15) | (a) Reviewer pass on `9eed864`, `31974eb`, `3c6ba61` and `03d0c4d`. (b) `docs/MODAL.md` CPU-mode section. Being completed by another session. |
| 16 | graphiti-stepA-save | **Should** | [resume-all-stages](2026-11-14-resume-all-stages.md) "Housekeeping" | WAITING (Graphiti down) | Save the Step A outcome to Graphiti once it is back online. Fallback: `~/.claude/knowledge-export.md`. |
| 17 | modal-daemon-cleanup | **Could** | [local-resume-and-guard](2026-11-14-local-resume-and-guard.md) "Modal refactor" | DONE (PR #30 merged 2026-11-14) | `runs/modal_run.py` daemon + SIGTERM cleanup (commit `651d789`). `contextlib.suppress` nit included. |
| 19 | run-record-replay | **Could** | Session plan `~/.claude/plans/with-our-current-setup-tranquil-rossum.md` (not in repo) | UNCLAIMED | Add `python3 -m nanochat.run_record replay <run-dir>` that prints `git checkout`, `uv sync`, and the `runmodal.sh --mode` line. Only `nanochat/run_record.py` changes. |
| 20 | cli-learn-color-could-items | **Could** | [cli-learn-color-ux](2026-11-14-cli-learn-color-ux.md) "Could" | UNCLAIMED | Colour and define lines for `tok_train`, `tok_eval`, `chat_sft`, `chat_eval`; `chat_cli` welcome plaque. Decide whether `base_train` keeps its `CORE` define call. |
| 21 | train-heartbeat-eta | **Could** | [GROWTH.md](GROWTH.md) fix-forward-mac-phase15 "Deferred" | UNCLAIMED | Heartbeat and train-loop ETA. Planned in phase 1.5, never built. |
| 22 | review-fix-could-suggestions | **Could** | [pr27-29-review-fixes](2026-11-14-pr27-29-review-fixes.md) S3,S4,N1-N2 | NOT STARTED | S3 (add trap-ownership comment in `_run_lock.sh:34-36`), S4 (cap serialised `doc_buffer` in `dataloader.py`). Nits: N1 (explicit `time_due` param in `should_save_checkpoint`), N2 (`conv_buffer: list[list[list[int]]]` type annotation in `sft_loader.py`). |

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
| modal-nanochat-minimum | [2026-11-14-modal-nanochat-minimum.md](2026-11-14-modal-nanochat-minimum.md) | MERGED; live run pending (see rank 14) | [PR #14](https://github.com/neomatrix369/nanochat/pull/14) merged `5225a3d`. CPU-mode commits `9eed864`, `31974eb`, `3c6ba61` are on master. Live Modal run not executed. |
| resume-all-stages | [2026-11-14-resume-all-stages.md](2026-11-14-resume-all-stages.md) | MERGED (`c9cf80e`); resume NOT equivalent yet | [PR #26](https://github.com/neomatrix369/nanochat/pull/26) merged to master. `base_train` mid-stage resume, time-based save, atomic writes. Step A done 2026-11-14: data position wrong (`rg` 3 vs 1). Fix is rank 7. |
| local-resume-and-guard | [2026-11-14-local-resume-and-guard.md](2026-11-14-local-resume-and-guard.md) | SUPERSEDED (MERGED) | Work shipped: Step A (`docs/evidence/resume-step-a-pass-2026-11-15/`), `chat_sft` resume (`scripts/chat_sft.py:170-203`), local lock (`runs/_run_lock.sh`, `tests/test_run_lock.py`), guard tests (`tests/test_runmodal_mode.py`). 86 tests pass across the resume, lock, dataloader, runmodal and SIGTERM suites. Do not rebuild. |
| modal-guard-hardware-capture | [2026-11-14-modal-guard-hardware-capture.md](2026-11-14-modal-guard-hardware-capture.md) | MERGED (guard); tests, runtime docs on master; accelerator open | [PR #24](https://github.com/neomatrix369/nanochat/pull/24) and [PR #25](https://github.com/neomatrix369/nanochat/pull/25) merged (`282c84c`, `ffea067`). Runtime-config docs on master (`fe8b8ba`). Ranks 7, 10, 13. |
| cli-learn-color-ux | [2026-11-14-cli-learn-color-ux.md](2026-11-14-cli-learn-color-ux.md) | MERGED | PRs [#20](https://github.com/neomatrix369/nanochat/pull/20) to [#23](https://github.com/neomatrix369/nanochat/pull/23) merged to master (`edfb3d3`, `9153d28`, `cf49fab`, `993962f`). Could items at rank 15. |

### Housekeeping done

- Stray branch `docs/resume-priority-plan` is gone (not in `git branch`).
- Plan commit `3f1b67f` reached master through PR #27 and PR #28.
- PR #30 created 2026-11-14 covering ranks 12 (modal-guard-tests, `dbd6445`) and 17 (modal-daemon-cleanup, `651d789`). Awaiting review and merge.
- `nw-software-crafter-reviewer` review of PRs #27–#29 completed 2026-11-14: CHANGES_REQUESTED, 5 blockers. Fixes tracked at ranks 1–6, 11. Plan: [pr27-29-review-fixes](2026-11-14-pr27-29-review-fixes.md).
- 4-reviewer pass on PR #31 (`docs/pr27-29-review-plan`) completed 2026-11-14: B5 promoted to Must (rank 6); B1+B2 must ship together (ranks 1, 4); B2/B3/B5 fix specs corrected; B4 locator fixed (`:860-927` → `:183-255`); rank 11 split into Should (rank 11) and Could (rank 22).
