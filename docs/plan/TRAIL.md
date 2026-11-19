# TRAIL

## Needs your decision

None right now. Rank 14 (modal-cpu-live-run) is complete. See new ranks 15–19 for arch-review follow-up work.

## Backlog (order agreed 2026-11-15)

Work top to bottom. MoSCoW from the plans. A row moves to the history table below once merged or closed.

| rank | slice_id | MoSCoW | plan | state | notes |
|------|----------|--------|------|-------|-------|
| 1 | modal-fix-b1-download | **Must** | [modal-lifecycle-bugs](2026-11-15-modal-lifecycle-bugs.md) B1 | DONE (verified 2026-11-17) | `_download_artifacts()` (`runs/modal_run.py:503-524`) walks `chatsft_checkpoints/` recursively (commit `829d5f7`). Live verified: `modal volume get` on run `20261116-154817-cpu-d6` downloaded `model_001499.pt` + `meta_001499.json` to `~/.cache/nanochat-modal/cpu/chatsft_checkpoints/20261116-154817-cpu-d6/`. Confirmed valid PyTorch checkpoint (zip archive). Local chat (`python3 -m scripts.chat_cli`) and remote chat (`modal run modal_run.py::chat`) both answered "Paris". |
| 2 | modal-fix-b2-depth-markers | **Must** | [modal-lifecycle-bugs](2026-11-15-modal-lifecycle-bugs.md) B2 | DONE (code + tests, verified 2026-11-15) | Stage markers are profile+depth-scoped: `.stage_<stage>_<PROFILE>_d<DEPTH>` (`runs/_stage_lib.sh:144`). Covered by `tests/test_stage_markers.py` (passes). |
| 3 | modal-fix-b3-heartbeat-lock | **Should** | [modal-lifecycle-bugs](2026-11-15-modal-lifecycle-bugs.md) B3 | DONE (code + tests, verified 2026-11-15) | Lock uses run ID + heartbeat with a 600 s stale threshold (`runs/_run_lock.sh`, commits `985b465`, `c93588a`). Covered by `tests/test_run_lock.py` (passes). |
| 3a | env-test-import-python-pin | **Should** | PR #41 caveat | DONE (verified 2026-11-15) | `.python-version` is 3.12; `pyproject.toml` has `requires-python = ">=3.12"` and `pythonpath = ["."]`. |
| 3b | base-train-resume-guard | **Should** | [DIAGRAMS.md](../DIAGRAMS.md#checkpoint-and-resume) | DONE (verified 2026-11-15) | `scripts/base_train.py:382` calls `assert_resume_config_matches`. Covered by `tests/test_sft_resume.py` (base_train guard section, passes). |
| 4 | run-record-replay | **Could** | Session plan `~/.claude/plans/with-our-current-setup-tranquil-rossum.md` (not in repo) | NOT STARTED | Add `python3 -m nanochat.run_record replay <run-dir>`. Print `git checkout <code_rev>`, `uv sync --extra cpu` for local runs, and `MODE=<mode> bash runs/runmodal.sh`. The old plan said `--mode`, but `runmodal.sh` reads `MODE` from the environment. Only `nanochat/run_record.py` changes. |
| 5 | cli-learn-color-could-items | **Could** | [cli-learn-color-ux](2026-11-14-cli-learn-color-ux.md) "Could" | NOT STARTED | Colour and define lines for `tok_train`, `tok_eval`, `chat_sft`, `chat_eval`; `chat_cli` welcome plaque. Decide whether `base_train` keeps its `CORE` define call. |
| 6 | train-heartbeat-eta | **Could** | [GROWTH.md](GROWTH.md) fix-forward-mac-phase15 "Deferred" | NOT STARTED | Heartbeat and train-loop ETA. Planned in phase 1.5, never built. Not the same as B3 (lock heartbeat). |
| 7 | review-fix-could-suggestions | **Could** | [pr27-29-review-fixes](2026-11-14-pr27-29-review-fixes.md) S3,S4,N1-N2 | NOT STARTED | S3 (trap-ownership comment in `_run_lock.sh:34-36`), S4 (cap serialised `doc_buffer` in `dataloader.py`). Nits: N1 (explicit `time_due` param in `should_save_checkpoint`), N2 (`conv_buffer` type annotation in `sft_loader.py`). |
| 13 | modal-preemption-resume | **Should** | [resume-all-stages](2026-11-14-resume-all-stages.md) "Modal path not covered" | PARTIAL: manual stop/resume verified; true Modal preemption not verified | Local resume verified (Step A; SIGTERM on MPS). Manual stop+resume on Modal verified 2026-11-17: run `20261117-181526-cpu-d6` stopped via `modal app stop`, relaunched with `NANOCHAT_RESUME_RUN_ID`; log confirmed `Resuming optimization from step 39`; `app.id` updated to new app (PR #57). PR #55 added `NANOCHAT_LOCK_SAME_ID_WAIT_SECS` (default 150 s) to handle stale-heartbeat window after a stop. True Modal-initiated preemption (container suspended by Modal scheduler) still untested. Arch review (2026-11-19) flagged that heartbeat stale ≠ process dead under Modal suspension — see rank 15. |
| 14 | modal-cpu-live-run | **Should** | [HANDOFF.md](../../HANDOFF.md) steps 1 to 8 | DONE (verified 2026-11-17) | Run `20261116-154817-cpu-d6` printed `ALL DONE` (cumulative 1h46m). `.done` marker confirmed on Modal volume (`modal volume ls nanochat-modal cpu/runs/20261116-154817-cpu-d6`). Artifacts downloaded to `~/.cache/nanochat-modal/cpu/`. Local chat and remote Modal chat both answered correctly. Per-run model tag (`model_tag` in `last_run.json`) set correctly. Unblocks rank 8 (modal-cpu-timing-docs). |
| 8 | modal-cpu-timing-docs | **Should** | [MODAL.md](../MODAL.md), [README.md](../../README.md) "Cloud CPU" row | UNBLOCKED (rank 14 done) | README and MODAL.md say CPU timing is "Unmeasured". Run `20261116-154817-cpu-d6` finished in 1h46m cumulative. Measure wall time, stage times and tok/s with `run_record compare` on the pulled run record and put numbers in both docs. Stage events are in `~/.cache/nanochat-modal/cpu/runs/20261116-154817-cpu-d6/events.jsonl`. |
| 9 | modal-pipeline-diagram | **Could** | [MODAL.md](../MODAL.md) "How the pieces fit"; [DIAGRAMS.md](../DIAGRAMS.md) | DONE (aafb840) | Modal data flow and commit handshake added to MODAL.md; training pipeline and skip-decision diagrams in DIAGRAMS.md. Checked against the runners. Checkpoint and resume diagram still planned (DIAGRAMS.md). |
| 10 | agents-md-decision | **Could** | global `~/.claude/CLAUDE.md` (imports `@AGENTS.md`) | DECISION NEEDED | Repo has no `AGENTS.md` or `CLAUDE.md`, so the sync-docs agent rows are not applicable. Decide: add a short `AGENTS.md` pointing to this `docs/plan/` tracker, or leave it out. Leaving it out is the default. |
| 11 | modal-gpu-phase | **Won't (for now)** | [HANDOFF.md](../../HANDOFF.md) | BLOCKED | 8×H100 run (~$40–60, ~2 h). No payment method on the `neomatrix369` Modal workspace. Needs a card and your explicit go-ahead. Do only after rank 2 (B2) lands. |

| 15 | arch-lock-preemption-assumption | **Should** | nw-solution-architect-reviewer pass 2026-11-19 | NOT STARTED | `_run_lock.sh:186–207` reclaims the lock when heartbeat goes stale, assuming the previous holder is dead. Modal preemption can suspend a container (process still alive, heartbeat expires). Lock then gets stolen from a live process, risking data corruption. Fix: add `kill -0 <pid>` check on the previous holder's PID before reclaiming; fail loudly if the PID is still alive and the heartbeat is stale. Requires storing the PID in the lock directory at acquire time. |
| 16 | arch-ssot-run-id-resolution | **Should** | nw-solution-architect-reviewer pass 2026-11-19 | NOT STARTED | Shell `pick_run_id` (`_stage_lib.sh:49–63`) and Python `resolve_cpu_run_id` (`modal_run.py:732–747`) independently decide "which run to continue" — same logic duplicated across two languages; depth-match validation also duplicated (`runcpu.sh:55–57` vs `modal_run.py:722–726`). Risk: implementation drift between local and Modal runs. Fix: extract shared logic into `nanochat/run_resolver.py` with a thin shell wrapper `runs/_resolve_run_id.sh`; update all callers. |
| 17 | arch-lock-adr | **Could** | nw-solution-architect-reviewer pass 2026-11-19 | NOT STARTED | Complex lock timing state machine (60 s heartbeat, 600 s staleness, 150 s same-id-wait) has no design document. Future maintainers cannot validate the thresholds. Create `docs/adr/adr-lock-timing-strategy.md`: state-machine diagram, timeout rationale, failure modes (stale-but-alive, reclaim-race), recovery procedures. |
| 18 | runmodal-run-id-format-validation | **Could** | nw-software-crafter-reviewer pass 2026-11-19 D2 | NOT STARTED | RUN_ID extracted from `resolve_run` output via `sed` with no format check (`runs/runmodal.sh:784`). If Modal CLI output changes or `resolve_run` crashes partway, the parse silently returns empty or garbage. Fix: validate extracted RUN_ID against `^[0-9]{8}-[0-9]{6}-cpu-d[0-9]+$` and exit 1 on mismatch. Add a unit test for malformed `resolve_run` output. |
| 19 | arch-knob-registry | **Could** | nw-solution-architect-reviewer pass 2026-11-19 | NOT STARTED | `NANOCHAT_*` env vars (lock thresholds, eval caps, resume flags, run-id overrides) are scattered across `_run_lock.sh`, `modal_run.py`, `runcpu.sh`, `runmodal.sh` with no single reference. Create `docs/knobs.md` listing every `NANOCHAT_*` var: default, description, usage location. |

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
- PRs #54–#57 merged to master 2026-11-19: stale-lock path docs (#54), same-run-id restart wait (#55), ChatCORE CPU cap (#56), resume app.id fix (#57). Closed ranks 1 (live download verified), 14 (live run complete), 13 (manual stop+resume verified).
- `nw-software-crafter-reviewer` + `nw-solution-architect-reviewer` pass on PRs #54–#57 completed 2026-11-19. Craft: APPROVED (5 minor notes). Architecture: REJECTED_PENDING_REVISIONS (3 blockers). Follow-up items added at ranks 15–19.
