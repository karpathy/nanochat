# TRAIL

## Backlog (priority order, 2026-11-14)

Work top to bottom. MoSCoW from the plans. A row moves to the history table below once merged or closed.

| rank | slice_id | MoSCoW | plan | state | notes |
|------|----------|--------|------|-------|-------|
| 1 | resume-data-position | **Must** | [resume-all-stages](2026-11-14-resume-all-stages.md) "Prioritized work order" #1 | IN PROGRESS (uncommitted) | Branch `fix/resume-data-position`, uncommitted change in `nanochat/dataloader.py`. Restore `rg_idx`/`pq_idx` on resume. Pass = Step A control (`rg` and `epoch`/`pq` match the uninterrupted run; loss within a stated tolerance). Shared path: needs review before merge. Evidence: `docs/evidence/resume-step-a-2026-11-14/`. |
| 2 | resume-wording-correction | **Must** | [resume-all-stages](2026-11-14-resume-all-stages.md) #2 | BLOCKED by rank 1 | Fix "verified" wording in `docs/RUNNING.md` and `resume-all-stages.md`. Say the restart picks the right checkpoint and LR schedule; data order is not restored until rank 1 passes. |
| 3 | master-warning-70ce9e2 | **Must** | [resume-all-stages](2026-11-14-resume-all-stages.md) "Housekeeping" | DECISION PENDING (user) | Commit `70ce9e2` (MoSCoW plan) landed on `master` and `origin/master` by mistake. Choose: revert commit (safer) or force-push. Do not touch `master` until chosen. |
| 4 | live-modal-apps-check | **Must** | [modal-guard-hardware-capture](2026-11-14-modal-guard-hardware-capture.md) "Open blocker" | UNVERIFIED | Two live `nanochat-speedrun` apps (`stopped_at: null`) were seen. Confirm they are the user's and stop them, or the guard refuses launches and they keep billing. |
| 5 | chat-sft-resume | **Must** | [local-resume-and-guard](2026-11-14-local-resume-and-guard.md) Step B | NOT STARTED (test interview first) | Idempotent, recoverable `chat_sft` resume. Round-trip test first, then config refusal. One question at a time before any test code. Owner: parity session (`e7db86`). |
| 6 | local-run-lock | **Should** | [local-resume-and-guard](2026-11-14-local-resume-and-guard.md) Step C | UNCLAIMED | `runs/_run_lock.sh`: atomic `mkdir` + PID file (not `flock`). Second run is refused before any download. Stale PID lock is reclaimed. Needs test interview before tests. |
| 7 | modal-guard-tests | **Should** | [modal-guard-hardware-capture](2026-11-14-modal-guard-hardware-capture.md) Workstream A | PLAN APPROVED (waiting for go-ahead to switch to `fix/runmodal-concurrency-guard`) | Cases a, b, c1, c2, d, e, f in `tests/test_runmodal_guard.py`. Three falsifier drills. Do not edit `tests/test_runmodal_mode.py:24`. |
| 8 | modal-preemption-resume | **Should** | [resume-all-stages](2026-11-14-resume-all-stages.md) "Modal path not covered" | UNCLAIMED | Modal preemption path not verified. Needed before any Modal resume claim. Includes the commit gap (SIGTERM before `vol.commit()`). |
| 9 | modal-nanochat-minimum-rereview | **Should** | [modal-nanochat-minimum](2026-11-14-modal-nanochat-minimum.md) | UNCLAIMED | Commit `03d0c4d` applied reviewer follow-ups after approval and was not re-reviewed. Run the reviewer pass. |
| 10 | modal-docs-runtime-config | **Should** | [modal-guard-hardware-capture](2026-11-14-modal-guard-hardware-capture.md) Workstream B | IN PROGRESS | `docs/MODAL.md` "Runtime config and environment" section. Check it is committed, not just in the working tree. Link to RUNNING.md §12 for batch-size table. |
| 11 | graphiti-stepA-save | **Should** | [resume-all-stages](2026-11-14-resume-all-stages.md) "Housekeeping" | WAITING (Graphiti down) | Save the Step A outcome to Graphiti once it is back online. |
| 12 | modal-daemon-cleanup | **Could** | [local-resume-and-guard](2026-11-14-local-resume-and-guard.md) "Modal refactor" | UNCLAIMED | Tidy `runs/modal_run.py` commit daemon and SIGTERM handling. No behaviour change. Last, only if time allows. |
| 13 | accelerator-capture | **Could** | [modal-guard-hardware-capture](2026-11-14-modal-guard-hardware-capture.md) Workstream C | IDEA ONLY | Record accelerator and RAM in `run.json`. C1: pin `system_ram_bytes` behaviour with a test, then extract `nanochat/hardware.py`. C2: add `ram_total_bytes` to `host_info()` (zero-arg). Device fields come from the training entry, not the launcher mode. |
| 14 | run-record-replay | **Could** | Session plan `~/.claude/plans/with-our-current-setup-tranquil-rossum.md` (not in repo) | UNCLAIMED | Add `python3 -m nanochat.run_record replay <run-dir>` that prints `git checkout`, `uv sync`, and the `runmodal.sh --mode` line. Only `nanochat/run_record.py` changes. |
| 15 | cli-learn-color-could-items | **Could** | [cli-learn-color-ux](2026-11-14-cli-learn-color-ux.md) "Could" | UNCLAIMED | Colour and define lines for `tok_train`, `tok_eval`, `chat_sft`, `chat_eval`; `chat_cli` welcome plaque. Decide whether `base_train` keeps its `CORE` define call. |

### Won't (this iteration)

| slice_id | reason |
|----------|--------|
| live-modal-run | 8×H100 train, volume get, billing stop: about $40–60. Needs explicit go-ahead. |
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
| modal-nanochat-minimum | [2026-11-14-modal-nanochat-minimum.md](2026-11-14-modal-nanochat-minimum.md) | MERGED; live run pending (see rank 4 and Won't) | [PR #14](https://github.com/neomatrix369/nanochat/pull/14) merged `5225a3d`. Live Modal run not executed. |
| resume-all-stages | [2026-11-14-resume-all-stages.md](2026-11-14-resume-all-stages.md) | MERGED (`c9cf80e`); resume NOT equivalent yet | [PR #26](https://github.com/neomatrix369/nanochat/pull/26) merged to master. `base_train` mid-stage resume, time-based save, atomic writes. Step A done 2026-11-14: data position wrong (`rg` 3 vs 1). Fix is rank 1. |
| local-resume-and-guard | [2026-11-14-local-resume-and-guard.md](2026-11-14-local-resume-and-guard.md) | DECIDED (not started) | Plan approved 2026-11-14. Steps B, C and guard tests are ranks 5, 6, 7. |
| modal-guard-hardware-capture | [2026-11-14-modal-guard-hardware-capture.md](2026-11-14-modal-guard-hardware-capture.md) | MERGED (guard); docs and accelerator open | [PR #24](https://github.com/neomatrix369/nanochat/pull/24) and [PR #25](https://github.com/neomatrix369/nanochat/pull/25) merged (`282c84c`, `ffea067`). Ranks 7, 10, 13. |
| cli-learn-color-ux | [2026-11-14-cli-learn-color-ux.md](2026-11-14-cli-learn-color-ux.md) | MERGED | PRs [#20](https://github.com/neomatrix369/nanochat/pull/20) to [#23](https://github.com/neomatrix369/nanochat/pull/23) merged to master (`edfb3d3`, `9153d28`, `cf49fab`, `993962f`). Could items at rank 15. |
