# Progress

Delivery ledger for `docs/plan/` slices (non-nWave TRAIL adaptation).

## Decision Log

| date | slice_id | decision | why |
|------|----------|----------|-----|
| 2026-11-13 | fix-forward-mac-phase1 | Ship Phase 1 (echoes/skip/rms_norm) before Phase 2 (defaults/memory auto-fit); freeze echo contract | Phase 2 only adds auto-fit lines; dual-shipping would mix SSOT work with the current Mac train blocker |
| 2026-11-13 | fix-forward-mac-phase1 | Shell owns stage map; Python owns boundary/fail detail; markers written by eval scripts | Keeps orchestration readable in bash and avoids hot-loop echo spam |
| 2026-11-13 | fix-forward-mac-phase1 | Pack `next:` + `wrote:` + `mem:` on one DONE enrichment line | Honors “one enrichment line” discipline while still meeting mem/disk checklist items |
| 2026-11-13 | fix-forward-mac-phase1 | Keep RUNNING.md as single L2 (how-to + echo reference) instead of splitting into three DIVIO files | Plan doc mapping freezes L2 as one journey page; TOC separates modes without extra hops |
| 2026-11-13 | fix-forward-mac-phase1 | Land Phase 1 via PR #4, then PR #5 for nw-review follow-ups (comment + TOC note) | #4 shipped the feature; post-merge review found one stale comment — fixed in a follow-up rather than amending the merged PR |
| 2026-11-13 | fix-forward-mac-phase2 | Python `nanochat/defaults.py` is SSOT; shell gets `runs/profiles.sh` via `python -m nanochat.defaults emit-shell` (checked in) | Avoid dual-truth between bash and Python; Python wins on conflict |
| 2026-11-13 | fix-forward-mac-phase2 | Parallel streams in `.worktrees/p2-{defaults,memory,wire,docs}` on `stream/p2-*`, merge into `feat/fix-forward-mac-phase2` | Independent writers without clobbering the shared slice tip |
| 2026-11-13 | fix-forward-mac-phase2 | Memory fudge `MEMORY_ACTIVATION_FUDGE=1.5`, margin `0.6` / `NANOCHAT_MEMORY_FIT_MARGIN` live in defaults; memory.py imports them | Single knob owner for estimate + docs; env override without code edit |
| 2026-11-13 | fix-forward-mac-phase2 | `PROFILE_RUNCPU.device_batch_size=8` while global argparse default stays 32 | Safer Mac educational default without changing CUDA/speedrun CLI default |
| 2026-11-13 | fix-forward-mac-phase2 | OOM re-halve mid-loop only in `base_train`; eval/SFT get preflight fit only | DDP-safe; avoids silent multi-rank shape drift; documented in RUNNING.md §11 |
| 2026-11-13 | fix-forward-mac-phase2 | Keep third echo variant when batch already 1 and still over budget | Frozen table has shrink/unchanged; claiming "fits" would be false |
| 2026-11-13 | fix-forward-mac-phase2 | chat_sft inherit fallbacks use DEFAULT_SEQ_LEN / DEFAULT_DEVICE_BATCH_SIZE from defaults | Gate 4 SA: avoid dual-truth literals 2048/32 |
| 2026-11-13 | fix-forward-mac-phase2 | After OOM rehalve, train on eager `orig_model` (not compiled `dynamic=False` graph) | Fixed-shape compile cannot survive a mid-run batch change; matches eval/sample pattern |
| 2026-11-13 | fix-forward-mac-phase15 | Pack `purpose:` + `health:` on one START enrichment line | Honors Phase 1 one-enrichment discipline while still shipping AC2 health content |
| 2026-11-13 | fix-forward-mac-phase15 | Trend via fail-soft `emit_metric_trend` (never abort train/eval) | Enrichments are hints; shell already used `\|\| true` — Python needed the same invariant |
| 2026-11-13 | fix-forward-mac-phase15 | Tip text SSOT in `nanochat/run_tip.py`; shell prints via `python -m` | Enables AC7 unit test without brittle bash asserts |
| 2026-11-13 | fix-forward-mac-phase15 | Parallel worktrees `p15-{shell,trend,tests,docs}` then merge to feat branch | Same Phase 2 pattern; independent writers without clobbering tip |
| 2026-11-13 | fix-mps-skip-inductor | Skip `torch.compile` / Inductor on MPS (model + fused optim steps); stay eager | Inductor has no MPS backend; was crashing val eval at step 0 |
| 2026-11-14 | fix-mps-skip-inductor | Gate 4 review: step-0-only announce; compile echo in helper; muon/foreach tests; val-bpb progress non-CUDA only | Reviewers: code-reviewer APPROVE notes; critic+architect CHANGES_REQUESTED (echo firehose, under-tested Muon) |
| 2026-11-14 | modal-nanochat-minimum | SKIP_SETUP always activates `.venv` (no PATH fallback); image must provide `/root/nanochat/.venv` | User decision; fails loudly instead of silently using another Python |
| 2026-11-14 | modal-nanochat-minimum | Modal image uses explicit `uv sync` into `/root/nanochat/.venv`, not `Image.uv_sync` | SDK `uv_sync` puts the venv at `/.uv/.venv` (`_image.py:1981`), which speedrun.sh would not activate |
| 2026-11-14 | modal-nanochat-minimum | Commit failure writes `.modal_commit_failed` and keeps the request file; hook fails fast on it | Review blocker: a failed commit was being acked as success |
| 2026-11-14 | modal-nanochat-minimum | Post-approval follow-ups (retry pacing, idle clock from hook start) merged without re-review | Small, verified locally; flagged so the next reviewer checks `03d0c4d` |
| 2026-11-14 | fix-mps-skip-inductor | Grow RoPE cos/sin on demand (2× buffer) when eval prompts exceed `sequence_len * 10` | CORE few-shot assert `2596 > 2560` at short `max_seq_len=256`; static 10× cache is not enough |
| 2026-11-14 | fix-mps-skip-inductor | Require `.train_ok` for train SKIP; hard-fail `speedrun.sh` without CUDA | Partial `model_*.pt` must not skip train; Mac educational path is `runcpu` only |
| 2026-11-14 | backlog | Softcap / logits memory parked (gather-at-needed-pos vs `PROFILE_RUNCPU` knobs) | Playgroup softcap/eval discussion; no design pick yet — auto-fit remains the Mac OOM mitigator |
| 2026-11-14 | cli-learn-color-ux | Colour and plaques are added around the frozen echo contract; define lines are no-ops when the shell has already briefed the stage | Learners see stage boundaries at a glance without changing the tokens scripts and logs grep for |
| 2026-11-14 | local-resume-and-guard | Local run lock uses an atomic `mkdir` with a PID file, not `flock` | `flock` is not installed on stock macOS |
| 2026-11-14 | local-resume-and-guard | Modal guard stays in `runs/runmodal.sh`; not shared with the local lock | The Modal guard calls the `modal` CLI; sharing it would make `modal` a requirement for local runs |
| 2026-11-14 | resume-all-stages | Do not claim resume is equivalent (or "verified") until the data-position fix passes Step A | Step A found the loader reads `rg: 3` where the uninterrupted run reads `rg: 1`; loss drifts to 0.109 by step 199 |
| 2026-11-14 | resume-all-stages | Work order: data-position fix, then wording, then `chat_sft` resume, then local lock, then daemon cleanup | Must items first; the wording claim depends on the fix; daemon cleanup is Could |
| 2026-11-14 | resume-all-stages | Master warning `70ce9e2` stays open until the user picks revert or force-push | Changing `master` needs the user's choice; revert is the safer default |
| 2026-11-14 | docs | Trackers track one backlog in `TRAIL.md`, ordered by rank with MoSCoW | Stale PR and Step A statuses had hidden open work; one ordered list shows what to do next |
| 2026-11-14 | modal-nanochat-minimum | Modal CPU mode runs first; GPU phase waits for a payment method and a separate go-ahead | HANDOFF: GPU needs billing set up at Modal; the CPU run proves the pipeline for less money |
| 2026-11-14 | modal-guard-hardware-capture | CPU and GPU Modal runs must not run at once; the launch guard blocks it | Both mount the shared Volume `nanochat-modal`, so stage markers, archives and commits collide |
| 2026-11-14 | resume-all-stages | Checkpoint and resume logic has one owner shared by local and Modal runs | Verified in the session plan `can-we-check-if-shimmering-storm`; Modal adds only the commit daemon, so no second copy to keep in sync |
## Slice states

| slice_id | state | evidence |
|----------|-------|----------|
| fix-forward-mac-phase1 | PASSED (merged) | [PR #4](https://github.com/neomatrix369/nanochat/pull/4) merged 2026-11-13; [PR #5](https://github.com/neomatrix369/nanochat/pull/5) follow-ups merged 2026-11-13 (`2aa56ed` on master); pytest 51p/18s; Gate 4 APPROVED (software-crafter after comment fix, acceptance, solution-architect); ddd N/A |
| fix-forward-mac-phase2 | PASSED (merged) | [PR #6](https://github.com/neomatrix369/nanochat/pull/6) merged 2026-11-13 (`f8d7174` on master); Gate 4 APPROVED (software-craft, acceptance, solution-architect); ddd N/A; pytest defaults+memory 58p / suite 109p/18s |
| fix-forward-mac-phase15 | PASSED (merged) | [PR #8](https://github.com/neomatrix369/nanochat/pull/8) merged 2026-11-13 (`3054e4a` on master); Gate 4 APPROVED (code-reviewer, critic, architect); pre-push 126p/18s; phase15 unit 17p; ddd N/A |
| fix-mps-skip-inductor | PASSED (merged) | [PR #10](https://github.com/neomatrix369/nanochat/pull/10) merged 2026-11-14 (`f6ef779` on master); skip Inductor on MPS + MPS AdamW/Muon; progress flush; `.train_ok` + speedrun CUDA hard-fail; RoPE grow-on-demand (`22d1282`); Gate 4 polish |
| modal-nanochat-minimum | MERGED — live run pending | [PR #14](https://github.com/neomatrix369/nanochat/pull/14) merged 2026-11-14 (merge `5225a3d` on master; branch commits `4d60902`…`03d0c4d`). Local verification done: 142 passed / 16 skipped; bash -n and py_compile clean; nWave code, platform and architecture/docs reviewers APPROVED (last commit `03d0c4d` applied their follow-ups after approval, not re-reviewed). **Not verified:** Modal image build, prefetch, 8×H100 train, volume get, `modal app stop` billing stop. Next: see [HANDOFF.md](../../HANDOFF.md). |
| resume-all-stages | MERGED; Step A PASSED 2026-11-15 | [PR #26](https://github.com/neomatrix369/nanochat/pull/26) merged to master. Data-position fix shipped in [PR #33](https://github.com/neomatrix369/nanochat/pull/33) (`97ec941`). Step A verified 2026-11-15: steps 7–199 match uninterrupted run (evidence `docs/evidence/resume-step-a-pass-2026-11-15/`). Modal path not yet covered (TRAIL rank 13). |
| resume-data-position | MERGED (via PR #33) | Fix: `ddp_rank_position` extracted to `nanochat/common.py` as SSOT; used in `dataloader.py` and `sft_loader.py`. Step A passed 2026-11-15. |
| local-resume-and-guard | MERGED (PRs #30, #33) | Run-lock + daemon cleanup in [PR #30](https://github.com/neomatrix369/nanochat/pull/30). `chat_sft` resume, review fixes, `world_size` guard, `apply_sft_resume_state` in [PR #33](https://github.com/neomatrix369/nanochat/pull/33). 303 tests pass. |
| modal-guard-hardware-capture | MERGED (guard) | PRs #24 and #25 merged (`282c84c`, `ffea067`). Docs and accelerator recording open. |
| cli-learn-color-ux | MERGED | PRs #20 to #23 merged to master. |

## Recent merges (Mac fix-forward track)

| When (UTC) | PR | What landed |
|------------|----|-------------|
| 2026-11-15 13:10 | [#35](https://github.com/neomatrix369/nanochat/pull/35) | `SftLoaderState.initial` in `chat_sft`; guard keys read from `user_config` subdict when present; tests for that path |
| 2026-11-15 12:39 | [#34](https://github.com/neomatrix369/nanochat/pull/34) | Run lock stores PID + start time so PID reuse is detected; DDP resume tests |
| 2026-11-15 | [#33](https://github.com/neomatrix369/nanochat/pull/33) | `ddp_rank_position` SSOT in `common.py`; `apply_sft_resume_state` + `assert_resume_config_matches` in `checkpoint_manager.py`; `world_size` guarded on resume; run-lock live-unknown (empty PID) guard; 303 tests pass |
| 2026-11-14 | [#32](https://github.com/neomatrix369/nanochat/pull/32) | Modal CPU pre-flight check and run-watching guidance in `docs/MODAL.md` |
| 2026-11-14 | [#31](https://github.com/neomatrix369/nanochat/pull/31) | Plan doc: nw-crafter review findings for PRs #27–29 |
| 2026-11-14 | [#30](https://github.com/neomatrix369/nanochat/pull/30) | Modal daemon clean SIGTERM + `contextlib.suppress`; concurrency guard tests (`tests/test_runmodal_guard.py`) |
| 2026-11-14 | [#29](https://github.com/neomatrix369/nanochat/pull/29) | `base_train` resume at exact batch via `ddp_rank_position` (data-position fix) |
| 2026-11-14 | [#28](https://github.com/neomatrix369/nanochat/pull/28) | Plan docs: Step A resume result + data-position bug record |
| 2026-11-14 | [#27](https://github.com/neomatrix369/nanochat/pull/27) | Plan docs: MoSCoW pending items, owners, resume evidence |
| 2026-11-14 12:00 | [#14](https://github.com/neomatrix369/nanochat/pull/14) | Modal wrapper (runmodal.sh, modal_run.py, modal_hook.sh); env opt-in for NANOCHAT_BASE_DIR / SKIP_SETUP / post-stage hook; always-activate `.venv`; python3 everywhere; docs/MODAL.md; review fixes |
| 2026-11-14 | [#10](https://github.com/neomatrix369/nanochat/pull/10) | MPS skip Inductor/compile; MPS optim paths; `.train_ok`; speedrun CUDA hard-fail; RoPE grow-on-demand |
| 2026-11-13 16:46 | [#8](https://github.com/neomatrix369/nanochat/pull/8) | Phase 1.5: knob diff, packed health, trend crumb, tip |
| 2026-11-13 16:02 | [#6](https://github.com/neomatrix369/nanochat/pull/6) | Phase 2: defaults SSOT, MPS auto-fit, profiles, RUNNING/README |
| 2026-11-13 15:13 | [#5](https://github.com/neomatrix369/nanochat/pull/5) | Review follow-ups: `tok_eval` marker comment, RUNNING.md TOC note, `.gitignore` chore |
| 2026-11-13 14:46 | [#4](https://github.com/neomatrix369/nanochat/pull/4) | Phase 1: fail-fast echoes/skip, rms_norm fallback, markers, NaN abort, RUNNING.md |
| 2026-11-13 11:48 | [#3](https://github.com/neomatrix369/nanochat/pull/3) | SDPA GQA without `enable_gqa`; skip memory limit on darwin |
| 2026-11-13 11:41 | [#2](https://github.com/neomatrix369/nanochat/pull/2) | Torch select by OS/arch for Intel macOS install |
| 2026-11-13 10:38 | [#1](https://github.com/neomatrix369/nanochat/pull/1) | Platform-aware torch pins for macOS |
