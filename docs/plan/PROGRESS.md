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
| 2026-11-14 | fix-mps-skip-inductor | Grow RoPE cos/sin on demand (2× buffer) when eval prompts exceed `sequence_len * 10` | CORE few-shot assert `2596 > 2560` at short `max_seq_len=256`; static 10× cache is not enough |
| 2026-11-14 | fix-mps-skip-inductor | Require `.train_ok` for train SKIP; hard-fail `speedrun.sh` without CUDA | Partial `model_*.pt` must not skip train; Mac educational path is `runcpu` only |
| 2026-11-14 | backlog | Softcap / logits memory parked (gather-at-needed-pos vs `PROFILE_RUNCPU` knobs) | Playgroup softcap/eval discussion; no design pick yet — auto-fit remains the Mac OOM mitigator |

## Slice states

| slice_id | state | evidence |
|----------|-------|----------|
| fix-forward-mac-phase1 | PASSED (merged) | [PR #4](https://github.com/neomatrix369/nanochat/pull/4) merged 2026-11-13; [PR #5](https://github.com/neomatrix369/nanochat/pull/5) follow-ups merged 2026-11-13 (`2aa56ed` on master); pytest 51p/18s; Gate 4 APPROVED (software-crafter after comment fix, acceptance, solution-architect); ddd N/A |
| fix-forward-mac-phase2 | PASSED (merged) | [PR #6](https://github.com/neomatrix369/nanochat/pull/6) merged 2026-11-13 (`f8d7174` on master); Gate 4 APPROVED (software-craft, acceptance, solution-architect); ddd N/A; pytest defaults+memory 58p / suite 109p/18s |
| fix-forward-mac-phase15 | PASSED (merged) | [PR #8](https://github.com/neomatrix369/nanochat/pull/8) merged 2026-11-13 (`3054e4a` on master); Gate 4 APPROVED (code-reviewer, critic, architect); pre-push 126p/18s; phase15 unit 17p; ddd N/A |
| fix-mps-skip-inductor | PASSED (merged) | [PR #10](https://github.com/neomatrix369/nanochat/pull/10) merged 2026-11-14 (`f6ef779` on master); skip Inductor on MPS + MPS AdamW/Muon; progress flush; `.train_ok` + speedrun CUDA hard-fail; RoPE grow-on-demand (`22d1282`); Gate 4 polish |

## Recent merges (Mac fix-forward track)

| When (UTC) | PR | What landed |
|------------|----|-------------|
| 2026-11-14 | [#10](https://github.com/neomatrix369/nanochat/pull/10) | MPS skip Inductor/compile; MPS optim paths; `.train_ok`; speedrun CUDA hard-fail; RoPE grow-on-demand |
| 2026-11-13 16:46 | [#8](https://github.com/neomatrix369/nanochat/pull/8) | Phase 1.5: knob diff, packed health, trend crumb, tip |
| 2026-11-13 16:02 | [#6](https://github.com/neomatrix369/nanochat/pull/6) | Phase 2: defaults SSOT, MPS auto-fit, profiles, RUNNING/README |
| 2026-11-13 15:13 | [#5](https://github.com/neomatrix369/nanochat/pull/5) | Review follow-ups: `tok_eval` marker comment, RUNNING.md TOC note, `.gitignore` chore |
| 2026-11-13 14:46 | [#4](https://github.com/neomatrix369/nanochat/pull/4) | Phase 1: fail-fast echoes/skip, rms_norm fallback, markers, NaN abort, RUNNING.md |
| 2026-11-13 11:48 | [#3](https://github.com/neomatrix369/nanochat/pull/3) | SDPA GQA without `enable_gqa`; skip memory limit on darwin |
| 2026-11-13 11:41 | [#2](https://github.com/neomatrix369/nanochat/pull/2) | Torch select by OS/arch for Intel macOS install |
| 2026-11-13 10:38 | [#1](https://github.com/neomatrix369/nanochat/pull/1) | Platform-aware torch pins for macOS |
