# Plan: Fix blockers from nw-crafter review of PRs #27–#29

## Context

PRs #27–#29 (commits `3f1b67f`–`f5d6548` on `master`) introduced:
- `runs/_run_lock.sh` — atomic mkdir concurrency guard
- `scripts/chat_sft.py` — mid-training SFT resume with `SftLoaderState`
- `nanochat/sft_loader.py` — `SftLoaderState` dataclass
- `nanochat/dataloader.py` — `rg_idx`/`pq_idx` restoration on resume
- `nanochat/checkpoint_manager.py` — `should_save_checkpoint` pure function
- `tests/test_sft_resume.py`, `tests/test_run_lock.py` — associated tests

Review run: `nw-software-crafter-reviewer` agent, 2026-11-14.
Verdict: **CHANGES_REQUESTED** — 5 blockers, 7 suggestions, 2 nits.
Test suite passes (`32 passed`) but blockers live in code paths no test executes.

## Blockers (must fix before merge)

### B1 — NameError on every resume (`scripts/chat_sft.py:216`)

`del _saved, _saved_meta` — `_saved` is never bound (line 202 assigns
`_, _, _saved_meta = load_checkpoint(...)`).
Every `resume_from_step > 0` run crashes before training starts.

**Fix:** `del _saved_meta` (or drop the `del`).
**Effort:** 1 line.

### B2 — Resume restores position but discards SFT weights (`scripts/chat_sft.py:107,169,201-216`)

The resume path restores loader state and step counter. Model weights come from
`load_model("base", …)` and optimizer from `load_optimizer_state("base", …)` — the SFT
checkpoint files (`model_{step}.pt`, `optim_{step}_rank{r}.pt`) are **never loaded**.
All prior SFT training is silently discarded; the run restarts from base weights while
skipping the already-consumed data, making it worse than a clean restart.

**Fix:** When `resume_from_step > 0`, call `load_checkpoint(..., load_optimizer=True, rank=ddp_rank)`
at line 202 and apply the returned state dicts: `orig_model.load_state_dict(model_data)` and
`optimizer.load_state_dict(optimizer_data)`. Place the optimizer load after line 176 (base LR
reset). Guard the `init_lr_frac` scaling at `chat_sft.py:185-188` with
`if args.resume_from_step == 0:` to prevent cumulative LR decay on each crash/resume cycle.
**Effort:** Medium — wire state dicts (not model objects), preserve LR ordering, add resume guard.

### B3 — DDP resume reads identical data on all ranks (`nanochat/dataloader.py:25-28`)

`rg_idx = state.get("rg_idx", ddp_rank)` uses the saved value verbatim.
Only rank-0 writes `meta_*.json`, so every rank gets rank-0's row-group index and
advances by `ddp_world_size` → all ranks read identical batches.
The old code re-derived per-rank indices from the single saved value.
Single-process runs (`world_size==1`) are unaffected.

**Fix:** `rg_idx = (saved_rg // ws) * ws + ddp_rank`. For `rank != 0`, discard the saved
`doc_buffer` on resume — the saved buffer contains rank-0's row group, not rank-r's; let
`refill_buffer` load fresh data from the corrected `rg_idx`. (Note: the old code had a `+1`
to advance past the last-read group; the formula correctly omits it because the saved index
is already the next-unread group.)
**Effort:** Small — restore per-rank index derivation and clear non-zero-rank buffers on resume.

### B4 — Config-mismatch tests are circular (`tests/test_sft_resume.py:183-255`)

`TestConfigMismatchGuard` copies the guard loop into the test body and asserts
`pytest.raises` on its **own** `raise ValueError`. Deleting the guard from `chat_sft.py`
leaves all five tests green. This is why B1 (in the same code block) shipped undetected.
Additionally, `test_no_mismatch_does_not_raise` (`:235-255`) contains no assertions — it
only raises `AssertionError` manually against the same dict it wrote, making it tautological.

**Fix:** Extract the guard into `nanochat/` (e.g. `assert_resume_config_matches(saved_meta, current, keys)`),
call it from `chat_sft.py`, and import + test that function in the test file.
**Effort:** Medium — new module, one import in `chat_sft.py`, update tests.

### B5 — Run lock reclaim is not atomic (`runs/_run_lock.sh:46-56`)

`mv "${_LOCK_DIR}" "${stale_tmp}"` runs unconditionally once the PID looks dead, with no
re-check. Race: P1 reclaims and writes its live PID; P2 (which read the _old_ dead PID
before P1 rewrote it) `mv`s P1's live lock aside and takes the lock. Both hold the lock
simultaneously; P1's EXIT trap deletes P2's directory.

**Fix:** (1) Gate reclaim behind `mkdir "${_LOCK_DIR}.reclaim"` (single atomic claimant).
(2) Re-verify the PID **before** `mv` while `.reclaim` is held — the re-verify must sit between
`mkdir .reclaim` and `mv` to abort if the PID is now alive.
(3) Clean up `${_LOCK_DIR}.reclaim` in `_release_run_lock` or a separate EXIT trap — a process
crash mid-reclaim leaves `.reclaim` permanently, blocking all future reclaims.
(4) Add ownership-checked release: compare `${_LOCK_DIR}/pid` content to `$$` before
`rm -rf "${_LOCK_DIR}"` in `_release_run_lock`.
**Effort:** Medium — rework the reclaim branch in `_run_lock.sh`, add `.reclaim` cleanup to EXIT trap.

---

## Suggestions (address after blockers)

| # | Location | Observation | Priority |
|---|---|---|---|
| S1 | `_run_lock.sh:42-44` | PID reuse: unrelated process inheriting dead holder's PID makes lock un-reclaimable. Store PID + start time. | Should |
| S2 | `_run_lock.sh:12,16` | `NANOCHAT_BASE_DIR=""` → `rm -rf /.run.lock`. Use `${NANOCHAT_BASE_DIR:?must be set}`. | Must |
| S3 | `_run_lock.sh:34-36` | `trap` overwrites any caller trap. Add comment noting trap ownership. | Could |
| S4 | `dataloader.py:97,121` | `doc_buffer` serialised into `meta_*.json` on every save — can be multi-MB per checkpoint. Cap or drop. | Could |
| S5 | `sft_loader.py:28` vs `chat_sft.py:271-275` | `SftLoaderState.initial()` unused in production; fresh-start branch re-states the same 5 values. Use `initial()` in production. | Should |
| S6 | `chat_sft.py:513-516` | Guard keys duplicated as top-level meta alongside `user_config`. Guard reads copies, not `user_config`. Old checkpoints lack keys → guard silently passes on cross-version resume. | Should |
| S7 | test files | Zero tests for DDP resume arithmetic and SFT snapshot-replay. Add `world_size=4` test. | Should |

## Nits

| # | Location | Note |
|---|---|---|
| N1 | `checkpoint_manager.py:187-197` | `inf`/`0.0` broadcast-boolean sentinel is opaque. Consider explicit `time_due: bool` parameter. |
| N2 | `sft_loader.py:16` | `conv_buffer: list` — type as `list[list[list[int]]]`. |

---

## Prioritized work order

Work top to bottom. Ranks 1, 3, and 4 edit overlapping code in the same `if args.resume_from_step > 0:` block (`chat_sft.py:200-216`) and must ship together in one PR. Other items are independently committable.

| rank | id | MoSCoW | blocker? | effort | owner | notes |
|------|-----|--------|----------|--------|-------|-------|
| 1 | fix-b1-nameerror | **Must** | yes | 1 line | crafter | `del _saved` → `del _saved_meta` in `chat_sft.py:216` |
| 2 | fix-s2-base-dir-guard | **Must** | no | 1 line | crafter | `${NANOCHAT_BASE_DIR:?must be set}` on assignment at `_run_lock.sh:12` — same PR as B1 |
| 3 | fix-b4-extract-guard | **Must** | yes | medium | crafter | Extract `assert_resume_config_matches` to `nanochat/`; update test |
| 4 | fix-b2-load-sft-checkpoint | **Must** | yes | medium | crafter | Wire `load_checkpoint` return value in resume branch |
| 5 | fix-b3-ddp-rg-derivation | **Must** | yes | small | crafter | `rg_idx = (saved_rg // ws) * ws + ddp_rank` |
| 6 | fix-b5-atomic-reclaim | **Should** | yes | medium | crafter | Gate reclaim with `mkdir .run.lock.reclaim` |
| 7 | fix-s5-use-initial | **Should** | no | small | crafter | Use `SftLoaderState.initial(ddp_rank)` in `chat_sft.py` else-branch |
| 8 | fix-s6-meta-ssot | **Should** | no | small | crafter | Read guard keys from `meta["user_config"]`; drop top-level duplicates |
| 9 | fix-s7-ddp-tests | **Should** | no | medium | crafter | DDP `world_size=4` test for `_start_position`; SFT snapshot-replay test |
| 10 | fix-s1-pid-reuse | **Should** | no | small | crafter | Store PID + start time in lock |
| 11 | fix-s4-doc-buffer | **Could** | no | medium | crafter | Cap serialised `doc_buffer` |
| 12 | fix-n1-time-sentinel | **Could** | no | small | crafter | Explicit `time_due` param in `should_save_checkpoint` |
| 13 | fix-n2-conv-buffer-type | **Could** | no | 1 line | crafter | `conv_buffer: list[list[list[int]]]` |

---

## Session context (2026-11-14)

### What happened this session

| Action | Detail |
|---|---|
| PR #30 created | Covers TRAIL ranks 12 (modal-guard-tests, `dbd6445`) and 17 (modal-daemon-cleanup, `651d789`) on branch `fix/modal-daemon-cleanup`. Awaiting review. URL: https://github.com/neomatrix369/nanochat/pull/30 |
| nw-crafter review run | `nw-software-crafter-reviewer` reviewed diff of PRs #27–#29 (`3f1b67f`–`f5d6548`). 32 tests pass; 5 blockers found in untested code paths. |
| TRAIL updated | Ranks 1–6 added for review fixes. Existing ranks 1–4 shifted to 7–10. PR #30 status added to ranks 12 and 17. |

### Open decisions still pending (require user input)

| # | Decision | Why blocked |
|---|---|---|
| D1 | Commit `70ce9e2` on `master` by mistake — revert or force-push? | Changing `master` history needs explicit user sign-off. |
| D2 | Two live `nanochat-speedrun` Modal apps (possibly billing). Stop them? | Need `modal app list` output to confirm ownership. |
| D3 | PR #30 (`fix/modal-daemon-cleanup`) review — merge when approved? | Awaiting reviewer approval. |

### TRAIL rank reference (this session's new items)

| TRAIL rank | slice_id | MoSCoW |
|---|---|---|
| 1 | review-fix-b1-nameerror | **Must** |
| 2 | review-fix-s2-base-dir | **Must** |
| 3 | review-fix-b4-guard-extract | **Must** |
| 4 | review-fix-b2-sft-weights | **Must** |
| 5 | review-fix-b3-ddp-sharding | **Must** |
| 6 | review-fix-b5-lock-race | **Must** |
| 11 | review-fix-should-suggestions (S1, S5–S7) | **Should** |
| 22 | review-fix-could-suggestions (S3, S4, N1–N2) | **Could** |

---

## Exit criteria (all Must-level fixes resolved)

- `pytest tests/test_sft_resume.py tests/test_run_lock.py -q` — all pass, no circular assertions
- `scripts/chat_sft.py` resume path: starts from SFT checkpoint weights (not base), restores loader, advances step counter
- Two concurrent `_run_lock.sh` processes: exactly one acquires, the other exits 1 within 1 s
- Stale-PID reclaim: concurrent reclaim by two processes leaves exactly one holder
- (Should) DDP `world_size=4` resume: each rank gets a distinct row-group index
- (Should) `SftLoaderState.initial()` is the single source of truth for fresh-start field values
