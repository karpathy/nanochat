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
and use the returned model/optimizer (currently thrown away at line 202).
**Effort:** Medium — wire the existing `load_checkpoint` call to actually use its return value.

### B3 — DDP resume reads identical data on all ranks (`nanochat/dataloader.py:25-28`)

`rg_idx = state.get("rg_idx", ddp_rank)` uses the saved value verbatim.
Only rank-0 writes `meta_*.json`, so every rank gets rank-0's row-group index and
advances by `ddp_world_size` → all ranks read identical batches.
The old code re-derived per-rank indices from the single saved value.
Single-process runs (`world_size==1`) are unaffected.

**Fix:** `rg_idx = (saved_rg // ws) * ws + ddp_rank` (and handle `doc_buffer` for `rank != 0`).
**Effort:** Small — restore the derivation that was removed.

### B4 — Config-mismatch tests are circular (`tests/test_sft_resume.py:860-927`)

`TestConfigMismatchGuard` copies the guard loop into the test body and asserts
`pytest.raises` on its **own** `raise ValueError`. Deleting the guard from `chat_sft.py`
leaves all five tests green. This is why B1 (in the same code block) shipped undetected.

**Fix:** Extract the guard into `nanochat/` (e.g. `assert_resume_config_matches(saved_meta, current, keys)`),
call it from `chat_sft.py`, and import + test that function in the test file.
**Effort:** Medium — new module, one import in `chat_sft.py`, update tests.

### B5 — Run lock reclaim is not atomic (`runs/_run_lock.sh:46-56`)

`mv "${_LOCK_DIR}" "${stale_tmp}"` runs unconditionally once the PID looks dead, with no
re-check. Race: P1 reclaims and writes its live PID; P2 (which read the _old_ dead PID
before P1 rewrote it) `mv`s P1's live lock aside and takes the lock. Both hold the lock
simultaneously; P1's EXIT trap deletes P2's directory.

**Fix:** Gate reclaim behind `mkdir "${_LOCK_DIR}.reclaim"` (single atomic claimant);
re-verify the PID after acquiring the real lock.
**Effort:** Medium — rework the reclaim branch in `_run_lock.sh`.

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

Work top to bottom. Each item is independently committable.

| rank | id | MoSCoW | blocker? | effort | notes |
|------|-----|--------|----------|--------|-------|
| 1 | fix-b1-nameerror | **Must** | yes | 1 line | `del _saved` → `del _saved_meta` in `chat_sft.py:216` |
| 2 | fix-s2-base-dir-guard | **Must** | no | 1 line | `${NANOCHAT_BASE_DIR:?}` in `_run_lock.sh:16` — same PR as B1 |
| 3 | fix-b4-extract-guard | **Must** | yes | medium | Extract `assert_resume_config_matches` to `nanochat/`; update test |
| 4 | fix-b2-load-sft-checkpoint | **Must** | yes | medium | Wire `load_checkpoint` return value in resume branch |
| 5 | fix-b3-ddp-rg-derivation | **Must** | yes | small | `rg_idx = (saved_rg // ws) * ws + ddp_rank` |
| 6 | fix-b5-atomic-reclaim | **Should** | yes | medium | Gate reclaim with `mkdir .run.lock.reclaim` |
| 7 | fix-s5-use-initial | **Should** | no | small | Use `SftLoaderState.initial(ddp_rank)` in `chat_sft.py` else-branch |
| 8 | fix-s6-meta-ssot | **Should** | no | small | Read guard keys from `meta["user_config"]`; drop top-level duplicates |
| 9 | fix-s7-ddp-tests | **Should** | no | medium | DDP `world_size=4` test for `_start_position`; SFT snapshot-replay test |
| 10 | fix-s1-pid-reuse | **Should** | no | small | Store PID + start time in lock |
| 11 | fix-s4-doc-buffer | **Could** | no | medium | Cap serialised `doc_buffer` |
| 12 | fix-n1-time-sentinel | **Could** | no | small | Explicit `time_due` param in `should_save_checkpoint` |
| 13 | fix-n2-conv-buffer-type | **Could** | no | 1 line | `conv_buffer: list[list[list[int]]]` |

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
| 6 | review-fix-b5-lock-race | **Should** |
| 11 | review-fix-suggestions (S1, S5–S7) | **Should** |

---

## Exit criteria (all blockers resolved)

- `pytest tests/test_sft_resume.py tests/test_run_lock.py -q` — all pass, no circular assertions
- `scripts/chat_sft.py` resume path: starts from SFT checkpoint weights (not base), restores loader, advances step counter
- Two concurrent `_run_lock.sh` processes: exactly one acquires, the other exits 1 within 1 s
- Stale-PID reclaim: concurrent reclaim by two processes leaves exactly one holder
- DDP `world_size=4` resume: each rank gets a distinct row-group index
- `SftLoaderState.initial()` is the single source of truth for fresh-start field values
