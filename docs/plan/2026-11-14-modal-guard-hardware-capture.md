# Plan: Modal run safety, runtime docs, and hardware capture

## Context

Starting question: can CPU and GPU Modal runs run at the same time? Answer: not safely. Both mount the shared Volume `nanochat-modal`, so stage markers, archives and commits collide. That led to a launch guard (done, in PR #24), a documentation pass on runtime config (in progress), and a follow-up to record accelerator details in each run (not started). This plan keeps those three workstreams separate so each can be reviewed and merged on its own.

## Status of what exists

| Item | State | Where |
|---|---|---|
| Launch guard in `runmodal.sh` | Done, pushed, PR #24 open | branch `fix/runmodal-concurrency-guard`, commits `6f93b3d`, `c927d63` |
| `docs/MODAL.md` "Runtime config and environment" section | Written, **uncommitted** on the same branch's working tree | `docs/MODAL.md` (after Prerequisites) |
| Hardware and software requirements | Trimmed: local install and torch and dtype now link to README instead of repeating it | `docs/MODAL.md` |
| Accelerator capture in run record | Idea only | `nanochat/run_record.py`, `nanochat/memory.py` |
| `/verify-slice` on the guard | `UNKNOWN`: no spec defines exit criteria | n/a |

Open blocker before merge: two `ephemeral` nanochat-speedrun apps are live. The guard will refuse new launches until they stop.

## Workstream A: Launch guard (PR #24)

No code change planned. Remaining steps:
1. Confirm the two live apps are the user's and stop them, or leave them and merge anyway (guard will then block until they end).
2. **Blocker: the guard has no test coverage.** `tests/test_runmodal_mode.py:24` fakes `modal` by writing argv to a file and printing nothing, so `modal app list --json` returns empty output. The JSON parse fails and every test takes the "could not check, proceeding" branch. Fix: add a second fake that branches on `$1 == app` and prints a JSON array to stdout; for all other subcommands (`run`, `volume`, …) append `$@` to a record file. Put the guard tests in a new `tests/test_runmodal_guard.py`, not in the existing file. Use `>>` (append) not `>` so the record survives the `app list` call that fires first. Cases:
   - (a) live `nanochat-speedrun` (`stopped_at: null`): exit non-zero, stderr names `RUNMODAL_ALLOW_CONCURRENT`, no `modal run` line in the record. *(Falsifier — proves the guard fires.)*
   - (b) same plus `RUNMODAL_ALLOW_CONCURRENT=1`: launches (a `run` line appears in the record).
   - (c1) only stopped apps (`stopped_at` non-null): launches.
   - (c2) only apps with a different description: launches. *(c1/c2 are split so a failure names which filter broke.)*
   - (d) `app list` exits non-zero: warning on stderr, launches — assert both the warning text and that a `run` line is in the record (graceful-degrade contract is explicit, not just implicit from exit 0).
   - (e) `app list` exits 0 but stdout is non-JSON: same warning+launch path. *(Locks down the gap the existing blind spot exposed.)*
   - (f) the `app list` argv recorded by the fake must contain `--json`. *(Locks down the `--json` regression: without this, dropping `--json` from the script leaves all cases green.)*
3. Known limit kept out of scope: not atomic. Two launches within seconds can both pass. Volume lock is a follow-up.

## Workstream B: Runtime config docs (`docs/MODAL.md`)

Scope: hyperparameters the scripts use, env vars, and a short pointer for hardware and software. Versions stay out (per the user's request). Values come from the code, not memory:

- Profile values: `nanochat/defaults.py` (`PROFILE_RUNCPU`, `PROFILE_SPEEDRUN`), emitted to `runs/profiles.sh`. Do not hand-edit `profiles.sh`.
- Modal overrides: top of `runs/modal_run.py` (device batch 32, eval batch 16, 5000 iterations, 64 cores, 128 GiB, timeouts, shard counts).
- Hardcoded in scripts: SFT iterations 1500 and eval every 200 in `runs/runcpu.sh`; `torchrun --nproc_per_node=8`, `--fp8`, `OMP_NUM_THREADS=1` in `runs/speedrun.sh`.
- Hardware and software: link to README `Setup` and `Precision / dtype`. Keep only Modal-specific items: Python 3.11+ for the local `modal` CLI, `modal` on PATH, network and screen or tmux, disk under `~/.cache/nanochat-modal`.

Reviewer findings to carry into the doc:
- Base-train eval interval (100, from `PROFILE_RUNCPU`) and SFT eval interval (200, hardcoded in `runcpu.sh`) must be separate rows. The current draft already separates them.
- `PROFILE_SPEEDRUN` sets no `num_iterations`. The doc should say the speedrun iteration count comes from the script and the trainer default, not the profile.
- Verification B must also spot-check `runs/runcpu.sh` for the SFT row.

Known inconsistency to note in the doc, not fix here: the Modal CPU device batch (32) differs from `PROFILE_RUNCPU` (8). `docs/RUNNING.md` §12 already explains the batch-size table; the new doc should link to it instead of restating it.

Decision (confirmed by user): keep hyperparameter tables in `docs/MODAL.md`. `docs/RUNNING.md` §12 stays the local reference and is linked.

## Workstream C: Accelerator and memory capture in the run record

Today: `host_info()` in `nanochat/run_record.py` records `machine` and `cores` only. `nanochat/memory.py` already has the detection helpers (`_device_type`, `_system_ram_bytes`, `available_memory_bytes`) used by auto-fit.

Constraint (verified): `nanochat/run_record.py` is documented as stdlib-only so the shell scripts can call `python3 -m nanochat.run_record start` without loading torch. `nanochat/memory.py:16` imports torch at module level. So `run_record.py` cannot import `memory.py` directly.

Proposed change (revised after NW reviewer blockers):

**Field-to-owner mapping (the key design constraint):**

| Field | Owner | How |
|---|---|---|
| `ram_total_bytes` | `host_info()` via `hardware.system_ram_bytes()` | stdlib `os.sysconf` — no torch |
| `device_type` | `host_info()` via `hardware.device_type_from_mode(mode)` | derived from the `--mode` arg already passed to `start_run` (e.g. `"local-cpu"` → `"cpu"`, `"modal-gpu"` → `"cuda"`) — no torch |
| `ram_available_bytes` | training entry via `update_run()` | torch-derived; written after torch is loaded |
| `accelerator_name` | training entry via `update_run()` | torch-derived |
| `accelerator_count` | training entry via `update_run()` | torch-derived |

The `available_memory_bytes` helper in `memory.py` calls `torch.cuda.mem_get_info()` / `torch.mps.recommended_max_memory()` and is **not** stdlib-safe; it must not move to `hardware.py`.

**Changes:**
- Add `nanochat/hardware.py`: two functions only — `system_ram_bytes() -> int | None` (returns `None` on `sysconf` failure, never the 8 GiB fallback; that fallback lives in `memory.py`'s call site) and `device_type_from_mode(mode: str) -> str` (maps mode string to device family without torch). One stdlib-safe owner for both. `memory.py` calls `hardware.system_ram_bytes()` and applies its own 8 GiB fallback; `run_record.py` calls both.
- Extend `host_info(mode)` to add `ram_total_bytes` and `device_type`. All other new fields (`ram_available_bytes`, `accelerator_name`, `accelerator_count`) start as `null` and are filled by one named training entry.
- **Named training entry:** `scripts/base_train.py` is the entry that calls `update_run()` after torch is loaded. Accelerator and available-RAM fields are written there only — no duplication across the four entry scripts.
- Keep `host_info()` safe: fields are `null` when the probe fails; the run never fails.
- Tests go in `tests/test_run_record.py` (no `host_info` test exists today, `nanochat/run_record.py:109`). Pattern: `tests/test_memory.py` monkeypatches attributes. Test `system_ram_bytes()` returning `None` (not 8 GiB) on failure.
- Modal runs already write `cores_configured` and `memory_configured_gib` through `_report_run`. Leave those unchanged.

**Commit split for Workstream C:**
- **C1**: Extract `hardware.py` + repoint `memory.py`. Run `tests/test_memory.py` before and after — it must pass unchanged (preservation baseline).
- **C2**: Extend `host_info(mode)` + `tests/test_run_record.py` covering both the RAM and device-type fields.
- **C3**: Accelerator and available-RAM capture in `scripts/base_train.py` + tests.

Ships in the same PR #24 as the guard, per the user's decision. This widens PR #24 from a runner change to include core `nanochat/` code, so the reviewer should look at `nanochat/run_record.py` and `nanochat/hardware.py` as separate concerns.

## Critical files

- `runs/runmodal.sh`: launch guard (done)
- `runs/modal_run.py`: Modal overrides (read-only for this plan)
- `docs/MODAL.md`: runtime config section (uncommitted)
- `docs/RUNNING.md` §12: batch-size table to link to
- `nanochat/run_record.py` `host_info()`: hardware capture (Workstream C)
- `nanochat/memory.py`: helpers to reuse (Workstream C)
- `tests/test_runmodal_mode.py`: guard coverage gap (Workstream A)

## Verification

- Commands: `bash -n runs/runmodal.sh`; `uv run pytest` (dev group; pytest is the runner per `pyproject.toml` `[tool.pytest.ini_options]`).
- A: `uv run pytest tests/test_runmodal_guard.py` must pass all four cases above. Case (a) must fail if the JSON check is removed (run once with the check disabled to confirm it goes red). Real CLI refusal with a live app is a manual check, and only repeatable while a live app exists.
- B: Every number in the new tables matches its source file. Spot-check against `nanochat/defaults.py`, `runs/modal_run.py`, and `runs/runcpu.sh` (SFT rows). Each relative link resolves (check with `ls` or a grep for the target path). Commit only after the user confirms placement.
- C: `uv run pytest tests/test_run_record.py` covers `host_info()` with the memory helper patched to return values and to return nothing. Short local run with `runs/runcpu.sh`: `run.json` `host` block has the new fields with values. Modal check: run `MODE=cpu` (not GPU, no H100 cost), then assert the named `host` fields are present in the Modal `run.json`. Run only after the user approves the cost line.
- Nothing merges to `master` without the review cycle used for PR #23 and #24.

## Decisions (from the user)

1. Hyperparameter tables live in `docs/MODAL.md`. `docs/RUNNING.md` §12 stays the local reference and is linked.
2. Accelerator name and count are collected in the Python training entry, where torch is loaded. The shell `start` call records RAM and device type only.
3. Workstream C ships in PR #24 with the guard. (Recommended against: this mixes a runner fix with core-code changes. Keep the reviewer's concern split in mind.)
4. The `docs/MODAL.md` change is committed to `fix/runmodal-concurrency-guard` so PR #24 documents the new flag.

## Ordered execution steps (after approval)

1. **Workstream A tests**: add `tests/test_runmodal_guard.py` (cases a, b, c1, c2, d, e, f as defined above). Confirm case (a) goes red with the JSON check in `runmodal.sh` disabled, then restore it. Confirm case (f) goes red with `--json` dropped from the `app list` call, then restore it.
2. **Workstream B**: commit `docs/MODAL.md` to the branch before requesting PR review so the runtime-config docs are visible on GitHub (commit: `docs(modal): document runtime config and hyperparameters`). Verify every number in the tables against source files; apply the speedrun iterations note (comes from trainer default, not `PROFILE_SPEEDRUN`).
3. **Workstream C1**: extract `hardware.py` + repoint `memory.py`. Run `uv run pytest tests/test_memory.py` before and after — must pass unchanged.
4. **Workstream C2**: extend `host_info(mode)` + add `tests/test_run_record.py` covering `ram_total_bytes` and `device_type`, including the `None`-on-failure case for `system_ram_bytes`.
5. **Workstream C3**: accelerator and available-RAM capture in `scripts/base_train.py` + tests; commit separately.
6. Run `uv run pytest` (full suite), push all commits, and update PR #24's body with the test plan, the new doc, and the hardware capture changes.
