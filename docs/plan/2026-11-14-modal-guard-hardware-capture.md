# Plan: Modal run safety, runtime docs, and hardware capture

## Context

Starting question: can CPU and GPU Modal runs run at the same time? Answer: not safely. Both mount the shared Volume `nanochat-modal`, so stage markers, archives and commits collide. That led to a launch guard (done, in PR #24), a documentation pass on runtime config (in progress), and a follow-up to record accelerator details in each run (not started). This plan keeps those three workstreams separate so each can be reviewed and merged on its own.

## Status of what exists

| Item | State | Where |
|---|---|---|
| Launch guard in `runmodal.sh` | Done, merged (PRs #24, #25) | branch `fix/runmodal-concurrency-guard`, commits `6f93b3d`, `c927d63` |
| `docs/MODAL.md` "Runtime config and environment" section | Written and on master | `docs/MODAL.md` (after Prerequisites) |
| Hardware and software requirements | Trimmed: local install and torch and dtype now link to README instead of repeating it | `docs/MODAL.md` |
| Accelerator capture in run record | Idea only | `nanochat/run_record.py`, `nanochat/memory.py` |
| `/verify-slice` on the guard | `UNKNOWN`: no spec defines exit criteria | n/a |

Open blocker before merge: two `ephemeral` nanochat-speedrun apps are live. The guard will refuse new launches until they stop.

## Workstream A: Launch guard (PR #24) — DONE


No code change planned. Remaining steps:
1. Confirm the two live apps are the user's and stop them, or leave them and merge anyway (guard will then block until they end).
2. **Gap: guard test coverage is partial.** Checked 2026-11-15: `tests/test_runmodal_mode.py` has three guard tests (`test_guard_refuses_when_app_running`, `test_guard_bypassed_with_allow_concurrent`, `test_guard_warns_and_proceeds_when_modal_unavailable`). The plan's seven cases (a–f) and three falsifier drills are not all covered. `tests/test_runmodal_mode.py:24` fakes `modal` by writing argv to a file and printing nothing, so `modal app list --json` returns empty output. The JSON parse fails and every test takes the "could not check, proceeding" branch.

   Fix: add a new fake in `tests/test_runmodal_guard.py` only — do **not** modify `tests/test_runmodal_mode.py:24` (flipping it from `>` to `>>` breaks 2 of its 3 existing tests, proven by measurement). The new fake writes to **two separate files**: `app_list_record` (for `$1 == app` subcommand, written with `>>`) and `run_record` (for all other subcommands, appended with `>>`). `app list` returns a JSON array to stdout and exits 0 by default; test cases that need a different response pass their own fixture variant. The `_run` helper in the new test module must accept an env mapping (case (b) needs `RUNMODAL_ALLOW_CONCURRENT=1`).

   Cases:
   - (a) live `nanochat-speedrun` (`stopped_at: null`): exit non-zero, stderr names `RUNMODAL_ALLOW_CONCURRENT`, `run_record` does not exist. Assert `not run_record.exists()` — not line-based string matching. *(Falsifier — proves the guard fires.)*
   - (b) same plus `RUNMODAL_ALLOW_CONCURRENT=1`: `run_record` exists and contains a `run` line.
   - (c1) only stopped apps (`stopped_at` non-null, using an explicit non-null string not a missing key): `run_record` exists. *(Locks down the `stopped_at is None` filter.)*
   - (c2) only apps with a different description: `run_record` exists. *(Locks down the `description ==` filter. c1/c2 are split so a failure names which filter broke.)*
   - (d) `app list` exits non-zero: warning text on stderr AND `run_record` exists (graceful-degrade contract is explicit, not just implicit from exit 0).
   - (e) `app list` exits 0 but stdout is non-JSON: same warning+launch path as (d). *(Locks down the gap the existing blind spot exposed.)*
   - (f) `assert "--json" in app_list_record.read_text()`. Uses the same happy-path fake as (c1/c2): `app list` returns `[]` and exits 0, `run` appends to `run_record`. Assert `app_list_record` contains `--json`. *(Locks down the `--json` regression: without this, dropping `--json` from the script leaves all cases green.)*

   Edge cases to add: `RUNMODAL_ALLOW_CONCURRENT=""` must still block (the guard uses `-z` which treats empty as unset); `app list` returning a JSON object rather than an array silently fails `a.get()` — this takes the warning path, same as (e).
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

## Critical files

- `runs/runmodal.sh`: launch guard (done)
- `runs/modal_run.py`: Modal overrides; `_available_cores()` at line 114 calls `host_info()` with no args — C2 must not add a required positional
- `docs/MODAL.md`: runtime config section (on master)
- `docs/RUNNING.md` §12: batch-size table to link to
- `tests/test_runmodal_mode.py`: **do not modify** — existing fake uses `>` and passes; changing it breaks 2 of 3 tests

## Verification

- Commands: `bash -n runs/runmodal.sh`; `uv run pytest` (dev group; pytest is the runner per `pyproject.toml` `[tool.pytest.ini_options]`).
- A: the guard cases live in `tests/test_runmodal_mode.py`. Add the seven cases (a, b, c1, c2, d, e, f) there, or in a new file if the existing one is kept unchanged. Three falsifier drills required: (1) case (a) must fail if the guard's JSON check in `runmodal.sh` is disabled — proves the guard fires; (2) case (e) must fail if the shell swallows the Python non-zero exit status (e.g. by removing the `&&` chain) — proves the warning path is reachable; (3) case (f) must fail with `--json` dropped from the `app list` call — proves the flag is enforced. Restore after each drill. Real CLI refusal with a live app is a manual check, only repeatable while a live app exists.
- B: Every number in the new tables matches its source file. Spot-check against `nanochat/defaults.py`, `runs/modal_run.py`, and `runs/runcpu.sh` (SFT rows). Each relative link resolves (check with `ls` or a grep for the target path). Commit only after the user confirms placement.
- Nothing merges to `master` without the review cycle used for PR #23 and #24.

## Decisions (from the user)

1. Hyperparameter tables live in `docs/MODAL.md`. `docs/RUNNING.md` §12 stays the local reference and is linked.
2. The `docs/MODAL.md` change is on master, so the runtime-config docs are already reviewable.

## Ordered execution steps (after approval)

1. **Workstream A tests**: add the seven guard cases to `tests/test_runmodal_mode.py` (or a new file). Include the three falsifier drills.
2. **Workstream B**: done (on master).
3. Run `uv run pytest` (full suite) before closing Workstream A.
