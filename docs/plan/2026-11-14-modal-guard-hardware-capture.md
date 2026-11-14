# Plan: Modal run safety, runtime docs, and hardware capture

## Context

Starting question: can CPU and GPU Modal runs run at the same time? Answer: not safely. Both mount the shared Volume `nanochat-modal`, so stage markers, archives and commits collide. That led to a launch guard (done, in PR #24), a documentation pass on runtime config (in progress), and a follow-up to record accelerator details in each run (not started). This plan keeps those three workstreams separate so each can be reviewed and merged on its own.

## Status of what exists

| Item | State | Where |
|---|---|---|
| Launch guard in `runmodal.sh` | Done, merged (PRs #24, #25) | branch `fix/runmodal-concurrency-guard`, commits `6f93b3d`, `c927d63` |
| `docs/MODAL.md` "Runtime config and environment" section | Written, **uncommitted** on the same branch's working tree | `docs/MODAL.md` (after Prerequisites) |
| Hardware and software requirements | Trimmed: local install and torch and dtype now link to README instead of repeating it | `docs/MODAL.md` |
| Accelerator capture in run record | Idea only | `nanochat/run_record.py`, `nanochat/memory.py` |
| `/verify-slice` on the guard | `UNKNOWN`: no spec defines exit criteria | n/a |

Open blocker before merge: two `ephemeral` nanochat-speedrun apps are live. The guard will refuse new launches until they stop.

## Workstream A: Launch guard (PR #24)

No code change planned. Remaining steps:
1. Confirm the two live apps are the user's and stop them, or leave them and merge anyway (guard will then block until they end).
2. **Blocker: the guard has no test coverage.** `tests/test_runmodal_mode.py:24` fakes `modal` by writing argv to a file and printing nothing, so `modal app list --json` returns empty output. The JSON parse fails and every test takes the "could not check, proceeding" branch.

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

## Workstream C: Accelerator and memory capture in the run record

Today: `host_info()` in `nanochat/run_record.py` records `machine` and `cores` only. `nanochat/memory.py` already has the detection helpers (`_device_type`, `_system_ram_bytes`, `available_memory_bytes`) used by auto-fit.

Constraint (verified): `nanochat/run_record.py` is documented as stdlib-only so the shell scripts can call `python3 -m nanochat.run_record start` without loading torch. `nanochat/memory.py:16` imports torch at module level. So `run_record.py` cannot import `memory.py` directly.

Proposed change (revised after two rounds of NW reviewer blockers):

**Why `device_type` cannot come from the launch mode (key design finding):**
`runs/runcpu.sh:70-79` does not pass `--device-type`; `scripts/base_train.py:102` calls `autodetect_device_type()` which prefers cuda → mps → cpu. So mode `local-cpu` on Apple Silicon **trains on mps**. Mode is a launcher label, not the actual device. Recording `device_type` from mode would be a false fact. `device_type` is torch-derived and must be written from the training entry, same owner as `accelerator_name`.

**Field-to-owner mapping (the key design constraint):**

| Field | Owner | How |
|---|---|---|
| `ram_total_bytes` | `host_info()` via `hardware.system_ram_bytes()` | stdlib `os.sysconf` — no torch |
| `device_type` | `base_train.py` via `update_run(run_dir, config={"device_type": resolved})` | torch-derived; written after `autodetect_device_type()` resolves at line 102. Fixes the existing gap: `base_train.py:98` snapshots `user_config = vars(args)` *before* line 102 resolves the device, so `config.device_type` is `''` today and `summarize()` always shows `-`. |
| `ram_available_bytes` | `base_train.py` via `update_run()` as a **top-level key** | torch-derived; written after torch is loaded |
| `accelerator_name` | `base_train.py` via `update_run()` as a **top-level key** | torch-derived |
| `accelerator_count` | `base_train.py` via `update_run()` as a **top-level key** | torch-derived |

**Key structure (resolves `update_run()` shallow-merge blocker):** `update_run()` does `run.update(fields)` — a shallow merge that replaces the entire `host` dict if passed `host={...}`. To avoid clobbering the `host` block written by C2: `ram_available_bytes`, `accelerator_name`, and `accelerator_count` are written as direct top-level keys. `device_type` is written via the existing `config` merge path (`update_run(config={"device_type": ...})`), which `summarize()`/`compare_table` already read at `run_record.py:242`.

The `available_memory_bytes` helper in `memory.py` calls `torch.cuda.mem_get_info()` / `torch.mps.recommended_max_memory()` and is **not** stdlib-safe; it must not move to `hardware.py`.

**Changes:**
- Add `nanochat/hardware.py`: **one function only** — `system_ram_bytes() -> int | None` (returns `None` on `sysconf` failure, never the 8 GiB fallback; that fallback stays in `memory.py`'s call site). `memory.py` imports `hardware.system_ram_bytes()` and applies its own 8 GiB fallback; `run_record.py` imports it for `host_info()`.
- Extend `host_info()` (**zero-arg, signature unchanged**) to add `ram_total_bytes` only. `modal_run.py:_available_cores()` calls `host_info()` with no arguments — the signature must not gain a required positional. All device fields belong to the training entry.
- **Named training entry:** `scripts/base_train.py` is the entry that calls `update_run()` after torch is loaded. Write `device_type` into `config` (fixing the existing empty-string gap), and write `ram_available_bytes`, `accelerator_name`, `accelerator_count` as top-level keys. No duplication across the four entry scripts. `chat_sft`, `chat_rl`, and `base_eval` will show `null` accelerator fields until their own entries call `update_run()` — this is accepted scope; `null` is not an error.
- Keep `host_info()` safe: `ram_total_bytes` is `null` when the probe fails; the run never fails.
- Tests go in `tests/test_run_record.py` (no `host_info` test exists today, `nanochat/run_record.py:109`). Pattern: `tests/test_memory.py` monkeypatches attributes. Test `system_ram_bytes()` returning `None` (not 8 GiB) on failure.
- Modal runs already write `cores_configured` and `memory_configured_gib` through `_report_run`. Leave those unchanged.

**Commit split for Workstream C:**
- **C1**: Before extracting, add a test patching `os.sysconf` to raise and asserting `FALLBACK_SYSTEM_RAM_BYTES` is returned — this locks down the behaviour being split. Then extract `hardware.py` + repoint `memory.py`. Update the monkeypatch target in `tests/test_memory.py` deliberately (old target `memory._system_ram_bytes` becomes a wrapper or dead; patch `hardware.system_ram_bytes` instead). The "must pass unchanged" claim from an earlier draft is dropped — the test must be deliberately updated. Run `uv run pytest tests/test_memory.py` before and after to confirm no unintentional regressions.
- **C2**: Extend `host_info()` (zero-arg) + `tests/test_run_record.py` covering `ram_total_bytes` and the `None`-on-failure case for `system_ram_bytes`. No `device_type` in C2 — it ships in C3 only.
- **C3**: `device_type`, `ram_available_bytes`, `accelerator_name`, `accelerator_count` capture in `scripts/base_train.py` + tests. Also update `summarize()` to fall back to `run.get("ram_available_bytes")` and `run.get("accelerator_name")` if those are added to the compare table, so the display is populated from the named-entry values.

Ships in the same PR #24 as the guard, per the user's decision. This widens PR #24 from a runner change to include core `nanochat/` code, so the reviewer should look at `nanochat/run_record.py` and `nanochat/hardware.py` as separate concerns.

## Critical files

- `runs/runmodal.sh`: launch guard (done)
- `runs/modal_run.py`: Modal overrides; `_available_cores()` at line 114 calls `host_info()` with no args — C2 must not add a required positional
- `docs/MODAL.md`: runtime config section (uncommitted)
- `docs/RUNNING.md` §12: batch-size table to link to
- `nanochat/run_record.py` `host_info()`: hardware capture (Workstream C2); `summarize()` at line 242 reads `config.device_type` (Workstream C3)
- `nanochat/memory.py`: helpers to reuse; monkeypatch target changes in C1
- `scripts/base_train.py`: torch-derived fields owner (Workstream C3); `autodetect_device_type()` resolves at line 102, snapshot is at line 98
- `tests/test_runmodal_mode.py`: **do not modify** — existing fake uses `>` and passes; changing it breaks 2 of 3 tests

## Verification

- Commands: `bash -n runs/runmodal.sh`; `uv run pytest` (dev group; pytest is the runner per `pyproject.toml` `[tool.pytest.ini_options]`).
- A: `uv run pytest tests/test_runmodal_guard.py` must pass all seven cases (a, b, c1, c2, d, e, f). Three falsifier drills required: (1) case (a) must fail if the guard's JSON check in `runmodal.sh` is disabled — proves the guard fires; (2) case (e) must fail if the shell swallows the Python non-zero exit status (e.g. by removing the `&&` chain) — proves the warning path is reachable; (3) case (f) must fail with `--json` dropped from the `app list` call — proves the flag is enforced. Restore after each drill. Real CLI refusal with a live app is a manual check, only repeatable while a live app exists.
- B: Every number in the new tables matches its source file. Spot-check against `nanochat/defaults.py`, `runs/modal_run.py`, and `runs/runcpu.sh` (SFT rows). Each relative link resolves (check with `ls` or a grep for the target path). Commit only after the user confirms placement.
- C: `uv run pytest tests/test_run_record.py` covers `host_info()` with the memory helper patched to return values and to return nothing. Short local run with `runs/runcpu.sh`: `run.json` `host` block has the new fields with values. Modal check: run `MODE=cpu` (not GPU, no H100 cost), then assert the named `host` fields are present in the Modal `run.json`. Run only after the user approves the cost line.
- Nothing merges to `master` without the review cycle used for PR #23 and #24.

## Decisions (from the user)

1. Hyperparameter tables live in `docs/MODAL.md`. `docs/RUNNING.md` §12 stays the local reference and is linked.
2. All torch-derived fields (`device_type`, `accelerator_name`, `accelerator_count`, `ram_available_bytes`) are collected in the Python training entry (`base_train.py`). The shell `start` call (`host_info()`) records `ram_total_bytes` only — no mode-derived device type (mode is a launcher label, not actual device).
3. Workstream C ships in PR #24 with the guard. (Recommended against: this mixes a runner fix with core-code changes. Keep the reviewer's concern split in mind.)
4. The `docs/MODAL.md` change is committed to `fix/runmodal-concurrency-guard` so PR #24 documents the new flag.

## Ordered execution steps (after approval)

1. **Workstream A tests**: add `tests/test_runmodal_guard.py` with two separate record files (`app_list_record` and `run_record`) — do not modify `tests/test_runmodal_mode.py`. Implement all seven cases (a, b, c1, c2, d, e, f) as defined above. Run three falsifier drills: (a) goes red with guard's JSON check disabled; (e) goes red when shell swallows Python exit status; (f) goes red with `--json` dropped. Restore after each. Run `uv run pytest tests/test_runmodal_mode.py` after to confirm existing tests still pass.
2. **Workstream B**: commit `docs/MODAL.md` to the branch before requesting PR review so the runtime-config docs are visible on GitHub (commit: `docs(modal): document runtime config and hyperparameters`). Verify every number in the tables against source files; apply the speedrun iterations note (comes from trainer default, not `PROFILE_SPEEDRUN`).
3. **Workstream C1**: before extracting, add a test to `tests/test_memory.py` patching `os.sysconf` to raise and asserting `FALLBACK_SYSTEM_RAM_BYTES` is returned (locks down the behaviour being split). Then extract `hardware.py` (one function: `system_ram_bytes()`) + repoint `memory.py`. Deliberately update the monkeypatch target in `tests/test_memory.py` to `hardware.system_ram_bytes`. Run `uv run pytest tests/test_memory.py` before and after to confirm no unintentional regressions.
4. **Workstream C2**: extend `host_info()` (zero-arg, unchanged signature) + add `tests/test_run_record.py` covering `ram_total_bytes` and the `None`-on-failure case for `system_ram_bytes`. No `device_type` in C2.
5. **Workstream C3**: `device_type` (via `update_run(config={"device_type": resolved})`), `ram_available_bytes`, `accelerator_name`, `accelerator_count` in `scripts/base_train.py` + tests; commit separately. Update `summarize()` to read the now-correct `config.device_type` (and add a test asserting the compare table `device` column is populated for a completed run).
6. Run `uv run pytest` (full suite), push all commits, and update PR #24's body with the test plan, the new doc, and the hardware capture changes.
