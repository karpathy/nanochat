# Modal lifecycle bugs — fix slices

Identified during docs lifecycle audit (2026-11-15). All three are documented with
workarounds in `docs/MODAL.md` (PR #37, merged). Code fixes are separate slices.

---

## B1: `_download_artifacts()` searches wrong folder

**File:** `runs/modal_run.py:503`
**Severity:** Must — fix before the next paid Modal run.

`_download_artifacts()` lists only the top level of `chatsft_checkpoints/` for
`model_*.pt`. Checkpoints are saved under `chatsft_checkpoints/d<depth>/`
(`scripts/chat_sft.py:166-167`, `nanochat/checkpoint_manager.py:73`), so the top
level holds only `d<depth>/` subdirs and marker files. The check always returns
empty and raises "SFT incomplete: no model checkpoints".

A second risk: `modal volume get` preserves the remote folder name, so a re-download
with `--force` can land in `chatsft_checkpoints/chatsft_checkpoints/`.

**Smallest fix:** download into the parent cache dir, then search recursively for
`model_*.pt`. Function: `_download_artifacts()` in `runs/modal_run.py:492-507`.

**Test:** Bash fixture mock of `modal volume get`; verify checkpoint count in output.
File: new `runs/test_modal_download.sh`.

**Workaround (in docs):** manual `modal volume get nanochat-modal chatsft_checkpoints/d<depth>/ ~/.cache/nanochat-modal/<mode>/chatsft_checkpoints/`.

---

## B2: Stage markers are depth-free — shared across CPU and GPU

**File:** `runs/_stage_lib.sh:109-112, 116-119, 264`
**Severity:** Must — fix before the next GPU run.

Stage markers (`_stage_lib.sh:109, 111`) carry no depth or profile. `last_run.json`
is display-only (`:205-214`). CPU (depth 6) and GPU (depth 24) share the same Modal
Volume path (`modal_run.py:41-42`). A GPU run after a CPU run reports success on the
depth-6 checkpoint — silently wrong.

**Smallest fix:** embed profile + depth in marker filename
(`.stage_${stage}_${profile}_d${depth}`); verify `last_run.json` on skip.
Functions: `stage_done()`, `run_stage()` in `runs/_stage_lib.sh:116-120, 260-280`.

**Test:** Python mock `last_run.json` with `depth: 6`, call `stage_done("base_train")`
with env `PROFILE=gpu DEPTH=24`, assert False. File: new `tests/test_stage_markers.py`.

**Workaround (in docs):** `FORCE_STAGES=all MODE=gpu bash runs/runmodal.sh`.

---

## B3: Stale run lock after cancelled Modal run

**File:** `runs/_run_lock.sh`
**Severity:** Should.

A cancelled Modal run leaves `/data/nanochat-modal/.run.lock` on the Volume.
`_run_lock.sh` uses `kill -0` and `ps -o etimes=` with container-local PIDs; these
mean nothing across container restarts. PID 5 (or any low container PID) in the lock
file never matches, so reclaim logic fires — but that races when multiple containers
try simultaneously. Locks without a start-time field are refused outright
(`_run_lock.sh:88-95`). Error message says `rm -rf`, which doesn't work on Modal Volumes.

**Smallest fix:** replace PID + start-time with run-ID + heartbeat. Stale threshold
600 s. Functions: `_acquire_run_lock()`, `_claim_run_lock()`, + new
`_update_heartbeat()` in `runs/_run_lock.sh`; heartbeat caller in
`runs/modal_run.py:300` (training daemon).

**Test:** Bash: mock time advance with `touch -t` to backdate heartbeat; call
`_acquire_run_lock`; verify exit code. File: new `tests/test_run_lock_modal.sh`.

**Workaround (in docs):** `modal volume rm -r nanochat-modal .run.lock`.
