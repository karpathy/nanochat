"""Modal app: nanochat on Modal, in one of two modes.

  --mode cpu (default): runs/runcpu.sh on a heavy CPU machine (no GPU, no payment gate).
  --mode gpu:           runs/speedrun.sh on 8×H100 (needs a GPU payment method).

Both modes: prefetch dataset shards to the Volume, train with wait-for-ack commits,
then download tokenizer + chat checkpoints to the local machine.

CRITICAL: Do not import nanochat.dataset at module level (binds DATA_DIR at import time).
Set NANOCHAT_BASE_DIR in environment before any python -m calls.
"""

import hashlib
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import modal

# modal_stats lives next to this file locally and under /root/nanochat/runs in the container.
# Put both on sys.path so the import works in either place.
for _runs_dir in (os.path.dirname(os.path.abspath(__file__)), "/root/nanochat/runs"):
    if _runs_dir not in sys.path:
        sys.path.insert(0, _runs_dir)

from modal_stats import parse_memory_max, summarize_run

# ─────────────────────────────────────────────────────────────────────────────
# Modal app, Volume and sizing

app = modal.App(name="nanochat-speedrun")

# Shared Volume: mounted at BASE_DIR (= NANOCHAT_BASE_DIR on Modal)
BASE_DIR = "/data/nanochat-modal"
NANOCHAT_MODAL_VOLUME = os.environ.get("NANOCHAT_MODAL_VOLUME", "nanochat-modal")
vol = modal.Volume.from_name(NANOCHAT_MODAL_VOLUME, create_if_missing=True)

PREFETCH_TIMEOUT = 2 * 60 * 60
PREFETCH_SHARDS_GPU = 170  # speedrun.sh needs ~170 shards
PREFETCH_SHARDS_CPU = 8  # runcpu.sh needs 8 shards
GPU_TRAIN_TIMEOUT = 4 * 60 * 60
CPU_TRAIN_TIMEOUT = 12 * 60 * 60  # worst-case bound; billing follows actual run time
# Modal-only runcpu overrides (read by runs/runcpu.sh). Max micro-batch that keeps the 16384-token
# total batch in one pass: 16384 / (32 * 512) = 1, so no grad accumulation.
CPU_DEVICE_BATCH_SIZE = 32
CPU_EVAL_DEVICE_BATCH_SIZE = 16
# ChatCORE samples per categorical task (ARC, MMLU) during SFT. Full sets (-1) take hours on CPU;
# 500 keeps each eval block to minutes. The local default stays -1 (full set) in runcpu.sh.
CPU_CHATCORE_MAX_CAT = 500
CPU_NUM_ITERATIONS = 5000  # profile default; kept for model quality (timeout is set to fit it)
CPU_CORES = 64  # heavy machine: CPU mode is meant to use every core the account allows
CPU_MEMORY_MIB = 131072  # 128 GiB

# GPU functions are only defined in GPU mode. Modal checks every function in the app at
# start-up, so a GPU function in a CPU run fails the whole run (H100 payment gate).
MODAL_MODE = os.environ.get("NANOCHAT_MODAL_MODE", "cpu")


# Must match .python-version and pyproject requires-python. tests/test_modal_image_python.py enforces it.
PYTHON_VERSION = "3.12"


def _nanochat_image(extra: str) -> modal.Image:
    """Image with the uv-synced deps for one extra (gpu or cpu). Local source is the last layer."""
    return (
        modal.Image.debian_slim(python_version=PYTHON_VERSION)
        .pip_install("uv")
        .add_local_file("pyproject.toml", remote_path="/root/nanochat/pyproject.toml", copy=True)
        .add_local_file("uv.lock", remote_path="/root/nanochat/uv.lock", copy=True)
        .run_commands(f"cd /root/nanochat && uv sync --no-install-project --extra {extra} --frozen")
        .env(
            {
                "PATH": "/root/nanochat/.venv/bin:/usr/local/bin:/usr/bin:/bin",
                "PYTHONPATH": "/root/nanochat",
                "NANOCHAT_MODAL_MODE": extra,  # gates GPU function definitions in containers too
            }
        )
        # Local source is the LAST layer: Modal rejects build steps after add_local_*.
        .add_local_dir(
            ".",
            remote_path="/root/nanochat",
            ignore=[".venv", "**/.venv", ".git", ".omc", ".claude", ".worktrees", ".reports", ".test-results", "**/.pytest_cache", "**/__pycache__"],
        )
    )


image = _nanochat_image("gpu")  # GPU mode (speedrun)
image_cpu = _nanochat_image("cpu")  # CPU mode (runcpu), CPU torch wheels, no CUDA

# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers


def _modal_env(base_dir: str) -> dict[str, str]:
    """Build the Modal environment dict for subprocess calls."""
    return {
        "NANOCHAT_BASE_DIR": base_dir,
        "HF_HOME": os.path.join(base_dir, "hf_home"),
        "SKIP_SETUP": "1",
        "NANOCHAT_POST_STAGE_HOOK": "bash /root/nanochat/runs/modal_hook.sh",
        "PYTHONPATH": "/root/nanochat",
        "NO_COLOR": "1",
    }


def _record(run_id: str, **fields: Any) -> None:
    """Merge config or resources into the run's run.json. The container has nanochat on PYTHONPATH."""
    from nanochat.run_record import update_run

    update_run(_run_dir(run_id), **fields)


def _available_cores() -> int:
    """Cores this container may use (respects cgroup/affinity limits, not the host count)."""
    from nanochat.run_record import host_info

    return host_info()["cores"]


# ─────────────────────────────────────────────────────────────────────────────
# Wait-for-ack daemon (runs in background during a training function)


_VOLUME_COMMIT_LOCK = threading.Lock()
PERIODIC_COMMIT_SECS = 300
PERIODIC_COMMIT_RETRY_SECS = 30


def _commit_volume_locked(volume: modal.Volume) -> bool:
    try:
        with _VOLUME_COMMIT_LOCK:
            volume.commit()
        return True
    except Exception as e:  # noqa: BLE001 - a failed periodic commit must not kill the daemon
        print(f"[daemon] ERROR: commit failed: {e}")
        return False


def _periodic_commit(volume: modal.Volume, last_commit: float, now: float, every_secs: float) -> float:
    """Commit the Volume if the interval has passed; return the new last-commit time.

    A failed commit is retried after PERIODIC_COMMIT_RETRY_SECS, not on every loop.
    """
    if now - last_commit < every_secs:
        return last_commit
    if _commit_volume_locked(volume):
        print("[daemon] periodic commit ok")
        return now
    return now - every_secs + PERIODIC_COMMIT_RETRY_SECS


def _commit_daemon(
    volume: modal.Volume,
    base_dir: str,
    every_secs: float,
    stop_event: threading.Event,
) -> None:
    """
    Daemon thread: watch for commit requests, run vol.commit(), update heartbeat.

    Protocol:
    - On .modal_commit_request file: write heartbeat, run commit in worker thread,
      update heartbeat every 10s while commit runs.
    - On success: remove request file.
    - On failure: write .modal_commit_failed and leave request file.
    - Exits its loop when stop_event is set.
    """
    heartbeat_file = os.path.join(base_dir, ".modal_commit_heartbeat")
    request_file = os.path.join(base_dir, ".modal_commit_request")
    failed_file = os.path.join(base_dir, ".modal_commit_failed")
    heartbeat_interval = 10  # seconds between heartbeat updates

    def _write_heartbeat() -> None:
        try:
            with open(heartbeat_file, "w") as f:
                f.write(str(int(time.time())))
        except Exception as e:  # noqa: BLE001 - a heartbeat failure must not kill the daemon
            print(f"[daemon] WARNING: heartbeat write failed: {e}")

    def _commit_worker() -> list[bool]:
        result = [False]
        try:
            with _VOLUME_COMMIT_LOCK:
                volume.commit()
            result[0] = True
        except Exception as e:  # noqa: BLE001 - any commit error must become a failed ack
            print(f"[daemon] ERROR: commit failed: {e}")
        return result

    last_commit = time.time()
    while not stop_event.is_set():
        # Check for request
        if os.path.exists(request_file):
            # Write initial heartbeat and start commit in background
            _write_heartbeat()
            result_box: list[bool] = []

            def _run_commit(out: list[bool]) -> None:
                out.extend(_commit_worker())

            commit_thread = threading.Thread(target=_run_commit, args=(result_box,), daemon=True)
            commit_thread.start()

            # Update heartbeat every 10s until commit finishes
            last_heartbeat = time.time()
            while commit_thread.is_alive():
                time.sleep(1)
                now = time.time()
                if now - last_heartbeat >= heartbeat_interval:
                    _write_heartbeat()
                    last_heartbeat = now

            # Commit finished; check success
            commit_thread.join()
            success = result_box[0] if result_box else False

            if success:
                last_commit = time.time()
                # Remove request on success (hook checks this to know we're done)
                try:
                    os.remove(request_file)
                    print("[daemon] commit succeeded, request removed")
                except FileNotFoundError:
                    pass
            else:
                # Write failure marker and leave request file
                try:
                    with open(failed_file, "w") as f:
                        f.write(str(int(time.time())))
                    print("[daemon] commit failed, marker written")
                except Exception as e:  # noqa: BLE001 - keep the daemon loop alive
                    print(f"[daemon] WARNING: failed to write failure marker: {e}")
                # Rate-limit retries while the hook sees the failed marker, instead of spinning.
                time.sleep(2)
        else:
            last_commit = _periodic_commit(volume, last_commit, time.time(), every_secs)
            time.sleep(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Shared run helpers (used by both modes)


def _run_prefetch(shards: int) -> None:
    """Download ClimbMix shards into the Volume (skip-if-exist), then commit."""
    mode = os.environ.get("NANOCHAT_MODAL_MODE", "cpu")
    base_dir_with_mode = f"{BASE_DIR}/{mode}"
    modal_env = _modal_env(base_dir_with_mode)

    print(f"[prefetch] Starting dataset prefetch ({shards} shards)...")
    result = subprocess.run(
        ["python3", "-m", "nanochat.dataset", "-n", str(shards)],
        cwd="/root/nanochat",
        env={**os.environ, **modal_env},
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"prefetch dataset failed with exit code {result.returncode}")

    print("[prefetch] Dataset complete, committing volume...")
    vol.commit()
    print("[prefetch] Volume committed.")


def _run_record_env(run_id: str, mode: str, code_rev: str) -> dict[str, str]:
    """Env that points a run's record at its Volume folder (run.json and events.jsonl)."""
    return {
        "NANOCHAT_RUN_ID": run_id,
        "NANOCHAT_RUN_MODE": mode,
        "NANOCHAT_RUN_DIR": _run_dir(run_id),
        "NANOCHAT_CODE_REV": code_rev,
    }


def _terminate_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def _run_training(script: str, extra_env: dict[str, str], commit_every_secs: float = PERIODIC_COMMIT_SECS) -> None:
    """Run one stage script with the commit daemon, marker cleanup and final commit."""
    os.chdir("/root/nanochat")

    # Set environment: Modal hooks and stage markers go in Volume, with mode prefix
    mode = os.environ.get("NANOCHAT_MODAL_MODE", "cpu")
    base_dir_with_mode = f"{BASE_DIR}/{mode}"
    modal_env = {**_modal_env(base_dir_with_mode), **extra_env}
    for key, value in modal_env.items():
        os.environ[key] = value

    # Start commit daemon in background
    stop_event = threading.Event()
    daemon_thread = threading.Thread(
        target=_commit_daemon,
        args=(vol, base_dir_with_mode, commit_every_secs, stop_event),
        daemon=True,
    )
    daemon_thread.start()
    print("[train] Commit daemon started")

    # On preemption, stop the child and let the normal teardown below run the final commit.
    # Committing inside the handler raced the commit daemon on the same Volume.
    _proc: list[subprocess.Popen] = []
    stop_requested: list[bool] = []

    def _sigterm_stop_child(signum, frame):
        print("[train] SIGTERM received — stopping training, then committing volume", flush=True)
        stop_requested.append(True)
        if _proc:
            _terminate_group(_proc[0])
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _sigterm_stop_child)
    else:
        print("[train] WARNING: not on the main thread; SIGTERM handler not installed", flush=True)

    print(f"[train] Running {script}...")
    proc = subprocess.Popen(
        ["bash", script],
        cwd="/root/nanochat",
        env={**os.environ, **modal_env},
        start_new_session=True,
    )
    _proc.append(proc)
    if stop_requested:
        _terminate_group(proc)
    returncode = proc.wait()
    stop_event.set()  # signal daemon to stop before cleanup and final commit

    # Clean up hook files BEFORE final commit
    for marker in [".modal_commit_request", ".modal_commit_failed", ".modal_commit_heartbeat"]:
        try:
            os.remove(os.path.join(base_dir_with_mode, marker))
        except FileNotFoundError:
            pass

    # Always commit final state (even if the script failed)
    print(f"[train] {script} finished (exit code {returncode}); final volume commit...")
    with _VOLUME_COMMIT_LOCK:
        vol.commit()

    if returncode != 0:
        raise RuntimeError(f"{script} failed with exit code {returncode}")


def _read_memory_limit() -> int | None:
    """Container memory limit from cgroup v2, or None if unlimited or unreadable."""
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            return parse_memory_max(f.read())
    except OSError:
        return None


def _run_dir(run_id: str) -> str:
    """Volume folder for one run: run.json (record), manifest.json and artifacts/."""
    return os.path.join(BASE_DIR, "runs", run_id)


# Everything a run leaves on the Volume that the next run would overwrite.
ARCHIVE_ITEMS = ("tokenizer", "base_checkpoints", "chatsft_checkpoints", "last_metrics.json", "last_run.json", ".stage_elapsed")
# Per-run checkpoint folders (CPU): only this run's folder is archived, not every past run's.
PER_RUN_ITEMS = ("base_checkpoints", "chatsft_checkpoints")
# A run-id model tag: a sortable timestamp prefix, e.g. 20261116-114442-cpu-d6.
RUN_ID_TAG = re.compile(r"^\d{8}-\d{6}-")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_artifacts(run_id: str, mode: str) -> None:
    """Copy this run's outputs into runs/<run-id>/artifacts and write a manifest with checksums.

    Without this, the next run overwrites checkpoints, metrics and stage markers on the Volume.
    """
    artifacts_dir = os.path.join(_run_dir(run_id), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    manifest = []
    for item in ARCHIVE_ITEMS:
        # CPU runs use the run id as their model tag, so archive just that folder.
        rel = os.path.join(item, run_id) if mode == "cpu" and item in PER_RUN_ITEMS else item
        src = os.path.join(BASE_DIR, mode, rel)
        if not os.path.exists(src):
            continue
        dst = os.path.join(artifacts_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            files = [os.path.join(root, name) for root, _, names in os.walk(dst) for name in names]
        else:
            shutil.copy2(src, dst)
            files = [dst]
        for path in files:
            manifest.append({
                "path": os.path.relpath(path, artifacts_dir),
                "bytes": os.path.getsize(path),
                "sha256": _sha256(path),
            })
    with open(os.path.join(_run_dir(run_id), "manifest.json"), "w") as f:
        json.dump({"run_id": run_id, "files": manifest}, f, indent=2)
    print(f"[archive] {len(manifest)} files archived under runs/{run_id}/artifacts")
    vol.commit()


def _clear_stage_markers(mode: str) -> None:
    """Remove stage-completion markers from the Volume working directories after a successful run.

    Prevents the next run from skipping every stage because the markers from this run are
    still present. Called only on success (after the try/finally in train_cpu/train) so that
    a failed or preempted run still resumes correctly from where it stopped.

    Does NOT remove tokenizer.pkl — tok_train is deterministic and expensive; it stays skipped.
    """
    base = os.path.join(BASE_DIR, mode)
    removed = 0
    # Depth-independent: tok_eval marker
    tok_eval_marker = os.path.join(base, "tokenizer", ".tok_eval_ok")
    try:
        os.remove(tok_eval_marker)
        removed += 1
    except FileNotFoundError:
        pass
    # Depth-sensitive: all .stage_* files in base_checkpoints and chatsft_checkpoints
    for subdir in ("base_checkpoints", "chatsft_checkpoints"):
        d = os.path.join(base, subdir)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.startswith(".stage_"):
                try:
                    os.remove(os.path.join(d, name))
                    removed += 1
                except FileNotFoundError:
                    pass
    if removed:
        vol.commit()
    print(f"[archive] {removed} stage marker(s) cleared from {mode}/ — next run will start fresh.")


def _report_run(started: float, timeout_s: int, run_id: str) -> None:
    """Print and store what the run used vs what was configured, so sizing can be tuned."""
    children = os.times()
    report = summarize_run(
        wall_s=time.time() - started,
        child_cpu_s=children.children_user + children.children_system,
        cores_configured=CPU_CORES,
        peak_rss_kb=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        cgroup_memory_limit_bytes=_read_memory_limit(),
        configured_memory_bytes=CPU_MEMORY_MIB * 1024 * 1024,
        timeout_s=timeout_s,
    )
    print("[report] " + json.dumps(report, indent=2))
    _record(run_id, resources=report)
    vol.commit()


# ─────────────────────────────────────────────────────────────────────────────
# CPU mode: runcpu.sh on a heavy CPU machine (no GPU)


@app.function(timeout=PREFETCH_TIMEOUT, volumes={BASE_DIR: vol}, image=image_cpu)
def prefetch_cpu() -> None:
    """CPU-mode prefetch: 8 shards (what runcpu.sh needs)."""
    _run_prefetch(PREFETCH_SHARDS_CPU)


def _cpu_train_env(
    cores: int, run_id: str, code_rev: str, num_iterations: int, save_every_secs: float
) -> dict[str, str]:
    """Env for runs/runcpu.sh on the Modal CPU container (pure: no I/O, easy to test)."""
    return {
        "OMP_NUM_THREADS": str(cores),
        "MKL_NUM_THREADS": str(cores),
        "NANOCHAT_RUNCPU_DEVICE_BATCH_SIZE": str(CPU_DEVICE_BATCH_SIZE),
        "NANOCHAT_RUNCPU_EVAL_DEVICE_BATCH_SIZE": str(CPU_EVAL_DEVICE_BATCH_SIZE),
        "NANOCHAT_RUNCPU_NUM_ITERATIONS": str(num_iterations),
        "NANOCHAT_RUNCPU_CHATCORE_MAX_CAT": str(CPU_CHATCORE_MAX_CAT),
        "NANOCHAT_SAVE_EVERY_SECS": str(save_every_secs),
        **_run_record_env(run_id, "modal-cpu", code_rev),
    }


@app.function(
    image=image_cpu,
    cpu=CPU_CORES,
    memory=CPU_MEMORY_MIB,
    timeout=CPU_TRAIN_TIMEOUT,
    volumes={BASE_DIR: vol},
)
def train_cpu(
    run_id: str,
    code_rev: str,
    num_iterations: int = CPU_NUM_ITERATIONS,
    commit_every_secs: float = PERIODIC_COMMIT_SECS,
    save_every_secs: float = 300,
) -> None:
    """CPU train: runcpu.sh using every core in this container, same commit protocol."""
    cores = _available_cores()
    _record(run_id, config={"cores": cores, "memory_gib": CPU_MEMORY_MIB / 1024, "timeout_s": CPU_TRAIN_TIMEOUT})
    started = time.time()
    try:
        _run_training(
            "runs/runcpu.sh",
            _cpu_train_env(cores, run_id, code_rev, num_iterations, save_every_secs),
            commit_every_secs,
        )
    finally:
        _archive_artifacts(run_id, "cpu")
        _report_run(started, CPU_TRAIN_TIMEOUT, run_id)


# ─────────────────────────────────────────────────────────────────────────────
# GPU mode: speedrun.sh on 8×H100 (defined only when NANOCHAT_MODAL_MODE=gpu)

if MODAL_MODE == "gpu":

    @app.function(timeout=PREFETCH_TIMEOUT, volumes={BASE_DIR: vol}, image=image)
    def prefetch() -> None:
        """GPU-mode prefetch: 170 shards (what speedrun.sh needs)."""
        _run_prefetch(PREFETCH_SHARDS_GPU)

    @app.function(
        image=image,
        gpu="H100:8",
        timeout=GPU_TRAIN_TIMEOUT,
        volumes={BASE_DIR: vol},
    )
    def train(run_id: str, code_rev: str) -> None:
        """GPU train: speedrun.sh with SKIP_SETUP=1 and wait-for-ack hooks."""
        try:
            _run_training("runs/speedrun.sh", _run_record_env(run_id, "modal-gpu", code_rev))
        finally:
            _archive_artifacts(run_id, "gpu")
        _clear_stage_markers("gpu")


# ─────────────────────────────────────────────────────────────────────────────
# Modal chat: one-shot prompt against the trained model on the Volume (no download)

CHAT_TIMEOUT = 10 * 60  # model load + single-prompt generation


@app.function(
    image=image_cpu,
    timeout=CHAT_TIMEOUT,
    volumes={BASE_DIR: vol},
)
def chat_on_modal(mode: str, prompt: str, temperature: float = 0.3, top_k: int = 50) -> str:
    """Run chat_cli -p on the trained model on the Volume. Returns the response text."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}: use one of {MODES}")
    base_dir = f"{BASE_DIR}/{mode}"
    env = {
        **os.environ,
        **_modal_env(base_dir),
        "PATH": "/root/nanochat/.venv/bin:/usr/local/bin:/usr/bin:/bin",
    }
    result = subprocess.run(
        ["python", "-m", "scripts.chat_cli", "-p", prompt, "-t", str(temperature), "-k", str(top_k)],
        check=False,
        env=env,
        capture_output=True,
        text=True,
        cwd="/root/nanochat",
    )
    if result.returncode != 0:
        stdout_tail = result.stdout[-500:].strip() if result.stdout else "(empty)"
        raise RuntimeError(
            f"chat_cli failed (exit {result.returncode}).\n"
            f"stderr: {result.stderr.strip() or '(empty)'}\n"
            f"stdout tail: {stdout_tail}\n"
            "Hint: ensure the model is trained and the Volume is populated "
            f"(`modal run runs/modal_run.py --mode {mode}`)."
        )
    if not result.stdout.strip():
        raise RuntimeError(
            f"chat_cli exited 0 but produced no output (mode={mode!r}).\n"
            "Hint: check that the Volume contains a trained model for this mode."
        )
    return result.stdout


# ─────────────────────────────────────────────────────────────────────────────
# Local entrypoint: prefetch → train → download → tip

MODES = ("cpu", "gpu")


def _select_chat_files(sizes: dict[str, int], mode: str) -> list[str]:
    """Pick what chat needs from a Volume listing: tokenizer files, the SFT marker,
    and per depth folder the newest complete model_<step>.pt + meta_<step>.json.

    Older checkpoints are skipped: each model file is ~300 MB and chat loads only the newest.
    When run-id tag folders exist (timestamp-prefixed), only the newest run's folder is kept.
    """
    picked = [p for p in sizes if p.startswith(f"{mode}/tokenizer/") or p in (f"{mode}/chatsft_checkpoints/.train_ok", f"{mode}/last_run.json")]
    kinds_by_tag: dict[str, dict[int, set[str]]] = {}
    for path in sizes:
        match = re.fullmatch(rf"{re.escape(mode)}/chatsft_checkpoints/([^/]+)/(model|meta)_(\d+)\.(?:pt|json)", path)
        if match:
            tag, kind, step = match.group(1), match.group(2), int(match.group(3))
            kinds_by_tag.setdefault(tag, {}).setdefault(step, set()).add(kind)
    if not kinds_by_tag:
        raise RuntimeError(f"no model checkpoints under /{mode}/chatsft_checkpoints on the Volume")
    # Run ids start with a sortable timestamp, so the newest run is the largest such name.
    run_tags = [tag for tag in kinds_by_tag if RUN_ID_TAG.match(tag)]
    if run_tags:
        kinds_by_tag = {max(run_tags): kinds_by_tag[max(run_tags)]}
    for tag, kinds_by_step in kinds_by_tag.items():
        complete = [step for step, kinds in kinds_by_step.items() if kinds == {"model", "meta"}]
        if not complete:
            raise RuntimeError(f"no complete model+meta pair in /{mode}/chatsft_checkpoints/{tag}")
        newest = max(complete)
        if newest != max(kinds_by_step):
            print(f"WARNING: newest step in {tag} has no complete model+meta pair; using step {newest}", flush=True)
        picked.append(f"{mode}/chatsft_checkpoints/{tag}/model_{newest:06d}.pt")
        picked.append(f"{mode}/chatsft_checkpoints/{tag}/meta_{newest:06d}.json")
    return sorted(picked)


def _fetch_file(volume: modal.Volume, remote: str, size: int, local: str) -> bool:
    """Copy one Volume file to local. Returns False when a same-size copy already exists.

    Writes to <local>.part and renames only after the size matches, so a dropped
    transfer never leaves a file that looks complete.
    """
    if os.path.isdir(local):
        raise RuntimeError(f"{local} is a folder where a file is expected; move it aside and retry")
    if os.path.isfile(local) and os.path.getsize(local) == size:
        return False
    os.makedirs(os.path.dirname(local), exist_ok=True)
    part = f"{local}.part"
    with open(part, "wb") as out:
        out.writelines(volume.read_file(remote))
    if os.path.getsize(part) != size:
        os.remove(part)
        raise RuntimeError(f"size mismatch for {remote}: expected {size} bytes")
    os.replace(part, local)
    return True


def _verify_chat_files(targets: dict[str, str], sizes: dict[str, int]) -> None:
    """Check every selected file exists locally at its Volume size. targets maps remote path -> local path."""
    bad = [
        remote for remote, local in targets.items()
        if not os.path.isfile(local) or os.path.getsize(local) != sizes[remote]
    ]
    if bad:
        raise RuntimeError("chat files missing or incomplete locally: " + ", ".join(bad))


def _download_artifacts(mode: str) -> None:
    """Pull what chat needs for <mode> from the Volume into ~/.cache/nanochat-modal/<mode>.

    Layout matches NANOCHAT_BASE_DIR: tokenizer/ and chatsft_checkpoints/<model-tag>/ (runcpu: the run id; speedrun: d<depth>).
    """
    cache_dir = os.path.expanduser(f"~/.cache/nanochat-modal/{mode}")
    volume = modal.Volume.from_name(os.environ.get("NANOCHAT_MODAL_VOLUME", "nanochat-modal"))
    sizes = {
        e.path: e.size for e in volume.listdir(mode, recursive=True)
        if e.type == modal.volume.FileEntryType.FILE
    }
    wanted = _select_chat_files(sizes, mode)
    total_bytes = sum(sizes[p] for p in wanted)
    print(f"  → {len(wanted)} files, {total_bytes / 1e6:.0f} MB → {cache_dir}", flush=True)
    copied_bytes, started = 0, time.monotonic()
    targets = {remote: os.path.join(cache_dir, remote[len(mode) + 1:]) for remote in wanted}
    for index, remote in enumerate(wanted, 1):
        fetched = _fetch_file(volume, remote, sizes[remote], targets[remote])
        if fetched:
            copied_bytes += sizes[remote]
        elapsed = time.monotonic() - started
        rate = copied_bytes / elapsed if elapsed > 0 else 0
        remaining = total_bytes - sum(sizes[p] for p in wanted[:index])
        eta = remaining / rate if rate > 0 else 0
        status = "copied" if fetched else "already have"
        print(
            f"    [{index}/{len(wanted)}] {status} {remote} ({sizes[remote] / 1e6:.1f} MB) | "
            f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s",
            flush=True,
        )
    _verify_chat_files(targets, sizes)
    print(f"[local] chat files verified in {cache_dir}", flush=True)


# Default staleness threshold; runs/_run_lock.sh reads the same NANOCHAT_LOCK_STALE_SECS override.
LOCK_STALE_SECS = 600


def _lock_is_fresh(lock_mtime: float | None, now: float, stale_secs: float) -> bool:
    return lock_mtime is not None and now - lock_mtime < stale_secs


LOCK_REFUSAL = (
    "another CPU run is in progress (its heartbeat is fresh). Wait for it to finish, or if its "
    "container is gone, the heartbeat expires on its own; to clear it now run: "
    "modal volume rm -r nanochat-modal cpu/.run.lock"
)


def check_forced_cpu_run_id(
    run_id: str, suffix: str, run_ids: list[str], done_ids: set[str], lock_mtime: float | None, now: float, stale_secs: float,
) -> str:
    """Validate an explicitly named run before resuming it: a timestamped id of this depth that exists,
    is not already finished, and has no live run holding the lock."""
    if not (RUN_ID_TAG.match(run_id) and run_id.endswith(suffix)):
        raise SystemExit(f"{run_id} is not a CPU run id of this depth (expected <timestamp>{suffix}); refusing to resume it")
    if run_id not in run_ids:
        raise SystemExit(f"no run folder named {run_id} on the Volume; check the id (see runs/ on the Volume)")
    if run_id in done_ids:
        raise SystemExit(f"{run_id} already finished (it has a .done marker); it is never resumed")
    if _lock_is_fresh(lock_mtime, now, stale_secs):
        raise SystemExit(LOCK_REFUSAL)
    return run_id


def resolve_cpu_run_id(
    run_ids: list[str], done_ids: set[str], lock_mtime: float | None, now: float, suffix: str,
    stale_secs: float = LOCK_STALE_SECS,
) -> str | None:
    """Run id a CPU relaunch should continue, or None to start a new run.

    Refuses while another run holds a fresh lock. Otherwise continues ONLY the most recent run with
    this suffix (e.g. -cpu-d6), and only if it has no .done marker (interrupted or stopped). Older
    unfinished runs are never resumed automatically: name one explicitly with NANOCHAT_RESUME_RUN_ID.
    """
    if _lock_is_fresh(lock_mtime, now, stale_secs):
        raise SystemExit(LOCK_REFUSAL)
    newest = next(iter(sorted((r for r in run_ids if RUN_ID_TAG.match(r) and r.endswith(suffix)), reverse=True)), None)
    if newest is None or newest in done_ids:
        return None
    return newest


@app.local_entrypoint()
def resolve_run(mode: str = "cpu", depth: int = 6, resume: str = "") -> None:
    """Print RUN_ID=<id> for runs/runmodal.sh (CPU only).

    Without --resume: the most recent interrupted CPU run of this depth, else a new id.
    With --resume <id>: that run, after the same depth and liveness checks.
    """
    if mode != "cpu":
        raise SystemExit("resolve_run is for CPU runs only")
    now = time.time()
    suffix = f"-cpu-d{depth}"
    try:
        run_entries = vol.listdir("runs")
    except modal.exception.NotFoundError:
        run_entries = []
    run_ids = [e.path.split("/")[-1] for e in run_entries if e.type == modal.volume.FileEntryType.DIRECTORY]
    done_ids = {
        r for r in run_ids
        if r.endswith(suffix) and any(e.path.endswith(f"runs/{r}/.done") for e in vol.listdir(f"runs/{r}"))
    }
    try:
        lock_entries = vol.listdir("cpu/.run.lock")
    except modal.exception.NotFoundError:
        lock_entries = []
    lock_mtime = next((e.mtime for e in lock_entries if e.path.endswith("/heartbeat")), None)
    stale_secs = float(os.environ.get("NANOCHAT_LOCK_STALE_SECS", LOCK_STALE_SECS))
    if resume:
        run_id = check_forced_cpu_run_id(resume, suffix, run_ids, done_ids, lock_mtime, now, stale_secs)
    else:
        run_id = resolve_cpu_run_id(run_ids, done_ids, lock_mtime, now, suffix, stale_secs)
    if run_id is None:
        run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime(now)) + suffix
    print(f"RUN_ID={run_id}")


@app.local_entrypoint()
def main(mode: str = "cpu", run_id: str = "", code_rev: str = "unknown") -> None:
    if mode not in MODES:
        raise SystemExit(f"Unknown mode {mode!r}: use --mode cpu or --mode gpu")
    if mode != MODAL_MODE:
        raise SystemExit(f"--mode {mode} needs NANOCHAT_MODAL_MODE={mode}; use runs/runmodal.sh")
    if not run_id:
        raise SystemExit("Pass --run-id (runs/runmodal.sh sets it, along with the archive folder).")
    print(f"[local] run id: {run_id} (code {code_rev})")
    prefetch_fn = prefetch_cpu if mode == "cpu" else prefetch
    if mode == "cpu":
        train_call = train_cpu.remote
        train_args = (run_id, code_rev)
    else:
        train_call = train.remote
        train_args = (run_id, code_rev)
    label = "CPU (runcpu, heavy machine)" if mode == "cpu" else "GPU (speedrun, 8×H100)"
    print("═" * 80)
    print(f"nanochat on Modal, mode={mode}: {label}")
    print("═" * 80)
    print("\n[local] Step 1: prefetch...")
    prefetch_fn.remote()
    print(f"[local] Step 2: train ({mode})...")
    train_call(*train_args)
    print("\nSUCCESS: training complete.")
    print("═" * 80)
    print("\nArtifacts are on the Volume. To download to your laptop, run:")
    print("  bash runs/runmodal.sh finish <run-id>")
    print("\nThen chat locally with:")
    print(f"  NANOCHAT_BASE_DIR=~/.cache/nanochat-modal/{mode} uv run python -m scripts.chat_cli")
    print("\nSee docs/MODAL.md for details.")


@app.local_entrypoint()
def chat(mode: str = "cpu", prompt: str = "", temperature: float = 0.3, top_k: int = 50) -> None:
    """Ask the trained model a question on Modal. The model runs on Modal; nothing is downloaded.

    Usage:
        modal run runs/modal_run.py::chat --prompt "What is the capital of France?"
        modal run runs/modal_run.py::chat --mode gpu --prompt "Why is the sky blue?"
        modal run runs/modal_run.py::chat --temperature 0.1 --prompt "..."
    """
    if not prompt.strip():
        raise SystemExit("Pass --prompt (e.g. --prompt 'What is the capital of France?')")
    if mode not in MODES:
        raise SystemExit(f"Unknown mode {mode!r}: use --mode cpu or --mode gpu")
    if temperature < 0:
        raise SystemExit(f"--temperature must be >= 0 (got {temperature})")
    if top_k < 1:
        raise SystemExit(f"--top-k must be >= 1 (got {top_k})")
    print(f"[local] Running chat on Modal (mode={mode}, temperature={temperature}). No download needed.")
    response = chat_on_modal.remote(mode, prompt, temperature, top_k)
    print(response, end="")


if __name__ == "__main__":
    raise SystemExit("Use: modal run runs/modal_run.py --mode cpu|gpu")
