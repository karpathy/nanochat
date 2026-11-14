#!/usr/bin/env python3

"""Modal app: nanochat on 8×H100.

CPU prefetch: dataset -n 170, commit.
GPU train: speedrun.sh with SKIP_SETUP=1, wait-for-ack hooks, final commit.
Local: modal volume get tokenizer + chatsft_checkpoints, print chat tip.

CRITICAL: Do not import nanochat.dataset at module level (binds DATA_DIR at import time).
Set NANOCHAT_BASE_DIR in environment before any python -m calls.
"""

import os
import subprocess
import threading
import time

import modal

# ─────────────────────────────────────────────────────────────────────────────
# Modal app and image definition

app = modal.App(name="nanochat-speedrun")

# Shared Volume: mounted at /data/nanochat-modal
vol = modal.Volume.from_name("nanochat-modal", create_if_missing=True)

# Base image: debian_slim + python 3.11 + uv + deps + code + .venv
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .add_local_file("pyproject.toml", remote_path="/root/nanochat/pyproject.toml", copy=True)
    .add_local_file("uv.lock", remote_path="/root/nanochat/uv.lock", copy=True)
    .run_commands("cd /root/nanochat && uv sync --no-install-project --extra gpu --frozen")
    .add_local_dir(
        ".",
        remote_path="/root/nanochat",
        ignore=[".venv", ".git", "**/.pytest_cache", "**/__pycache__"],
    )
    .env(
        {
            "PATH": "/root/nanochat/.venv/bin:/usr/local/bin:/usr/bin:/bin",
            "PYTHONPATH": "/root/nanochat",
        }
    )
)

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
    }


# ─────────────────────────────────────────────────────────────────────────────
# Wait-for-ack daemon (runs in background during GPU function)


def _commit_daemon(volume: modal.Volume, base_dir: str) -> None:
    """
    Daemon thread: watch for commit requests, run vol.commit(), update heartbeat.

    Protocol:
    - On .modal_commit_request file: write heartbeat, run commit in worker thread,
      update heartbeat every 10s while commit runs.
    - On success: remove request file.
    - On failure: write .modal_commit_failed and leave request file.
    """
    heartbeat_file = os.path.join(base_dir, ".modal_commit_heartbeat")
    request_file = os.path.join(base_dir, ".modal_commit_request")
    failed_file = os.path.join(base_dir, ".modal_commit_failed")
    heartbeat_interval = 10  # seconds between heartbeat updates

    def _write_heartbeat() -> None:
        try:
            with open(heartbeat_file, "w") as f:
                f.write(str(int(time.time())))
        except Exception as e:
            print(f"[daemon] WARNING: heartbeat write failed: {e}")

    def _commit_worker() -> list[bool]:
        result = [False]
        try:
            volume.commit()
            result[0] = True
        except Exception as e:
            print(f"[daemon] ERROR: commit failed: {e}")
        return result

    while True:
        # Check for request
        if os.path.exists(request_file):
            # Write initial heartbeat and start commit in background
            _write_heartbeat()
            result_box: list[bool] = []

            def _run_commit(out: list[bool]) -> None:
                out.extend(_commit_worker())

            commit_thread = threading.Thread(target=_run_commit, args=(result_box,), daemon=False)
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
                except Exception as e:
                    print(f"[daemon] WARNING: failed to write failure marker: {e}")
        else:
            # No request yet; sleep and retry
            time.sleep(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Prefetch function (CPU only)


@app.function(timeout=2 * 60 * 60, volumes={"/data/nanochat-modal": vol}, image=image)
def prefetch() -> None:
    """CPU prefetch: download ClimbMix shards via dataset -n 170, commit."""
    base_dir = "/data/nanochat-modal"
    modal_env = _modal_env(base_dir)

    print("[prefetch] Starting dataset prefetch (170 shards)...")
    result = subprocess.run(
        ["python3", "-m", "nanochat.dataset", "-n", "170"],
        cwd="/root/nanochat",
        env={**os.environ, **modal_env},
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"prefetch dataset failed with exit code {result.returncode}")

    print("[prefetch] Dataset complete, committing volume...")
    vol.commit()
    print("[prefetch] Volume committed.")


# ─────────────────────────────────────────────────────────────────────────────
# Train function (GPU: 8×H100)


@app.function(
    image=image,
    gpu="H100:8",
    timeout=4 * 60 * 60,
    volumes={"/data/nanochat-modal": vol},
)
def train() -> None:
    """GPU train: speedrun.sh with SKIP_SETUP=1 and wait-for-ack hooks."""
    os.chdir("/root/nanochat")

    # Set environment: Modal hooks and stage markers go in Volume
    base_dir = "/data/nanochat-modal"
    modal_env = _modal_env(base_dir)
    for key, value in modal_env.items():
        os.environ[key] = value

    # Start commit daemon in background
    daemon_thread = threading.Thread(
        target=_commit_daemon,
        args=(vol, base_dir),
        daemon=True,
    )
    daemon_thread.start()
    print("[train] Commit daemon started")

    # Run speedrun.sh
    print("[train] Running speedrun.sh...")
    result = subprocess.run(
        ["bash", "runs/speedrun.sh"],
        cwd="/root/nanochat",
        env={**os.environ, **modal_env},
        check=False,
    )

    # Clean up hook files BEFORE final commit
    for marker in [".modal_commit_request", ".modal_commit_failed", ".modal_commit_heartbeat"]:
        try:
            os.remove(os.path.join(base_dir, marker))
        except FileNotFoundError:
            pass

    # Always commit final state (even if speedrun failed)
    print(f"[train] Speedrun complete (exit code {result.returncode}); final volume commit...")
    vol.commit()

    # Re-raise if speedrun failed
    if result.returncode != 0:
        raise RuntimeError(f"speedrun.sh failed with exit code {result.returncode}")


# ─────────────────────────────────────────────────────────────────────────────
# Local entrypoint: orchestrate prefetch → train → download → tip


@app.local_entrypoint()
def main() -> None:
    print("═" * 80)
    print("nanochat on Modal: prefetch → train → download")
    print("═" * 80)
    print("\n[local] Step 1: CPU prefetch 170 shards...")
    prefetch.remote()
    print("[local] Step 2: GPU train 8×H100...")
    train.remote()
    print("\n[local] Step 3: Downloading chat artifacts...")
    cache_dir = os.path.expanduser("~/.cache/nanochat-modal")
    os.makedirs(cache_dir, exist_ok=True)
    for remote, local in [("/tokenizer", f"{cache_dir}/tokenizer"), ("/chatsft_checkpoints", f"{cache_dir}/chatsft_checkpoints")]:
        print(f"  → {remote} → {local}")
        if subprocess.run(["modal", "volume", "get", "nanochat-modal", remote, local, "--force"], check=False).returncode:
            raise RuntimeError(f"volume get failed for {remote}")
    tokenizer_path, checkpoints_path = f"{cache_dir}/tokenizer", f"{cache_dir}/chatsft_checkpoints"
    if not os.path.exists(tokenizer_path) or not os.path.exists(checkpoints_path):
        raise RuntimeError("SFT incomplete: missing tokenizer or checkpoints")
    model_files = [f for f in os.listdir(checkpoints_path) if f.startswith("model_") and f.endswith(".pt")]
    if not model_files:
        raise RuntimeError("SFT incomplete: no model checkpoints")
    print(f"[local] Found {len(model_files)} checkpoint(s)\n" + "═" * 80)
    print("SUCCESS: training complete and artifacts downloaded.")
    print("═" * 80)
    print("\nTo chat locally, run:")
    print("  NANOCHAT_BASE_DIR=~/.cache/nanochat-modal python3 -m scripts.chat_cli -p \"Why is the sky blue?\"")
    print("\nSee docs/MODAL.md for details.")


if __name__ == "__main__":
    raise SystemExit("Use: modal run runs/modal_run.py")
