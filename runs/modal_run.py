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
import sys
import threading
import time
from pathlib import Path

# Modal SDK: Load only when imported inside Modal functions, not at module level.
import modal

# ─────────────────────────────────────────────────────────────────────────────
# Modal app and image definition

app = modal.App(name="nanochat-speedrun")

# Shared Volume: mounted at /data/nanochat-modal
vol = modal.Volume.from_name("nanochat-modal", create_if_missing=True)

# Base image: debian_slim + python 3.11 + deps + code + .venv
image = (
    modal.Image.debian_slim(python_version="3.11")
    .add_local_dir(
        ".",
        remote_path="/root/nanochat",
        ignore=[".venv", ".git", "**/.pytest_cache", "**/__pycache__"],
    )
    .run_commands("cd /root/nanochat && uv sync --extra gpu --frozen")
)

# ─────────────────────────────────────────────────────────────────────────────
# Wait-for-ack daemon (runs in background during GPU function)


def _commit_daemon(volume: modal.Volume, base_dir: str, modal_hook_path: str) -> None:
    """
    Daemon thread: watch for commit requests, run vol.commit(), update heartbeat.

    Protocol:
    - On .modal_commit_request file: write heartbeat, run commit in worker thread,
      update heartbeat every 10s while commit runs, remove request on success.
    - On failure: write .modal_commit_failed.
    """
    heartbeat_file = os.path.join(base_dir, ".modal_commit_heartbeat")
    request_file = os.path.join(base_dir, ".modal_commit_request")
    failed_file = os.path.join(base_dir, ".modal_commit_failed")
    heartbeat_interval = 10  # seconds between heartbeat updates

    def _write_heartbeat() -> None:
        """Write current timestamp to heartbeat file."""
        try:
            with open(heartbeat_file, "w") as f:
                f.write(str(int(time.time())))
        except Exception as e:
            print(f"[daemon] WARNING: failed to write heartbeat: {e}")

    def _commit_worker() -> bool:
        """Run vol.commit() in a worker thread, return success."""
        try:
            volume.commit()
            return True
        except Exception as e:
            print(f"[daemon] ERROR: volume.commit() failed: {e}")
            return False

    while True:
        # Check for request
        if os.path.exists(request_file):
            # Write initial heartbeat and start commit in background
            _write_heartbeat()
            commit_thread = threading.Thread(target=_commit_worker, daemon=False)
            commit_thread.start()

            # Update heartbeat every 10s until commit finishes
            while commit_thread.is_alive():
                time.sleep(1)
                # Don't spam heartbeat too often; update every 10s
                if time.time() % heartbeat_interval < 1:
                    _write_heartbeat()

            # Commit finished; check success
            commit_thread.join()

            # Remove request on success (hook checks this to know we're done)
            try:
                os.remove(request_file)
                print("[daemon] commit succeeded, request removed")
            except FileNotFoundError:
                pass
        else:
            # No request yet; sleep and retry
            time.sleep(0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Prefetch function (CPU only)


@app.function(timeout=2 * 60 * 60, volumes={"/data/nanochat-modal": vol})
def prefetch() -> None:
    """CPU prefetch: download ClimbMix shards via dataset -n 170, commit."""
    os.environ["NANOCHAT_BASE_DIR"] = "/data/nanochat-modal"
    os.environ["HF_HOME"] = "/data/nanochat-modal/hf_home"

    print("[prefetch] Starting dataset prefetch (170 shards)...")
    result = subprocess.run(
        ["python3", "-m", "nanochat.dataset", "-n", "170"],
        cwd="/root/nanochat",
        env={
            **os.environ,
            "NANOCHAT_BASE_DIR": "/data/nanochat-modal",
            "HF_HOME": "/data/nanochat-modal/hf_home",
        },
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
    os.environ["NANOCHAT_BASE_DIR"] = base_dir
    os.environ["HF_HOME"] = os.path.join(base_dir, "hf_home")
    os.environ["SKIP_SETUP"] = "1"
    os.environ["NANOCHAT_POST_STAGE_HOOK"] = "bash /root/nanochat/runs/modal_hook.sh"
    os.environ["PYTHONPATH"] = "/root/nanochat"

    # Start commit daemon in background
    modal_hook_path = "/root/nanochat/runs/modal_hook.sh"
    daemon_thread = threading.Thread(
        target=_commit_daemon,
        args=(vol, base_dir, modal_hook_path),
        daemon=True,
    )
    daemon_thread.start()
    print("[train] Commit daemon started")

    # Run speedrun.sh
    print("[train] Running speedrun.sh...")
    result = subprocess.run(
        ["bash", "runs/speedrun.sh"],
        cwd="/root/nanochat",
        env={
            **os.environ,
            "NANOCHAT_BASE_DIR": base_dir,
            "HF_HOME": os.path.join(base_dir, "hf_home"),
            "SKIP_SETUP": "1",
            "NANOCHAT_POST_STAGE_HOOK": "bash /root/nanochat/runs/modal_hook.sh",
            "PYTHONPATH": "/root/nanochat",
        },
    )

    # Always commit final state (even if speedrun failed)
    print("[train] Speedrun complete (exit code {}); final volume commit...".format(result.returncode))
    vol.commit()

    # Clean up any leftover hook files
    for marker_file in [".modal_commit_request", ".modal_commit_failed", ".modal_commit_heartbeat"]:
        path = os.path.join(base_dir, marker_file)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    # Re-raise if speedrun failed
    if result.returncode != 0:
        raise RuntimeError(f"speedrun.sh failed with exit code {result.returncode}")


# ─────────────────────────────────────────────────────────────────────────────
# Local entrypoint: orchestrate prefetch → train → download → tip


@app.local_entrypoint()
def main() -> None:
    """Orchestrate prefetch, train, and local download."""
    print("═" * 80)
    print("nanochat on Modal: prefetch → train → download")
    print("═" * 80)

    # Step 1: CPU prefetch
    print("\n[local] Step 1: CPU prefetch 170 shards...")
    prefetch.remote()

    # Step 2: GPU train
    print("[local] Step 2: GPU train 8×H100...")
    train.remote()

    # Step 3: Download artifacts to local cache
    print("\n[local] Step 3: Downloading chat artifacts to ~/.cache/nanochat-modal...")

    cache_dir = os.path.expanduser("~/.cache/nanochat-modal")
    os.makedirs(cache_dir, exist_ok=True)

    # Define what to download from Volume
    downloads = [
        ("/tokenizer", os.path.join(cache_dir, "tokenizer")),
        ("/chatsft_checkpoints", os.path.join(cache_dir, "chatsft_checkpoints")),
    ]

    # Modal volume get via subprocess (since we're in local entrypoint)
    for remote_path, local_path in downloads:
        print(f"  → {remote_path} → {local_path}")
        cmd = ["modal", "volume", "get", "nanochat-modal", remote_path, local_path]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"ERROR: failed to download {remote_path}", file=sys.stderr)
            raise RuntimeError(f"volume get failed for {remote_path}")

    # Verify artifacts exist
    tokenizer_path = os.path.join(cache_dir, "tokenizer")
    checkpoints_path = os.path.join(cache_dir, "chatsft_checkpoints")

    if not os.path.exists(tokenizer_path) or not os.path.exists(checkpoints_path):
        print(
            "ERROR: SFT training did not complete; chat artifacts missing.",
            file=sys.stderr,
        )
        raise RuntimeError("missing required chat artifacts (SFT training incomplete)")

    # Step 4: Print local chat tip
    print("\n" + "═" * 80)
    print("SUCCESS: training complete and artifacts downloaded.")
    print("═" * 80)
    print("\nTo chat locally, run:")
    print(
        f"  NANOCHAT_BASE_DIR=~/.cache/nanochat-modal python3 -m scripts.chat_cli -p \"Why is the sky blue?\""
    )
    print("\nSee docs/MODAL.md for details.")


if __name__ == "__main__":
    raise SystemExit("Use: modal run runs/modal_run.py")
