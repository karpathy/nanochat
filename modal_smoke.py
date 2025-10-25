import os
from pathlib import Path
import subprocess
import modal

APP_NAME = "nanochat-svilupp"
VOLUME_NAME = "nanochat-data"        # change if you like

app = modal.App(APP_NAME)

# Persisted volume to inspect results later
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Get the local directory path
LOCAL_DIR = Path(__file__).parent

# Mount local code and build image with dependencies
# Install Rust and build the tokenizer during image build for efficiency
# Use copy=True to copy files into the image so we can run build commands
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "build-essential", "pkg-config", "unzip")
    .add_local_dir("dev", "/nanochat/dev", copy=True)
    .add_local_dir("nanochat", "/nanochat/nanochat", copy=True)
    .add_local_dir("rustbpe", "/nanochat/rustbpe", copy=True)
    .add_local_dir("scripts", "/nanochat/scripts", copy=True)
    .add_local_dir("tasks", "/nanochat/tasks", copy=True)
    .add_local_dir("tests", "/nanochat/tests", copy=True)
    .add_local_file("pyproject.toml", "/nanochat/pyproject.toml", copy=True)
    .add_local_file(".python-version", "/nanochat/.python-version", copy=True)
    .add_local_file("run1000.sh", "/nanochat/run1000.sh", copy=True)
    .add_local_file("speedrun.sh", "/nanochat/speedrun.sh", copy=True)
    .add_local_file("README.md", "/nanochat/README.md", copy=True)
    .add_local_file("LICENSE", "/nanochat/LICENSE", copy=True)
    .workdir("/nanochat")
    .run_commands(
        # Install uv (Python package manager)
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        # Install Rust and set default toolchain
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable",
    )
    .env({"PATH": "/root/.cargo/bin:/root/.local/bin:$PATH"})
    .uv_sync(extras=["gpu"])
    .run_commands(
        # Build the Rust tokenizer (the slow part)
        "uv run maturin develop --release --manifest-path rustbpe/Cargo.toml",
    )
)

def _bash(cmd: str, *, cwd: str | None = None, env: dict | None = None):
    print(f"\n$ {cmd}")
    subprocess.run(["bash", "-lc", cmd], check=True, cwd=cwd, env=env)

@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=60 * 60 * 2,
    max_inputs=1,
    gpu="B200:8",
    secrets=[modal.Secret.from_dotenv()],
)
def smoke(shards: int = 4, max_chars: int = 10_000_000, wandb_run: str = "modal-smoke"):
    """
    Smoke test:
      - use local code (already copied to /nanochat during image build)
      - install deps with uv (already done in image build)
      - build rust tokenizer (already done in image build)
      - download a few dataset shards
      - train & eval tokenizer
    Artifacts persist in the mounted Modal Volume.
    """
    DATA = Path("/data")
    RUN_DIR = Path("/nanochat")               # local code is copied here
    BASE_DIR = DATA / ".cache" / "nanochat"   # repo uses this for caches/artifacts
    LOGS = DATA / "logs"
    for p in (BASE_DIR, LOGS):
        p.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            # Persist all nanochat caches/artifacts into the volume
            "NANOCHAT_BASE_DIR": str(BASE_DIR),
            # Keep CPU threads tame for reproducibility/CI-like behavior
            "OMP_NUM_THREADS": "1",
            # Rust is already installed in the image at /root/.cargo, so we don't override CARGO_HOME/RUSTUP_HOME
            # wandb configuration (env vars loaded from .env via modal.Secret.from_dotenv())
            # WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT are already in os.environ from the secret
        }
    )

    # Print wandb status for debugging
    if "WANDB_API_KEY" in os.environ:
        print("✓ WANDB_API_KEY found in environment")
    if "WANDB_ENTITY" in os.environ:
        print(f"✓ WANDB_ENTITY: {os.environ.get('WANDB_ENTITY')}")
    if "WANDB_PROJECT" in os.environ:
        print(f"✓ WANDB_PROJECT: {os.environ.get('WANDB_PROJECT')}")

    # Rust and tokenizer are already built in the image, so we skip that step here!

    # 1) Tiny data grab for smoke (download a few shards into base dir)
    _bash(f"uv run python -m nanochat.dataset -n {int(shards)}", cwd=str(RUN_DIR), env=env)
    vol.commit()

    # 2) Tokenizer train & eval on a tiny subset (fast)
    _bash(f"uv run python -m scripts.tok_train --max_chars={int(max_chars)}", cwd=str(RUN_DIR), env=env)
    _bash("uv run python -m scripts.tok_eval", cwd=str(RUN_DIR), env=env)
    vol.commit()

    print(f"\n📊 Training complete! Check wandb run: {wandb_run}")
    print(f"   Project: {os.environ.get('WANDB_PROJECT', 'not set')}")
    print(f"   Entity: {os.environ.get('WANDB_ENTITY', 'not set')}")

    # 3) Drop a pointer file for convenience
    with open(LOGS / "WHERE_IS_MY_STUFF.txt", "w") as f:
        f.write(
            f"""
Artifacts live in the Modal Volume: {VOLUME_NAME}

Local code at:
  /nanochat

nanochat base dir (datasets/tokenizer/eval bundle/etc.):
  /data/.cache/nanochat

To list files:
  modal volume ls {VOLUME_NAME} /

To download everything locally:
  modal volume get {VOLUME_NAME} / ./nanochat_volume_dump

Tokenizers live under:
  /data/.cache/nanochat (inside the volume)
"""
        )
    vol.commit()
    print("\n✅ Smoke test complete. Volume committed.")


# Convenience local entrypoint so you can just `modal run modal_runner.py`
@app.local_entrypoint()
def main(shards: int = 4, max_chars: int = 10_000_000, wandb_run: str = "modal-smoke"):
    """
    Run the smoke test on Modal.

    Args:
        shards: Number of dataset shards to download (default: 4)
        max_chars: Maximum characters for tokenizer training (default: 10M)
        wandb_run: Name for the wandb run (default: "modal-smoke")

    Example:
        modal run modal_runner.py --shards 8 --max-chars 20000000 --wandb-run my-test
    """
    smoke.remote(shards=shards, max_chars=max_chars, wandb_run=wandb_run)
