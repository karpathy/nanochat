"""
Modal deployment for nanochat - serves the existing chat_web.py FastAPI app on H100.

Usage:
    modal deploy modal_serve.py

This will:
1. Build a container image with PyTorch, FastAPI, and the nanochat module
2. Load the best available checkpoint (from sft by default)
3. Serve the chat UI and API endpoints from scripts/chat_web.py

The web UI will be available at the URL printed by Modal after deployment.

Note: Before deploying, upload your model checkpoints to the volume.
"""

import modal
from pathlib import Path

APP_NAME = "nanochat-serve"
VOLUME_NAME = "nanochat-data"  # Reuse the same volume as modal_speedrun.py

app = modal.App(APP_NAME)

# Reuse volume from modal_speedrun (or create if missing)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Get the local directory path
LOCAL_DIR = Path(__file__).parent

# Build Modal image with identical environment to modal_speedrun.py
# This ensures consistency between training and serving
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


@app.function(
    image=image,
    gpu="H100",
    volumes={"/data": vol},
    timeout=3600,  # 1 hour timeout
    scaledown_window=300,  # Keep alive for 5 min after last request
)
@modal.asgi_app()
def fastapi_app():
    """
    Import and return the FastAPI app from chat_web.py.

    This reuses all the existing logic: endpoints, streaming, validation, etc.
    The only difference is we run on Modal infrastructure with H100 GPU.
    """
    import sys
    import os

    # Set base directory to where checkpoints are mounted (same as modal_speedrun)
    BASE_DIR = "/data/.cache/nanochat"
    os.environ['NANOCHAT_BASE_DIR'] = BASE_DIR

    # Mock the command-line arguments that chat_web.py expects
    sys.argv = [
        'chat_web.py',
        '--num-gpus', '1',          # Single GPU (Modal handles scaling)
        '--source', 'sft',           # Load from sft checkpoints
        '--temperature', '0.8',      # Default temperature
        '--top-k', '50',             # Default top-k
        '--max-tokens', '512',       # Default max tokens
        '--device-type', 'cuda',     # Use CUDA
        '--dtype', 'bfloat16',       # Use bfloat16 for efficiency
    ]

    # Import the FastAPI app from chat_web
    # This will trigger model loading via the lifespan context manager
    from scripts.chat_web import app

    print(f"✅ NanoChat server initialized!")
    print(f"   Checkpoint directory: {BASE_DIR}")
    print(f"   GPU: H100 x 1")

    return app


# Convenience local entrypoint for testing
@app.local_entrypoint()
def main():
    """
    Deploy the nanochat serving endpoint.

    This is just a convenience wrapper. You can also run:
        modal deploy modal_serve.py
    """
    print("Deploying nanochat serving endpoint...")
    print(f"Using volume: {VOLUME_NAME}")
    print(f"GPU: H100 x 1")
    print("\nThe app will be available at the URL printed by Modal.")
