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
    timeout=60 * 60 * 5,                 # 5 hours for full speedrun
    max_inputs=1,
    gpu="B200:8",                        # 8xH100 for speedrun
    secrets=[modal.Secret.from_dotenv()],
)
def speedrun(wandb_run: str = "modal-speedrun", depth: int = 20):
    """
    Full speedrun: The best ChatGPT clone that $100 can buy.
    Runs the complete training pipeline from speedrun.sh:
      - Reset report and generate system info
      - Download dataset (240 shards for d20)
      - Train tokenizer on 2B characters
      - Evaluate tokenizer
      - Download eval_bundle
      - Pretrain base model (d20 = 561M params)
      - Evaluate base model
      - Midtraining (conversation, tool use, multiple choice)
      - Supervised finetuning
      - Generate final report

    All artifacts persist in the mounted Modal Volume.

    Args:
        wandb_run: Name for wandb run (default: "modal-speedrun")
        depth: Model depth (default: 20 for d20 model)
    """
    DATA = Path("/data")
    RUN_DIR = Path("/nanochat")
    BASE_DIR = DATA / ".cache" / "nanochat"
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

    print("\n" + "="*80)
    print("🚀 Starting nanochat speedrun - The best ChatGPT clone that $100 can buy")
    print("="*80 + "\n")

    # -----------------------------------------------------------------------------
    # Report setup
    print("\n📋 Step 1: Reset report and generate system info")
    _bash("uv run python -m nanochat.report reset", cwd=str(RUN_DIR), env=env)
    vol.commit()

    # -----------------------------------------------------------------------------
    # Tokenizer
    print("\n🔤 Step 2: Download dataset and train tokenizer")

    # Download initial shards for tokenizer training
    _bash("uv run python -m nanochat.dataset -n 8", cwd=str(RUN_DIR), env=env)
    vol.commit()

    # Download full dataset (240 shards for d20 model)
    # The d20 model is 561M parameters.
    # Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
    # Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
    # At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
    # Round up to 240 for safety.
    print("\n📦 Downloading full dataset (240 shards, ~24GB)...")
    _bash("uv run python -m nanochat.dataset -n 240", cwd=str(RUN_DIR), env=env)
    vol.commit()

    # Train tokenizer on 2B characters
    print("\n🏋️ Training tokenizer on 2B characters...")
    _bash("uv run python -m scripts.tok_train --max_chars=2000000000", cwd=str(RUN_DIR), env=env)
    _bash("uv run python -m scripts.tok_eval", cwd=str(RUN_DIR), env=env)
    vol.commit()

    # -----------------------------------------------------------------------------
    # Base model pretraining
    print("\n🧠 Step 3: Download eval_bundle and pretrain base model")

    # Download eval_bundle
    eval_bundle_path = BASE_DIR / "eval_bundle"
    if not eval_bundle_path.exists():
        print("\n📥 Downloading eval_bundle...")
        _bash(
            "curl -L -o eval_bundle.zip https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip && "
            f"unzip -q eval_bundle.zip && rm eval_bundle.zip && mv eval_bundle {eval_bundle_path}",
            cwd=str(RUN_DIR),
            env=env
        )
        vol.commit()

    # Pretrain the model
    # B200 has 192GB VRAM vs H100's 80GB, so we can double the batch size from 32 to 64
    print(f"\n🚂 Pretraining d{depth} model (561M params) with 2x batch size for B200...")
    _bash(
        f"uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth={depth} --device_batch_size=64 --run={wandb_run}",
        cwd=str(RUN_DIR),
        env=env
    )
    vol.commit()

    # Evaluate base model
    print("\n📊 Evaluating base model...")
    _bash("uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_loss", cwd=str(RUN_DIR), env=env)
    _bash("uv run torchrun --standalone --nproc_per_node=8 -m scripts.base_eval", cwd=str(RUN_DIR), env=env)
    vol.commit()

    # -----------------------------------------------------------------------------
    # Midtraining
    print("\n💬 Step 4: Midtraining (conversation, tool use, multiple choice)")

    # Download identity conversations
    identity_path = BASE_DIR / "identity_conversations.jsonl"
    if not identity_path.exists():
        print("\n📥 Downloading identity conversations...")
        _bash(
            f"curl -L -o {identity_path} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl",
            cwd=str(RUN_DIR),
            env=env
        )
        vol.commit()

    # Run midtraining
    print("\n🎓 Running midtraining with 2x batch size for B200...")
    _bash(
        f"uv run torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=64 --run={wandb_run}",
        cwd=str(RUN_DIR),
        env=env
    )
    _bash("uv run torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid", cwd=str(RUN_DIR), env=env)
    vol.commit()

    # -----------------------------------------------------------------------------
    # Supervised Finetuning
    print("\n✨ Step 5: Supervised Finetuning with 2x batch size for B200...")
    _bash(
        f"uv run torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device_batch_size=64 --run={wandb_run}",
        cwd=str(RUN_DIR),
        env=env
    )
    _bash("uv run torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft", cwd=str(RUN_DIR), env=env)
    vol.commit()

    # -----------------------------------------------------------------------------
    # Generate final report
    print("\n📝 Step 6: Generate final report")
    _bash("uv run python -m nanochat.report generate", cwd=str(RUN_DIR), env=env)
    vol.commit()

    print("\n" + "="*80)
    print(f"✅ Speedrun complete! Check wandb run: {wandb_run}")
    print(f"   Project: {os.environ.get('WANDB_PROJECT', 'not set')}")
    print(f"   Entity: {os.environ.get('WANDB_ENTITY', 'not set')}")
    print("="*80 + "\n")

    # Drop a pointer file for convenience
    with open(LOGS / "SPEEDRUN_COMPLETE.txt", "w") as f:
        f.write(
            f"""
🎉 SPEEDRUN COMPLETE! 🎉

Wandb Run: {wandb_run}
Model: d{depth} (561M parameters)

Artifacts live in the Modal Volume: {VOLUME_NAME}

Local code at:
  /nanochat

nanochat base dir (datasets/tokenizer/models/eval bundle/etc.):
  /data/.cache/nanochat

To list files:
  modal volume ls {VOLUME_NAME} /

To download everything locally:
  modal volume get {VOLUME_NAME} / ./nanochat_volume_dump

To download just the report:
  modal volume get {VOLUME_NAME} /.cache/nanochat/report.md ./report.md

Models and checkpoints live under:
  /data/.cache/nanochat
"""
        )
    vol.commit()
    print("\n✅ All artifacts persisted to volume. Speedrun complete!")


# Convenience local entrypoint
@app.local_entrypoint()
def main(wandb_run: str = "modal-speedrun", depth: int = 20):
    """
    Run the full speedrun on Modal.

    Args:
        wandb_run: Name for the wandb run (default: "modal-speedrun")
        depth: Model depth, e.g., 20 for d20 (default: 20)

    Example:
        modal run modal_speedrun.py
        modal run modal_speedrun.py --wandb-run my-experiment --depth 26
    """
    speedrun.remote(wandb_run=wandb_run, depth=depth)
