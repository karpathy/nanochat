import os
from pathlib import Path
import subprocess
import modal

APP_NAME = "nanochat-svilupp"
VOLUME_NAME = "nanochat-data"        # change if you like

# -----------------------------------------------------------------------------
# Batch Size Configuration
#
# Adjust GPU_VRAM_GB based on your GPU type:
#   - B200: 192GB (default)
#   - H100: 80GB
#   - A100: 80GB
#
# All batch sizes are calculated automatically based on VRAM to maximize
# throughput while avoiding OOM. The script uses 90% of theoretical capacity
# to leave headroom for memory spikes during checkpointing.
# -----------------------------------------------------------------------------
GPU_VRAM_GB = 192                    # VRAM per GPU in GB (B200=192, H100=80)
NUM_GPUS = 8                         # number of GPUs (match your torchrun --nproc_per_node)
MAX_SEQ_LEN = 2048                   # sequence length (from base_train.py default)

# Reference batch size for H100 (80GB)
DEVICE_BATCH_SIZE_H100 = 32
H100_VRAM_GB = 80

# HArdcode to 80!
DEVICE_BATCH_SIZE = 80 # int(DEVICE_BATCH_SIZE_H100 * (GPU_VRAM_GB / H100_VRAM_GB) * 0.9)

# Calculate total_batch_size: must be a multiple of (device_batch_size * max_seq_len * num_gpus)
# This ensures no remainder in gradient accumulation calculation
WORLD_TOKENS_PER_STEP = DEVICE_BATCH_SIZE * MAX_SEQ_LEN * NUM_GPUS
TOTAL_BATCH_SIZE = WORLD_TOKENS_PER_STEP * 1  # 1x for no gradient accumulation

# For SFT: 
TARGET_EXAMPLES_PER_STEP = DEVICE_BATCH_SIZE  # Target equals device batch size
# Calculate device_batch_size_sft so that: target % (device_batch_size_sft * num_gpus) == 0
DEVICE_BATCH_SIZE_SFT = TARGET_EXAMPLES_PER_STEP // NUM_GPUS  # e.g., 80 / 8 = 10

print(f"Batch Size Configuration:")
print(f"  GPU VRAM: {GPU_VRAM_GB}GB ({(GPU_VRAM_GB/H100_VRAM_GB):.1f}x H100)")
print(f"  Sequence length: {MAX_SEQ_LEN}")
print(f"\n  Base/Mid Training:")
print(f"    Device batch size: {DEVICE_BATCH_SIZE} (H100 baseline: {DEVICE_BATCH_SIZE_H100})")
print(f"    World tokens per step: {WORLD_TOKENS_PER_STEP:,}")
print(f"    Total batch size: {TOTAL_BATCH_SIZE:,}")
print(f"    Gradient accumulation: {TOTAL_BATCH_SIZE // WORLD_TOKENS_PER_STEP}")
print(f"\n  SFT Training:")
print(f"    Device batch size: {DEVICE_BATCH_SIZE_SFT}")
print(f"    Target examples per step: {TARGET_EXAMPLES_PER_STEP}")
print(f"    Examples per step: {DEVICE_BATCH_SIZE_SFT * NUM_GPUS}")
print(f"    Gradient accumulation: {TARGET_EXAMPLES_PER_STEP // (DEVICE_BATCH_SIZE_SFT * NUM_GPUS)}")
# -----------------------------------------------------------------------------

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
    gpu=f"B200:{NUM_GPUS}",                        
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
    print(f"\n🚂 Pretraining d{depth} model (561M params)...")
    print(f"    Batch config: device_batch_size={DEVICE_BATCH_SIZE}, total_batch_size={TOTAL_BATCH_SIZE}")
    _bash(
        f"uv run torchrun --standalone --nproc_per_node={NUM_GPUS} -m scripts.base_train -- --depth={depth} --device_batch_size={DEVICE_BATCH_SIZE} --total_batch_size={TOTAL_BATCH_SIZE} --run={wandb_run}",
        cwd=str(RUN_DIR),
        env=env
    )
    vol.commit()

    # Evaluate base model
    print("\n📊 Evaluating base model...")
    _bash(f"uv run torchrun --standalone --nproc_per_node={NUM_GPUS} -m scripts.base_loss", cwd=str(RUN_DIR), env=env)
    _bash(f"uv run torchrun --standalone --nproc_per_node={NUM_GPUS} -m scripts.base_eval", cwd=str(RUN_DIR), env=env)
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
    print("\n🎓 Running midtraining...")
    print(f"    Batch config: device_batch_size={DEVICE_BATCH_SIZE}, total_batch_size={TOTAL_BATCH_SIZE}")
    _bash(
        f"uv run torchrun --standalone --nproc_per_node={NUM_GPUS} -m scripts.mid_train -- --device_batch_size={DEVICE_BATCH_SIZE} --total_batch_size={TOTAL_BATCH_SIZE} --run={wandb_run}",
        cwd=str(RUN_DIR),
        env=env
    )
    _bash(f"uv run torchrun --standalone --nproc_per_node={NUM_GPUS} -m scripts.chat_eval -- -i mid", cwd=str(RUN_DIR), env=env)
    vol.commit()

    # -----------------------------------------------------------------------------
    # Supervised Finetuning
    # Note: chat_sft.py doesn't use total_batch_size, but uses target_examples_per_step
    # We load from "mid" checkpoint (midtraining) by default
    # Use smaller batch size for SFT due to variable sequence lengths
    print("\n✨ Step 5: Supervised Finetuning...")
    print(f"    Batch config: device_batch_size={DEVICE_BATCH_SIZE_SFT}, target_examples_per_step={TARGET_EXAMPLES_PER_STEP}")
    print(f"    Source checkpoint: mid")
    _bash(
        f"uv run torchrun --standalone --nproc_per_node={NUM_GPUS} -m scripts.chat_sft -- --device_batch_size={DEVICE_BATCH_SIZE_SFT} --target_examples_per_step={TARGET_EXAMPLES_PER_STEP} --source=mid --run={wandb_run}",
        cwd=str(RUN_DIR),
        env=env
    )
    _bash(f"uv run torchrun --standalone --nproc_per_node={NUM_GPUS} -m scripts.chat_eval -- -i sft", cwd=str(RUN_DIR), env=env)
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


@app.function(
    image=image,
    volumes={"/data": vol},
    timeout=60 * 60 * 1,                 # 1 hour should be enough for SFT
    max_inputs=1,
    gpu="B200:8",
    secrets=[modal.Secret.from_dotenv()],
)
def sft_only(wandb_run: str = "modal-sft", checkpoint: str = "mid"):
    """
    Run only the Supervised Finetuning stage.

    This is useful for iterating on SFT after completing the full speedrun,
    or for resuming from a midtraining checkpoint.

    Args:
        wandb_run: Name for wandb run (default: "modal-sft")
        checkpoint: Which checkpoint to start from: "mid" or "base" (default: "mid")

    Requirements:
        - Tokenizer must be trained (in volume)
        - Checkpoint must exist (either mid_model.pt or base_model.pt)
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
            "NANOCHAT_BASE_DIR": str(BASE_DIR),
            "OMP_NUM_THREADS": "1",
        }
    )

    # Print wandb status
    if "WANDB_API_KEY" in os.environ:
        print("✓ WANDB_API_KEY found in environment")
    if "WANDB_ENTITY" in os.environ:
        print(f"✓ WANDB_ENTITY: {os.environ.get('WANDB_ENTITY')}")
    if "WANDB_PROJECT" in os.environ:
        print(f"✓ WANDB_PROJECT: {os.environ.get('WANDB_PROJECT')}")

    print("\n" + "="*80)
    print(f"🚀 Running SFT-only from checkpoint source: {checkpoint}")
    print("="*80 + "\n")

    # Note: We don't need to validate the checkpoint here.
    # The load_model() function in chat_sft.py will automatically:
    # 1. Find the checkpoint directory (e.g., mid_checkpoints/)
    # 2. Find the largest model tag (e.g., d20)
    # 3. Find the latest checkpoint step (e.g., model_000323.pt)
    # If anything is missing, it will fail with a clear error message.

    # Run SFT
    print("\n✨ Running Supervised Finetuning...")
    print(f"    Batch config: device_batch_size={DEVICE_BATCH_SIZE_SFT}, target_examples_per_step={TARGET_EXAMPLES_PER_STEP}")
    print(f"    Source checkpoint: {checkpoint}")
    _bash(
        f"uv run torchrun --standalone --nproc_per_node={NUM_GPUS} -m scripts.chat_sft -- --device_batch_size={DEVICE_BATCH_SIZE_SFT} --target_examples_per_step={TARGET_EXAMPLES_PER_STEP} --source={checkpoint} --run={wandb_run}",
        cwd=str(RUN_DIR),
        env=env
    )

    # Evaluate
    print("\n📊 Evaluating SFT model...")
    _bash(f"uv run torchrun --standalone --nproc_per_node={NUM_GPUS} -m scripts.chat_eval -- -i sft", cwd=str(RUN_DIR), env=env)
    vol.commit()

    # Generate report
    print("\n📝 Generating report...")
    _bash("uv run python -m nanochat.report generate", cwd=str(RUN_DIR), env=env)
    vol.commit()

    print("\n" + "="*80)
    print(f"✅ SFT complete! Check wandb run: {wandb_run}")
    print(f"   Project: {os.environ.get('WANDB_PROJECT', 'not set')}")
    print(f"   Entity: {os.environ.get('WANDB_ENTITY', 'not set')}")
    print("="*80 + "\n")

    with open(LOGS / "SFT_COMPLETE.txt", "w") as f:
        f.write(f"SFT run complete: {wandb_run}\nCheckpoint: {checkpoint}\n")
    vol.commit()


# Convenience local entrypoint
@app.local_entrypoint()
def main(
    mode: str = "full",
    wandb_run: str = "",
    depth: int = 20,
    checkpoint: str = "mid"
):
    """
    Run training on Modal.

    Args:
        mode: "full" for complete speedrun, "sft" for SFT-only (default: "full")
        wandb_run: Name for wandb run (auto-generated if not specified)
        depth: Model depth for full run, e.g., 20 for d20 (default: 20)
        checkpoint: For SFT mode, which checkpoint to use: "mid" or "base" (default: "mid")

    Examples:
        # Full speedrun
        modal run modal_speedrun.py
        modal run modal_speedrun.py --wandb-run my-experiment --depth 26

        # SFT only (must have existing checkpoint in volume)
        modal run modal_speedrun.py --mode sft
        modal run modal_speedrun.py --mode sft --wandb-run my-sft-v2 --checkpoint base
    """
    if mode == "full":
        run_name = wandb_run if wandb_run else "modal-speedrun"
        speedrun.remote(wandb_run=run_name, depth=depth)
    elif mode == "sft":
        run_name = wandb_run if wandb_run else "modal-sft"
        sft_only.remote(wandb_run=run_name, checkpoint=checkpoint)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'full' or 'sft'")
