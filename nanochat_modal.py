"""
Ablation study: picochat (depth=8) — baseline vs SwiGLU vs RoPE-500K

Setup (one-time):
    modal setup                                   # authenticate (yoyoliuuu workspace)
    modal secret create nanochat-secrets \\
        WANDB_API_KEY=<your_key> \\
        HF_TOKEN=hf_<your_token>

Run all 3 ablations end-to-end (data + tokenizer once, then 3 training runs):
    modal run nanochat_modal.py

Run individual stages:
    modal run nanochat_modal.py::stage_data
    modal run nanochat_modal.py::stage_tokenizer
    modal run nanochat_modal.py::run_baseline
    modal run nanochat_modal.py::run_swiglu
    modal run nanochat_modal.py::run_rope500k

Cost reference (A10G at ~$1.10/hr):
    picochat d8, 1 GPU: ~1-2 hr per run → ~$1.50-3 per run
    3 runs total: ~$5-9

Notes:
    - Data and tokenizer are cached in a persistent Modal Volume.
      All 3 runs share the same data/tokenizer — download only happens once.
    - Stages are idempotent where possible.
    - The nanochat repo is copied into the container image at build time.
      If you change gpt.py or base_train.py, Modal auto-rebuilds the image.
    - W&B runs go to the yoyoliuuu workspace under project "nanochat".
      Each run is tagged: picochat-baseline, picochat-swiglu, picochat-rope500k.

Reference: Angela Sha, https://github.com/UofT-CSC490-W2026/022326-tutorial-nanochat
"""

import os
import subprocess
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION
# =============================================================================

# ── picochat config ────────────────────────────────────────────────────────────
# depth=8 → model_dim=512, n_heads=4, ~42M params. Trains in ~1-2hr on A10G.
DEPTH = 8
MAX_SEQ_LEN = 512
DEVICE_BATCH_SIZE = 16

# ── GPU ────────────────────────────────────────────────────────────────────────
# A10G: cheapest that comfortably fits picochat, no FA3 (SDPA fallback is fine)
# Switch to "H100:1" (~$3.09/hr) if you want 2x speed
GPU = "A10G:1"

# ── Data shards ────────────────────────────────────────────────────────────────
# picochat needs ~500M tokens → ~8 shards at 250M chars/shard ≈ 4 chars/token
# Use 12 to have a little extra buffer
NUM_SHARDS = 12

# ── W&B ───────────────────────────────────────────────────────────────────────
# WANDB_ENTITY is read from the Modal secret (set via modal secret create)

# ── Volume + paths ─────────────────────────────────────────────────────────────
VOLUME_MOUNT = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"

# ── Timeouts ──────────────────────────────────────────────────────────────────
TRAIN_TIMEOUT_SEC  = 60 * 60 * 4   # 4h max per picochat run (safe margin)
DOWNLOAD_TIMEOUT_SEC = 60 * 60     # 1h for data download

# ── Eval toggle ──────────────────────────────────────────────────────────────
# CORE metric is expensive (~20-40min). Set to -1 to skip during ablation.
# val/bpb logged every 250 steps is enough for comparing runs.
CORE_METRIC_EVERY = -1   # disable mid-training CORE eval to keep runs cheap

# =============================================================================
# MODAL PRIMITIVES
# =============================================================================

app = App("nanochat-ablation")

# Persistent volume: data shards, tokenizer, and checkpoints survive restarts
volume = Volume.from_name("nanochat-vol", create_if_missing=True)

# Secret: WANDB_API_KEY and HF_TOKEN injected as env vars
secret = Secret.from_name("nanochat-secrets")

# Container image — rebuilt automatically if source files change
image = (
    ModalImage.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git", "build-essential", "curl", "wget", "unzip")

    # Copy the nanochat repo (this file's own repo) into the container
    # Modal hashes the directory contents, so changes to gpt.py etc trigger rebuild
    .add_local_dir(
        local_path=".",
        remote_path="/root/nanochat",
        copy=True,
        ignore=[".venv", "__pycache__", "*.pyc", ".git", "rustbpe/target", "runs"],
    )
    .workdir("/root/nanochat")

    # Install uv and project deps (rustbpe installs as a prebuilt wheel from PyPI)
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> $HOME/.bashrc",
    )
    .pip_install("uv")
    .run_commands("uv sync --extra gpu --no-install-project")

    # Env vars: nanochat reads NANOCHAT_BASE_DIR to find its cache
    .env({
        "OMP_NUM_THREADS": "1",
        "NANOCHAT_BASE_DIR": NANOCHAT_CACHE,
        "HF_HOME": f"{VOLUME_MOUNT}/hf_cache",
    })
)

# =============================================================================
# HELPERS
# =============================================================================

def _python(module: str, args: list | None = None) -> None:
    args = args or []
    cmd = f"cd /root/nanochat && uv run python -m {module} {' '.join(args)}"
    _run(cmd)


def _torchrun(module: str, args: list | None = None, *, nproc: int = 1) -> None:
    args = args or []
    args_str = (" -- " + " ".join(args)) if args else ""
    cmd = (
        f"cd /root/nanochat && "
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m {module}{args_str}"
    )
    _run(cmd)


def _run(cmd: str) -> None:
    print(f"\n>>>  {cmd}\n")
    result = subprocess.run(["bash", "-c", cmd], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}):\n  {cmd}")


def _setup_cache() -> None:
    """Create the nanochat cache dir (NANOCHAT_BASE_DIR already points there via env)."""
    os.makedirs(NANOCHAT_CACHE, exist_ok=True)


# =============================================================================
# STAGE 0: DATA DOWNLOAD
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=8,
    memory=16384,
    timeout=DOWNLOAD_TIMEOUT_SEC,
)
def stage_data(num_shards: int = NUM_SHARDS) -> None:
    """Download FineWeb-EDU shards (CPU-only, cached in volume — runs once for all ablations)."""
    _setup_cache()
    print(f"Downloading {num_shards} FineWeb-EDU shards to volume...")
    _python("nanochat.dataset", [f"-n {num_shards}"])
    volume.commit()
    print("Done.")


# =============================================================================
# STAGE 1: TOKENIZER
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu="A10G:1",
    timeout=60 * 30,
)
def stage_tokenizer() -> None:
    """Train the BPE tokenizer (1 GPU, ~2 min — runs once for all ablations)."""
    _setup_cache()
    tokenizer_path = os.path.join(NANOCHAT_CACHE, "tokenizer.model")
    if os.path.exists(tokenizer_path):
        print("Tokenizer already exists, skipping.")
    else:
        print("Training tokenizer on 2B chars...")
        _python("scripts.tok_train", ["--max-chars=2000000000"])
        volume.commit()
    _python("scripts.tok_eval")


# =============================================================================
# STAGE 2: PRETRAIN (parametric — used by all 3 ablations)
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU,
    timeout=TRAIN_TIMEOUT_SEC,
)
def stage_pretrain(
    run_name: str,
    model_tag: str,
    mlp_type: str = "relu2",
    rope_base: int = 10000,
    num_mtp_steps: int = 0,
    mtp_loss_weight: float = 0.3,
) -> None:
    """
    Pretrain one picochat ablation variant.

    Args:
        run_name:       W&B run name (e.g. 'picochat-baseline')
        model_tag:      checkpoint directory name (e.g. 'picochat-baseline')
        mlp_type:       'relu2' or 'swiglu'
        rope_base:      RoPE base theta (10000 or 500000)
        num_mtp_steps:  MTP auxiliary heads (0=disabled, 1=predict 2 tokens ahead)
        mtp_loss_weight: weight for each MTP auxiliary loss term
    """
    _setup_cache()
    print(f"\n{'='*60}")
    print(f"Picochat ablation: {run_name}")
    print(f"  mlp_type={mlp_type}  rope_base={rope_base}  num_mtp_steps={num_mtp_steps}")
    print(f"  depth={DEPTH}  seq_len={MAX_SEQ_LEN}  gpu={GPU}")
    print(f"{'='*60}\n")

    _python("nanochat.report", ["reset"])

    _torchrun(
        "scripts.base_train",
        [
            f"--depth={DEPTH}",
            f"--max-seq-len={MAX_SEQ_LEN}",
            f"--device-batch-size={DEVICE_BATCH_SIZE}",
            f"--window-pattern=L",          # full context; SDPA works cleanly with this
            f"--mlp-type={mlp_type}",
            f"--rope-base={rope_base}",
            f"--num-mtp-steps={num_mtp_steps}",
            f"--mtp-loss-weight={mtp_loss_weight}",
            f"--run={run_name}",
            f"--model-tag={model_tag}",
            f"--core-metric-every={CORE_METRIC_EVERY}",  # skip expensive CORE eval
            "--save-every=500",             # checkpoint every 500 steps (survive disconnects)
            "--eval-every=100",             # val/bpb every 100 steps for dense W&B curves
        ],
    )
    volume.commit()
    print(f"Done: {run_name}")


# =============================================================================
# INDIVIDUAL ABLATION ENTRYPOINTS
# (use ::run_baseline etc. to re-run one study after data/tokenizer are ready)
# =============================================================================

@app.local_entrypoint()
def run_baseline() -> None:
    """Re-run baseline only (requires data+tokenizer already on volume)."""
    stage_pretrain.remote(
        run_name="picochat-baseline",
        model_tag="picochat-baseline",
        mlp_type="relu2",
        rope_base=10000,
    )


@app.local_entrypoint()
def run_swiglu() -> None:
    """Re-run SwiGLU ablation only (requires data+tokenizer already on volume)."""
    stage_pretrain.remote(
        run_name="picochat-swiglu",
        model_tag="picochat-swiglu",
        mlp_type="swiglu",
        rope_base=10000,
    )


@app.local_entrypoint()
def run_rope500k() -> None:
    """Re-run RoPE-500K ablation only (requires data+tokenizer already on volume)."""
    stage_pretrain.remote(
        run_name="picochat-rope500k",
        model_tag="picochat-rope500k",
        mlp_type="relu2",
        rope_base=500000,
    )


@app.local_entrypoint()
def run_mtp() -> None:
    """Re-run MTP ablation only (requires data+tokenizer already on volume)."""
    stage_pretrain.remote(
        run_name="picochat-mtp",
        model_tag="picochat-mtp",
        mlp_type="relu2",
        rope_base=10000,
        num_mtp_steps=1,
        mtp_loss_weight=0.3,
    )


# =============================================================================
# PIPELINE ORCHESTRATOR (runs on Modal servers, not locally)
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=1,
    timeout=60 * 60 * 5,  # 5 hours: enough for tokenizer + 3 training runs
)
def run_pipeline(num_shards: int = NUM_SHARDS) -> None:
    """
    Full ablation pipeline running entirely on Modal servers.
    Called via .spawn() from main() so your laptop can close immediately.

    Stages:
      1. Download data (idempotent — skips shards already on volume)
      2. Train tokenizer (idempotent — skips if tokenizer.pkl exists)
      3. Baseline   — relu2, RoPE 10K           (~50 min)
      4. SwiGLU     — swiglu, RoPE 10K          (~50 min)
      5. MTP        — relu2, 1 MTP step, w=0.3  (~52 min)
    """
    _setup_cache()
    print("\n" + "="*60)
    print("Picochat Ablation Study  |  yoyoliuuu/nanochat  |  W&B")
    print("="*60 + "\n")

    print("[1/5] Downloading data shards...")
    stage_data.remote(num_shards=num_shards)

    print("[2/5] Training tokenizer...")
    stage_tokenizer.remote()

    print("[3/5] Training baseline (relu2 + RoPE 10K)...")
    stage_pretrain.remote(
        run_name="picochat-baseline",
        model_tag="picochat-baseline",
        mlp_type="relu2",
        rope_base=10000,
    )

    print("[4/5] Training SwiGLU ablation (swiglu + RoPE 10K)...")
    stage_pretrain.remote(
        run_name="picochat-swiglu",
        model_tag="picochat-swiglu",
        mlp_type="swiglu",
        rope_base=10000,
    )

    print("[5/5] Training MTP ablation (relu2 + 1 MTP step, weight=0.3)...")
    stage_pretrain.remote(
        run_name="picochat-mtp",
        model_tag="picochat-mtp",
        mlp_type="relu2",
        rope_base=10000,
        num_mtp_steps=1,
        mtp_loss_weight=0.3,
    )

    print("\n" + "="*60)
    print("All done! Check W&B at wandb.ai/yoyoliuuu/nanochat")
    print("="*60 + "\n")


# =============================================================================
# MAIN ENTRYPOINT: just submits the pipeline and exits immediately
# =============================================================================

@app.local_entrypoint()
def main() -> None:
    """
    Submit the full pipeline to Modal and return immediately.
    The pipeline runs entirely on Modal servers — close your laptop anytime.
    Monitor at: wandb.ai/yoyoliuuu/nanochat  or  modal.com/apps
    """
    print("Submitting pipeline to Modal (runs server-side, safe to close terminal)...")
    run_pipeline.spawn()
    print("Submitted! Monitor at wandb.ai/yoyoliuuu/nanochat")
