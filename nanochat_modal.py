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

Run RL on GSM8K (Part 3):
    modal run nanochat_modal.py::run_rl_gsm8k

Cost reference (A10G at ~$1.10/hr):
    picochat d8, 1 GPU: ~1-2 hr per run → ~$1.50-3 per run
    3 runs total: ~$5-9
    RL run: ~1-2 hr → ~$1.50-3

Notes:
    - Data and tokenizer are cached in a persistent Modal Volume.
      All 3 runs share the same data/tokenizer — download only happens once.
    - Stages are idempotent where possible.
    - The nanochat repo is copied into the container image at build time.
      If you change gpt.py or base_train.py, Modal auto-rebuilds the image.
    - W&B runs go to the yoyoliuuu workspace under project "nanochat".
      Each run is tagged: picochat-baseline, picochat-swiglu, picochat-rope500k.
    - RL eval logs are saved to /vol/nanochat_cache/chatrl_eval_logs/<run_name>/

Reference: Angela Sha, https://github.com/UofT-CSC490-W2026/022326-tutorial-nanochat
"""

import os
import subprocess
import modal
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
# Runs appear in wandb.ai/yoyoliuuu/nanochat
WANDB_ENTITY = "yoyoliuuu"

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
        "WANDB_ENTITY": WANDB_ENTITY,
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


# =============================================================================
# PART 3: RL TRAINING ON GSM8K
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=2,
    timeout=60 * 30,  # 30 min for download
)
def stage_download_sft_checkpoint(
    hf_repo: str = "alvina-yang/csc490a4p2",
    checkpoint_name: str = "sft-baseline",
    step: int = 32146,
) -> None:
    """Download SFT checkpoint .pt files from HuggingFace into the Modal volume.
    Pass step=0 to auto-detect the step from common candidates."""
    import urllib.request
    dest_dir = f"{NANOCHAT_CACHE}/chatsft_checkpoints/{checkpoint_name}"
    os.makedirs(dest_dir, exist_ok=True)
    base_url = f"https://huggingface.co/{hf_repo}/resolve/main/{checkpoint_name}"
    # Auto-detect step if not specified
    if step == 0:
        for try_step in [50000, 40000, 32146, 30000, 25000, 20000, 15530, 15000, 10000, 5000]:
            meta_url = f"{base_url}/meta_{try_step:06d}.json"
            dest_meta = os.path.join(dest_dir, f"meta_{try_step:06d}.json")
            try:
                urllib.request.urlretrieve(meta_url, dest_meta)
                step = try_step
                print(f"Auto-detected step {step} for {checkpoint_name}")
                break
            except Exception:
                if os.path.exists(dest_meta):
                    os.remove(dest_meta)
        assert step != 0, f"Could not auto-detect step for {checkpoint_name} in {hf_repo}"
    files = [
        f"model_{step:06d}.pt",
        f"meta_{step:06d}.json",
    ]
    for fname in files:
        dest = os.path.join(dest_dir, fname)
        if os.path.exists(dest):
            print(f"Already exists, skipping: {dest}")
            continue
        url = f"{base_url}/{fname}"
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  Done: {size_mb:.1f} MB")
    volume.commit()
    print(f"Volume committed. Checkpoint: {checkpoint_name} step={step}")


@app.local_entrypoint()
def download_sft_checkpoint() -> None:
    """Download sft-baseline (d8) checkpoint from HuggingFace to the Modal volume."""
    stage_download_sft_checkpoint.remote()


@app.local_entrypoint()
def download_sft_checkpoint_d12() -> None:
    """Download sft-baseline-d12 checkpoint from HuggingFace to the Modal volume."""
    stage_download_sft_checkpoint.remote(
        checkpoint_name="sft-baseline-d12",
        step=15530,
    )



@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU,
    timeout=TRAIN_TIMEOUT_SEC,
)
def stage_rl(
    run_name: str = "rl-gsm8k-rep1",
    model_tag: str = "sft-baseline",
    model_step: int = 32146,
    num_epochs: int = 1,
    device_batch_size: int = 8,
    examples_per_step: int = 16,
    num_samples: int = 8,
    max_new_tokens: int = 256,
    eval_every: int = 60,
    eval_examples: int = 400,
    save_every: int = 60,
) -> None:
    """
    Run RL (GRPO-style REINFORCE) on GSM8K starting from an SFT checkpoint.

    Args:
        run_name:          W&B run name (logged to nanochat-rl project)
        model_tag:         SFT checkpoint directory name under chatsft_checkpoints/
                           chat_sft.py saves to chatsft_checkpoints/{base_model_tag}/
                           so for sft-baseline (trained from picochat-ctx-s1) this is
                           "picochat-ctx-s1".
        model_step:        exact step of the SFT checkpoint to load (e.g. 32146)
        num_epochs:        number of passes over GSM8K training set
        device_batch_size: max batch size per forward pass (also sets pass@k K)
        examples_per_step: total examples per optimizer step
        num_samples:       rollout samples per question
        max_new_tokens:    max tokens to generate per sample
        eval_every:        evaluate pass@k every N steps
        eval_examples:     number of GSM8K test problems for evaluation
        save_every:        checkpoint every N steps
    """
    _setup_cache()
    print(f"\n{'='*60}")
    print(f"RL on GSM8K: {run_name}")
    print(f"  SFT init: {model_tag} step={model_step}  epochs={num_epochs}")
    print(f"  device_batch_size={device_batch_size}  num_samples={num_samples}")
    print(f"{'='*60}\n")

    _torchrun(
        "scripts.chat_rl",
        [
            f"--run={run_name}",
            f"--model-tag={model_tag}",
            f"--model-step={model_step}",
            f"--num-epochs={num_epochs}",
            f"--device-batch-size={device_batch_size}",
            f"--examples-per-step={examples_per_step}",
            f"--num-samples={num_samples}",
            f"--max-new-tokens={max_new_tokens}",
            f"--eval-every={eval_every}",
            f"--eval-examples={eval_examples}",
            f"--save-every={save_every}",
        ],
        nproc=1,
    )
    volume.commit()
    print(f"Done: {run_name}")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu="H100:2",
    timeout=TRAIN_TIMEOUT_SEC,
)
def stage_rl_d12(
    run_name: str = "rl-gsm8k-d12",
    model_tag: str = "sft-baseline-d12",
    model_step: int = 15530,
    num_epochs: int = 1,
    device_batch_size: int = 8,
    examples_per_step: int = 64,
    num_samples: int = 8,
    max_new_tokens: int = 512,
    eval_every: int = 60,
    eval_examples: int = 400,
    save_every: int = 60,
) -> None:
    """RL on GSM8K from the nanochat d12 SFT checkpoint, 4x H100."""
    _setup_cache()
    print(f"\n{'='*60}")
    print(f"RL on GSM8K (d12, 4xH100): {run_name}")
    print(f"  SFT init: {model_tag} step={model_step}  epochs={num_epochs}")
    print(f"  device_batch_size={device_batch_size}  num_samples={num_samples}")
    print(f"{'='*60}\n")

    _torchrun(
        "scripts.chat_rl",
        [
            f"--run={run_name}",
            f"--model-tag={model_tag}",
            f"--model-step={model_step}",
            f"--num-epochs={num_epochs}",
            f"--device-batch-size={device_batch_size}",
            f"--examples-per-step={examples_per_step}",
            f"--num-samples={num_samples}",
            f"--max-new-tokens={max_new_tokens}",
            f"--eval-every={eval_every}",
            f"--eval-examples={eval_examples}",
            f"--save-every={save_every}",
        ],
        nproc=2,
    )
    volume.commit()
    print(f"Done: {run_name}")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=1,
    timeout=60,
)
def _ls_checkpoints() -> None:
    """List all checkpoint files on the volume to diagnose missing .pt files."""
    import glob as _glob
    for pattern in [
        f"{NANOCHAT_CACHE}/chatsft_checkpoints/**/*",
        f"{NANOCHAT_CACHE}/base_checkpoints/**/*",
    ]:
        files = _glob.glob(pattern, recursive=True)
        for f in sorted(files):
            print(f)

@app.local_entrypoint()
def ls_checkpoints() -> None:
    """Run: modal run nanochat_modal.py::ls_checkpoints"""
    _ls_checkpoints.remote()


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=1,
    timeout=60,
)
def _peek_eval(run_name: str, step: int = 0, n: int = 5) -> None:
    """Print first n generated texts from an eval log to inspect model output format."""
    import json, os
    path = f"{NANOCHAT_CACHE}/chatrl_eval_logs/{run_name}/eval_step_{step:06d}.json"
    if not os.path.exists(path):
        print(f"Not found: {path}")
        return
    with open(path) as f:
        records = json.load(f)
    if isinstance(records, dict):
        records = list(records.values())
    for i, record in enumerate(records[:n]):
        print(f"\n--- Example {i} ---")
        print(f"Question: {record.get('question', '')[:200]}")
        for j, outcome in enumerate(record.get("outcomes", [])[:2]):
            print(f"  Sample {j}: correct={outcome['is_correct']}")
            print(f"  Generated: {repr(outcome['generated_text'][:300])}")

@app.local_entrypoint()
def peek_eval() -> None:
    """Inspect model outputs from eval log. Run: modal run nanochat_modal.py::peek_eval"""
    _peek_eval.remote("rl-gsm8k-d12", step=0, n=5)


@app.local_entrypoint()
def run_rl_gsm8k() -> None:
    """
    Run RL on GSM8K starting from the sft-baseline SFT checkpoint.
    Logs to W&B project nanochat-rl under run name rl-gsm8k-rep1.
    Eval logs (generated text + correctness) saved to the Modal volume at:
      /vol/nanochat_cache/chatrl_eval_logs/rl-gsm8k-rep1/eval_step_XXXXXX.json
    """
    stage_rl.remote(
        run_name="rl-gsm8k-rep1",
        model_tag="sft-baseline",     # chatsft_checkpoints/sft-baseline/ on the volume
        model_step=32146,             # sft-baseline checkpoint step (see a4p2_checkpoints/sft-baseline/meta_032146.json)
        num_epochs=1,
        device_batch_size=8,
        examples_per_step=16,
        num_samples=8,
        eval_every=60,
        eval_examples=400,
        save_every=60,
    )


@app.local_entrypoint()
def run_rl_gsm8k_d12() -> None:
    """
    Run RL on GSM8K from the nanochat d12 SFT checkpoint (sft-baseline-d12, step 15530).
    Uses 4x H100 GPUs. Run download_sft_checkpoint_d12 first if needed.
    Logs to W&B project nanochat-rl under run name rl-gsm8k-d12.
    """
    stage_rl_d12.remote()


# =============================================================================
# TEAMMATE RL RUN — edit TEAMMATE_* vars below then run run_rl_teammate
# =============================================================================

# ── Fill these in before running ─────────────────────────────────────────────
TEAMMATE_HF_REPO        = "alvina-yang/csc490a4p2"   # HuggingFace repo
TEAMMATE_CHECKPOINT     = "sft-teammate"              # folder name in the HF repo (d24 SFT baseline)
TEAMMATE_STEP           = 483                         # sft-baseline-d24 step
TEAMMATE_RUN_NAME       = "rl-gsm8k-teammate"        # W&B run name + eval log folder
TEAMMATE_GPU            = "H100:2"                   # 2x H100 as requested
TEAMMATE_EPOCHS         = 1
TEAMMATE_DEVICE_BATCH   = 8
TEAMMATE_EXAMPLES_STEP  = 64
TEAMMATE_NUM_SAMPLES    = 8
TEAMMATE_MAX_NEW_TOKENS = 512
# ─────────────────────────────────────────────────────────────────────────────


@app.local_entrypoint()
def run_rl_teammate() -> None:
    """
    Step 1: download SFT checkpoint from HuggingFace to the Modal volume.
    Step 2: run RL training.

    Edit TEAMMATE_* constants above, then:
        uv run modal run --detach nanochat_modal.py::run_rl_teammate
    """
    # Download checkpoint from HuggingFace
    stage_download_sft_checkpoint.remote(
        hf_repo=TEAMMATE_HF_REPO,
        checkpoint_name=TEAMMATE_CHECKPOINT,
        step=TEAMMATE_STEP or 0,   # 0 triggers auto-detect inside stage_download
    )
    # Run RL on H100:4 (same as stage_rl_d12 but with teammate config)
    stage_rl_d12.remote(
        run_name=TEAMMATE_RUN_NAME,
        model_tag=TEAMMATE_CHECKPOINT,
        model_step=TEAMMATE_STEP,
        num_epochs=TEAMMATE_EPOCHS,
        device_batch_size=TEAMMATE_DEVICE_BATCH,
        examples_per_step=TEAMMATE_EXAMPLES_STEP,
        num_samples=TEAMMATE_NUM_SAMPLES,
        max_new_tokens=TEAMMATE_MAX_NEW_TOKENS,
        eval_every=60,
        eval_examples=400,
        save_every=60,
    )


@app.local_entrypoint()
def download_eval_logs() -> None:
    """
    Download all eval log JSONs from the Modal volume to ./eval_logs/<run_name>/ locally.
    Edit run_name below to match the run you want.
    Run: uv run modal run nanochat_modal.py::download_eval_logs
    """
    _download_eval_logs.remote(run_name=TEAMMATE_RUN_NAME)


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=1,
    timeout=120,
)
def _download_eval_logs(run_name: str) -> dict:
    """Return all eval log JSONs as a dict keyed by filename."""
    import json as _json, glob as _glob
    log_dir = f"{NANOCHAT_CACHE}/chatrl_eval_logs/{run_name}"
    files = sorted(_glob.glob(f"{log_dir}/eval_step_*.json"))
    if not files:
        print(f"No eval logs found at {log_dir}")
        return {}
    result = {}
    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath) as f:
            result[fname] = _json.load(f)
        print(f"  Loaded {fname}")
    return result


# =============================================================================
# INFERENCE + HF PUSH
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=2,
    timeout=60 * 15,
)
def stage_download_for_inference(
    hf_repo: str = "alvina-yang/csc490a4p2",
    base_checkpoint_name: str = "nanochat-d12",
    sft_checkpoint_name: str = "sft-baseline-d12",
    sft_step: int = 15530,
) -> int:
    """Download base + SFT checkpoints from HF; returns detected base step."""
    import urllib.request

    def dl(dest_dir, base_url, fnames):
        os.makedirs(dest_dir, exist_ok=True)
        for fname in fnames:
            dest = os.path.join(dest_dir, fname)
            if os.path.exists(dest):
                print(f"  Cached: {dest}")
                continue
            url = f"{base_url}/{fname}"
            print(f"  Downloading {url}")
            urllib.request.urlretrieve(url, dest)
            print(f"    {os.path.getsize(dest)/1e6:.1f} MB")

    # Detect base step
    base_dir = f"{NANOCHAT_CACHE}/base_checkpoints/{base_checkpoint_name}"
    base_url = f"https://huggingface.co/{hf_repo}/resolve/main/{base_checkpoint_name}"
    base_step = None
    os.makedirs(base_dir, exist_ok=True)
    for try_step in [15530, 16000, 14000, 12000, 10000, 8000, 6000, 4000]:
        meta_fname = f"meta_{try_step:06d}.json"
        dest_meta = os.path.join(base_dir, meta_fname)
        try:
            urllib.request.urlretrieve(f"{base_url}/{meta_fname}", dest_meta)
            base_step = try_step
            print(f"Found base checkpoint at step {base_step}")
            break
        except Exception:
            if os.path.exists(dest_meta):
                os.remove(dest_meta)
    if base_step is not None:
        dl(base_dir, base_url, [f"model_{base_step:06d}.pt", f"meta_{base_step:06d}.json"])
    else:
        print("WARNING: could not find base checkpoint on HuggingFace")

    # Download SFT
    sft_dir = f"{NANOCHAT_CACHE}/chatsft_checkpoints/{sft_checkpoint_name}"
    sft_url = f"https://huggingface.co/{hf_repo}/resolve/main/{sft_checkpoint_name}"
    dl(sft_dir, sft_url, [f"model_{sft_step:06d}.pt", f"meta_{sft_step:06d}.json"])
    volume.commit()
    return base_step or -1


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu="A10G:1",
    timeout=60 * 20,
)
def stage_inference(
    base_checkpoint_name: str = "nanochat-d12",
    base_step: int = None,
    sft_checkpoint_name: str = "sft-baseline-d12",
    sft_step: int = 15530,
    n_examples: int = 5,
    max_new_tokens: int = 512,
) -> None:
    """Run inference via uv subprocess (torch lives in uv venv, not system Python)."""
    def run(source, tag, step):
        step_flag = f"--step={step}" if step else ""
        _run(
            f"cd /root/nanochat && uv run python -m scripts.chat_inference"
            f" --source={source} --model-tag={tag} {step_flag}"
            f" --n={n_examples} --max-new-tokens={max_new_tokens}"
        )
    if base_step and base_step > 0:
        print(f"\n{'='*60}\nnanochat-d12 BASE (step {base_step})\n{'='*60}")
        run("base", base_checkpoint_name, base_step)
    print(f"\n{'='*60}\nnanochat-d12 SFT (step {sft_step})\n{'='*60}")
    run("sft", sft_checkpoint_name, sft_step)


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=2,
    timeout=60 * 30,
)
def stage_push_rl_checkpoint(
    run_name: str = "rl-gsm8k-d12",
    hf_repo: str = "alvina-yang/csc490a4p2",
    hf_folder: str = "rl-gsm8k-d12",
) -> None:
    """Push a trained RL checkpoint directory to HuggingFace via HTTP API."""
    import glob as _glob
    import urllib.request
    hf_token = os.environ.get("HF_TOKEN")
    assert hf_token, "HF_TOKEN not set in secrets"
    rl_dir = f"{NANOCHAT_CACHE}/chatrl_checkpoints/{run_name}"
    files = sorted(_glob.glob(os.path.join(rl_dir, "*")))
    if not files:
        print(f"No files found in {rl_dir}")
        return
    print(f"Uploading {len(files)} files from {rl_dir} to {hf_repo}/{hf_folder}/")
    for fpath in files:
        fname = os.path.basename(fpath)
        size_mb = os.path.getsize(fpath) / 1e6
        print(f"  Uploading {fname} ({size_mb:.1f} MB)...")
        upload_url = f"https://huggingface.co/api/models/{hf_repo}/upload/main/{hf_folder}/{fname}"
        with open(fpath, "rb") as f:
            data = f.read()
        req = urllib.request.Request(
            upload_url,
            data=data,
            method="POST",
            headers={"Authorization": f"Bearer {hf_token}", "Content-Type": "application/octet-stream"},
        )
        with urllib.request.urlopen(req) as resp:
            print(f"    HTTP {resp.status}")
    print("Done.")


@app.local_entrypoint()
def run_inference() -> None:
    """Run inference on nanochat-d12 base + SFT. modal run nanochat_modal.py::run_inference"""
    base_step = stage_download_for_inference.remote()
    stage_inference.remote(base_step=base_step)


@app.local_entrypoint()
def push_rl_to_hf() -> None:
    """Push trained RL checkpoint to HuggingFace. modal run nanochat_modal.py::push_rl_to_hf"""
    stage_push_rl_checkpoint.remote()
