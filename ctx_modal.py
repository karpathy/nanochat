"""
Context window extension via sequence-length curriculum learning.

Setup (one-time):
    modal setup
    modal secret create nanochat-secrets \\
        WANDB_API_KEY=<your_key> \\
        HF_TOKEN=hf_<your_token> \\
        WANDB_ENTITY=<your_wandb_username>

Run full pipeline (data + tokenizer + Stage 1 + Stage 2 + evals):
    modal run ctx_modal.py

Run individual stages:
    modal run ctx_modal.py::run_ctx_stage1
    modal run ctx_modal.py::run_ctx_stage2
    modal run ctx_modal.py::run_ctx_evals

Cost reference (A10G at ~$1.10/hr):
    Stage 1 (3000 steps, seq_len=512): ~30-45 min → ~$0.60
    Stage 2 (2000 steps, seq_len=2048): ~45-60 min → ~$0.90
    Evals: ~10 min each → ~$0.30
    Total: ~$2
"""

import os
import subprocess
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION
# =============================================================================

DEPTH = 8
DEVICE_BATCH_SIZE_SHORT = 16   # seq_len=512
DEVICE_BATCH_SIZE_LONG = 4     # seq_len=2048
MODEL_TAG = "picochat-ctx-s1"

GPU = "A10G:1"
NUM_SHARDS = 12

VOLUME_MOUNT = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"

TRAIN_TIMEOUT_SEC = 60 * 60 * 4
DOWNLOAD_TIMEOUT_SEC = 60 * 60
CORE_METRIC_EVERY = -1

# =============================================================================
# MODAL PRIMITIVES
# =============================================================================

app = App("nanochat-ctx-extension")

volume = Volume.from_name("nanochat-vol", create_if_missing=True)
secret = Secret.from_name("nanochat-secrets")

image = (
    ModalImage.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git", "build-essential", "curl", "wget", "unzip")
    .add_local_dir(
        local_path=".",
        remote_path="/root/nanochat",
        copy=True,
        ignore=[".venv", "__pycache__", "*.pyc", ".git", "rustbpe/target", "runs"],
    )
    .workdir("/root/nanochat")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "echo 'export PATH=\"$HOME/.local/bin:$PATH\"' >> $HOME/.bashrc",
    )
    .pip_install("uv")
    .run_commands("uv sync --extra gpu --no-install-project")
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
    os.makedirs(NANOCHAT_CACHE, exist_ok=True)


# =============================================================================
# DATA + TOKENIZER (shared with ablation runs via volume)
# =============================================================================

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    cpu=8, memory=16384, timeout=DOWNLOAD_TIMEOUT_SEC,
)
def stage_data(num_shards: int = NUM_SHARDS) -> None:
    """Download FineWeb-EDU shards."""
    _setup_cache()
    print(f"Downloading {num_shards} FineWeb-EDU shards to volume...")
    _python("nanochat.dataset", [f"-n {num_shards}"])
    volume.commit()


@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    gpu="A10G:1", timeout=60 * 30,
)
def stage_tokenizer() -> None:
    """Train the BPE tokenizer."""
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
# TRAINING
# =============================================================================

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    gpu=GPU, timeout=TRAIN_TIMEOUT_SEC,
)
def stage_ctx_train(
    run_name: str,
    model_tag: str,
    max_seq_len: int,
    device_batch_size: int,
    num_iterations: int,
    warmdown_ratio: float,
    save_every: int = 500,
    resume_from_step: int = -1,
    no_load_optimizer: bool = False,
) -> None:
    """Train a picochat model for the context extension experiment."""
    _setup_cache()
    print(f"\n{'='*60}")
    print(f"Context extension: {run_name}")
    print(f"  seq_len={max_seq_len}  depth={DEPTH}  iters={num_iterations}")
    print(f"  resume={resume_from_step}  no_load_opt={no_load_optimizer}")
    print(f"{'='*60}\n")

    _python("nanochat.report", ["reset"])

    args = [
        f"--depth={DEPTH}",
        "--head-dim=64",
        "--window-pattern=L",
        f"--max-seq-len={max_seq_len}",
        f"--device-batch-size={device_batch_size}",
        "--total-batch-size=16384",
        f"--num-iterations={num_iterations}",
        f"--warmdown-ratio={warmdown_ratio}",
        "--eval-every=100",
        "--eval-tokens=524288",
        f"--core-metric-every={CORE_METRIC_EVERY}",
        "--sample-every=-1",
        f"--save-every={save_every}",
        f"--model-tag={model_tag}",
        f"--run={run_name}",
    ]
    if resume_from_step >= 0:
        args.append(f"--resume-from-step={resume_from_step}")
    if no_load_optimizer:
        args.append("--no-load-optimizer")

    _torchrun("scripts.base_train", args)
    volume.commit()
    print(f"Done: {run_name}")


# =============================================================================
# EVALUATION
# =============================================================================

@app.function(
    image=image, secrets=[secret], volumes={VOLUME_MOUNT: volume},
    gpu=GPU, timeout=TRAIN_TIMEOUT_SEC,
)
def stage_ctx_eval(
    run_name: str,
    model_tag: str,
    step: int | None = None,
    seq_lens: str = "128,256,512,1024,2048",
    device_batch_size: int = 2,
) -> None:
    """Run multi-length BPB evaluation."""
    _setup_cache()
    print(f"\n{'='*60}")
    print(f"Context eval: {run_name} ({model_tag})")
    print(f"{'='*60}\n")

    args = [
        f"--model-tag={model_tag}",
        f"--seq-lens={seq_lens}",
        f"--device-batch-size={device_batch_size}",
        f"--run={run_name}",
    ]
    if step is not None:
        args.append(f"--step={step}")

    _python("scripts.ctx_eval", args)
    volume.commit()
    print(f"Done: {run_name}")


# =============================================================================
# ENTRYPOINTS
# =============================================================================

@app.local_entrypoint()
def run_ctx_stage1() -> None:
    """Stage 1: train at seq_len=512 for 3000 steps (Checkpoint 1)."""
    stage_ctx_train.remote(
        run_name="picochat-ctx-stage1",
        model_tag=MODEL_TAG,
        max_seq_len=512,
        device_batch_size=DEVICE_BATCH_SIZE_SHORT,
        num_iterations=3000,
        warmdown_ratio=0.0,
        save_every=500,
    )


@app.local_entrypoint()
def run_ctx_stage2() -> None:
    """Stage 2: extend to seq_len=2048, resume from step 3000 (Checkpoint 2)."""
    stage_ctx_train.remote(
        run_name="picochat-ctx-stage2",
        model_tag=MODEL_TAG,
        max_seq_len=2048,
        device_batch_size=DEVICE_BATCH_SIZE_LONG,
        num_iterations=5000,
        warmdown_ratio=0.5,
        save_every=500,
        resume_from_step=3000,
        no_load_optimizer=True,
    )


@app.local_entrypoint()
def run_ctx_evals() -> None:
    """Eval both checkpoints at multiple sequence lengths."""
    stage_ctx_eval.remote(
        run_name="ctx-eval-checkpoint1",
        model_tag=MODEL_TAG,
        step=3000,
    )
    stage_ctx_eval.remote(
        run_name="ctx-eval-checkpoint2",
        model_tag=MODEL_TAG,
        step=5000,
    )


@app.local_entrypoint()
def main() -> None:
    """
    Full context extension pipeline:
      1. Data + tokenizer (idempotent)
      2. Stage 1: seq_len=512, 3000 steps → Checkpoint 1
      3. Stage 2: seq_len=2048, resume from 3000 → Checkpoint 2
      4. Eval both checkpoints at multiple lengths
    """
    print("\n" + "="*60)
    print("Context Extension Experiment")
    print("="*60 + "\n")

    print("[1/4] Downloading data shards...")
    stage_data.remote(num_shards=NUM_SHARDS)

    print("[2/4] Training tokenizer...")
    stage_tokenizer.remote()

    print("[3/4] Stage 1: training at seq_len=512 for 3000 steps...")
    stage_ctx_train.remote(
        run_name="picochat-ctx-stage1",
        model_tag=MODEL_TAG,
        max_seq_len=512,
        device_batch_size=DEVICE_BATCH_SIZE_SHORT,
        num_iterations=3000,
        warmdown_ratio=0.0,
        save_every=500,
    )

    print("[4/4] Stage 2: extending to seq_len=2048 from step 3000...")
    stage_ctx_train.remote(
        run_name="picochat-ctx-stage2",
        model_tag=MODEL_TAG,
        max_seq_len=2048,
        device_batch_size=DEVICE_BATCH_SIZE_LONG,
        num_iterations=5000,
        warmdown_ratio=0.5,
        save_every=500,
        resume_from_step=3000,
        no_load_optimizer=True,
    )

    print("[eval] Evaluating Checkpoint 1 (step 3000) and Checkpoint 2 (step 5000)...")
    stage_ctx_eval.remote(
        run_name="ctx-eval-checkpoint1",
        model_tag=MODEL_TAG,
        step=3000,
    )
    stage_ctx_eval.remote(
        run_name="ctx-eval-checkpoint2",
        model_tag=MODEL_TAG,
        step=5000,
    )

    print("\n" + "="*60)
    print("All done! Check W&B for results.")
    print("="*60 + "\n")
