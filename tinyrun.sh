#!/bin/bash
set -e

# Many would like to run nanochat on a a single GPU or a tiny cluster.
# This script is the "Best ChatGPT clone that a crappy single GPU can buy",
# It is designed to run in ~1 hour on a single 3080 GPU with 10GB of VRAM.
# This will help you get started. The model will be bad, terribly bad, but helps you to get started.
# Comments are sparse, see speedrun.sh for more detail.

# 1) Example launch (simplest):
# bash tinyrun.sh

# 2) Example launch in a screen session (because the run takes ~1 hour):
# screen -L -Logfile tinyrun.log -S tinyrun bash tinyrun.sh

# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=tinyrun screen -L -Logfile tinyrun.log -S tinyrun bash tinyrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv

# install the repo dependencies
# Detect hardware to install the correct torch version
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing CUDA dependencies..."
    EXTRAS="gpu"
elif [ -e /dev/kfd ]; then
    echo "AMD GPU detected. Installing ROCm dependencies..."
    EXTRAS="amd"
else
    echo "No dedicated GPU detected. Installing CPU dependencies..."
    EXTRAS="cpu"
fi
uv sync --extra $EXTRAS

# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# Explicitly uninstall triton if present, as it conflicts with pytorch-triton-rocm
# and can cause "ImportError: cannot import name 'Config' from 'triton'" errors
# if the NVIDIA version of triton (e.g. 3.4.0) is accidentally installed.
if [ "$EXTRAS" == "amd" ]; then
    uv pip uninstall -q triton || true
    # Uninstalling triton may have deleted the shared 'triton' directory, breaking pytorch-triton-rocm.
    # Reinstall pytorch-triton-rocm to ensure it's intact.
    uv pip install --force-reinstall --index-url https://repo.amd.com/rocm/whl/gfx1151 pytorch-triton-rocm

    # Find and export the path to ld.lld from rocm-sdk-core if available, as torch.compile/triton needs it
    ROCM_LLD_PATH=$(python -c "import sysconfig; import os; p = f\"{sysconfig.get_paths()['purelib']}/_rocm_sdk_core/lib/llvm/bin/ld.lld\"; print(p) if os.path.exists(p) else print('')")
    if [ -n "$ROCM_LLD_PATH" ]; then
        export TRITON_HIP_LLD_PATH=$ROCM_LLD_PATH
        echo "Exported TRITON_HIP_LLD_PATH=$TRITON_HIP_LLD_PATH"
    fi

    # AMD Strix Halo / APU specific settings
    # Many APUs need this override to work with ROCm if the reported GFX version
    # doesn't match the installed kernels exactly. Strix Halo is gfx1151.
    # If users face issues, they might need to tweak this or use 11.0.0.
    if [ -z "$HSA_OVERRIDE_GFX_VERSION" ]; then
        export HSA_OVERRIDE_GFX_VERSION=11.5.1
        echo "Exported HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION (Strix Halo/APU compat)"
    fi

    # Disable SDMA to prevent system hangs on some APUs (common fix for Ryzen AI)
    export HSA_ENABLE_SDMA=0
    echo "Exported HSA_ENABLE_SDMA=0 (APU stability)"
fi

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# Initialize report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo (if not already installed)
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer (if not already built)
# use --no-sync to avoid re-installing triton on AMD
if ! python -c "import rustbpe" &> /dev/null; then
    uv run --no-sync --extra $EXTRAS maturin develop --release --manifest-path rustbpe/Cargo.toml
fi

# Download the dataset for pretraining
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval

echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# -----------------------------------------------------------------------------
# Process Setup

# Number of processes/GPUs to use
# Auto-detect if we have GPUs (including ROCm)
if python -c "import torch; exit(0) if torch.cuda.is_available() or (hasattr(torch.version, 'hip') and torch.version.hip) else exit(1)"; then
    # Detected GPU capability. Now get the actual count to avoid launching too many processes.
    NPROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
    # If for some reason it returns 0 (e.g. weird ROCm state), default to 1 to be safe.
    if [ "$NPROC_PER_NODE" -eq "0" ]; then
        echo "GPU detected but torch.cuda.device_count() is 0. Defaulting to NPROC_PER_NODE=1."
        NPROC_PER_NODE=1
    else
        echo "Detected $NPROC_PER_NODE GPUs."
    fi
else
    echo "No GPU detected. Defaulting to NPROC_PER_NODE=1 to avoid OOM and using multi-threading."
    NPROC_PER_NODE=1
    # If running on CPU, let PyTorch use all available cores for the single process
    unset OMP_NUM_THREADS
fi

# -----------------------------------------------------------------------------
# Training

# Train the base model on rather smaller parameters to get a sense of the code.
# depth=the depth of the Transformer model to train
# max_seq_len=max context length
# device_batch_size=per-device batch size (set to not OOM)
# eval_tokens=number of tokens to evaluate val loss on
# core_metric_every=every how many steps to evaluate the core metric (-1 = disable)
# total_batch_size=total desired batch size, in #tokens
# num_iterations=explicit number of steps of the optimization (-1 = disable)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss -- --device_batch_size=1 --split_tokens=512
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- --max-per-task=16

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_sft_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --device_batch_size=1 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

python -m nanochat.report generate
