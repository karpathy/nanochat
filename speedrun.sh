#!/bin/bash
set -e

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# bash speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
CONFIG_FILE="configs/speedrun.json"

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
if [ "$EXTRAS" == "amd" ]; then
    uv pip uninstall -q triton || true
    uv pip install --force-reinstall --index-url https://repo.amd.com/rocm/whl/gfx1151 pytorch-triton-rocm

    # Find and export the path to ld.lld from rocm-sdk-core if available
    ROCM_LLD_PATH=$(python -c "import sysconfig; import os; p = f\"{sysconfig.get_paths()['purelib']}/_rocm_sdk_core/lib/llvm/bin/ld.lld\"; print(p) if os.path.exists(p) else print('')")
    if [ -n "$ROCM_LLD_PATH" ]; then
        export TRITON_HIP_LLD_PATH=$ROCM_LLD_PATH
    fi

    # AMD Strix Halo / APU specific settings
    IS_STRIX_HALO=0
    if command -v rocminfo &> /dev/null; then
        if rocminfo | grep -q "gfx1151"; then
            IS_STRIX_HALO=1
        fi
    fi

    if [ "$IS_STRIX_HALO" -eq 1 ] && [ -z "$HSA_OVERRIDE_GFX_VERSION" ]; then
        export HSA_OVERRIDE_GFX_VERSION=11.5.1
    fi

    if [ "$IS_STRIX_HALO" -eq 1 ]; then
        export HSA_ENABLE_SDMA=0
    fi
fi

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# Initialize report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
uv run --no-sync --extra $EXTRAS maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download the first ~2B characters of pretraining dataset
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
if python -c "import torch; exit(0) if torch.cuda.is_available() or (hasattr(torch.version, 'hip') and torch.version.hip) else exit(1)"; then
    NPROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count())")
    if [ "$NPROC_PER_NODE" -eq "0" ]; then
        NPROC_PER_NODE=1
    else
        echo "Detected $NPROC_PER_NODE GPUs."
    fi
else
    NPROC_PER_NODE=1
    unset OMP_NUM_THREADS
fi

# -----------------------------------------------------------------------------
# Training

# Using config file
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train $CONFIG_FILE --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss $CONFIG_FILE
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train $CONFIG_FILE --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# Supervised Finetuning

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft $CONFIG_FILE --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

python -m nanochat.report generate
