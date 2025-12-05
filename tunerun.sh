#!/bin/bash
set -e

# This script is designed to auto-tune the system performance for nanochat.
# It uses scripts/tune_system.py to find the best configuration (batch size, compilation flags, etc.)
# and reports the results.

# 1) Example launch:
# bash tunerun.sh
# 2) Example launch in a screen session:
# screen -L -Logfile tunerun.log -S tunerun bash tunerun.sh

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
    # Try to detect if we are on a Strix Halo APU (gfx1151)
    # We use rocminfo if available, or lspci, or fallback to checking if the user set the override.
    IS_STRIX_HALO=0
    if command -v rocminfo &> /dev/null; then
        if rocminfo | grep -q "gfx1151"; then
            IS_STRIX_HALO=1
        fi
    fi

    # If users face issues, they might need to tweak this or use 11.0.0.
    if [ "$IS_STRIX_HALO" -eq 1 ] && [ -z "$HSA_OVERRIDE_GFX_VERSION" ]; then
        export HSA_OVERRIDE_GFX_VERSION=11.5.1
        echo "Exported HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION (Strix Halo detected)"
    fi

    # Disable SDMA to prevent system hangs on Strix Halo APUs
    if [ "$IS_STRIX_HALO" -eq 1 ]; then
        export HSA_ENABLE_SDMA=0
        echo "Exported HSA_ENABLE_SDMA=0 (Strix Halo detected)"
    fi
fi

# -----------------------------------------------------------------------------
# Tokenizer Setup (Needed for base_train to run)

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

# We need at least the tokenizer training to have happened or at least vocab
# But scripts/tune_system.py runs scripts/base_train.py which requires the tokenizer artifacts.
# So we need to ensure the tokenizer is built and trained.
# For simplicity, if tokenizer files don't exist, we run a quick tokenizer training.
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "Tokenizer not found. Training tokenizer on small data subset..."
    python -m nanochat.dataset -n 1 # Just 1 shard is enough for quick setup
    python -m scripts.tok_train --max_chars=10000000 # 10MB chars
fi

# -----------------------------------------------------------------------------
# Run the System Tuner

python -m scripts.tune_system
