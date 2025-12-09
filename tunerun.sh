#!/bin/bash
set -e

# This script is designed to auto-tune the system performance for nanochat.
# It uses scripts/tune_system.py to find the best configuration (batch size, compilation flags, etc.)
# and reports the results.

# 1) Example launch:
# bash tunerun.sh
# 2) Launch with specific profile:
# bash tunerun.sh --profile tiny

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Parse arguments
PROFILE="configs/medium.json" # Default profile

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --profile) PROFILE="configs/$2.json"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ ! -f "$PROFILE" ]; then
    echo "Error: Configuration file $PROFILE not found!"
    exit 1
fi

echo "Using profile: $PROFILE"

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

# Explicitly uninstall triton if present
if [ "$EXTRAS" == "amd" ]; then
    uv pip uninstall -q triton || true
    uv pip install --force-reinstall --index-url https://repo.amd.com/rocm/whl/gfx1151 pytorch-triton-rocm

    ROCM_LLD_PATH=$(python -c "import sysconfig; import os; p = f\"{sysconfig.get_paths()['purelib']}/_rocm_sdk_core/lib/llvm/bin/ld.lld\"; print(p) if os.path.exists(p) else print('')")
    if [ -n "$ROCM_LLD_PATH" ]; then
        export TRITON_HIP_LLD_PATH=$ROCM_LLD_PATH
    fi

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
# Tokenizer Setup (Needed for base_train to run)

if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"

if ! python -c "import rustbpe" &> /dev/null; then
    uv run --no-sync --extra $EXTRAS maturin develop --release --manifest-path rustbpe/Cargo.toml
fi

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "Tokenizer not found. Training tokenizer on small data subset..."
    python -m nanochat.dataset -n 1
    python -m scripts.tok_train --max_chars=10000000
fi

# -----------------------------------------------------------------------------
# Run the System Tuner

python -m scripts.tune_system --config "$PROFILE"
