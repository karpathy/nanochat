#!/bin/bash
# UPPMAX Pelle setup script for nanochat
# Run this once to set up the environment

set -e

echo "=== Setting up nanochat on UPPMAX Pelle ==="

# Load required modules
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv --system-site-packages
fi

# Activate and install dependencies
source .venv/bin/activate
pip install --upgrade pip

# Install dependencies directly (not editable mode)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install tiktoken numpy tqdm requests transformers datasets wandb

# Add nanochat to Python path
echo "export PYTHONPATH=\"\$PYTHONPATH:$(pwd)\"" >> .venv/bin/activate

echo "=== Setup complete! ==="
echo "To activate: source .venv/bin/activate"
