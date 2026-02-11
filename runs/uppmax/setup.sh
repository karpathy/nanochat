#!/bin/bash
# UPPMAX Pelle setup script for nanochat
# Run this once to set up the environment

set -e

echo "=== Setting up nanochat on UPPMAX Pelle ==="

# Check storage first
echo ""
echo "=== Storage Check ==="
USAGE=$(du -sm ~ 2>/dev/null | cut -f1)
echo "Current home usage: ~${USAGE}MB"
echo "This setup will use ~2-3GB for venv + packages"
echo ""

if [ "$USAGE" -gt 900000 ]; then
    echo "WARNING: You're using >900GB. Consider cleaning up before proceeding."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Load required modules
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

# Remove old venv if exists (to ensure clean install)
if [ -d ".venv" ] && [ "$1" == "--clean" ]; then
    echo "Removing old venv..."
    rm -rf .venv
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate and install dependencies
source .venv/bin/activate
pip install --upgrade pip

# Install PyTorch 2.9.1 with CUDA 12.8 (what nanochat expects)
echo "Installing PyTorch 2.9.1 (CUDA 12.8)..."
pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install nanochat dependencies including rustbpe
echo "Installing nanochat dependencies..."
pip install tiktoken numpy tqdm requests transformers datasets wandb
pip install regex zstandard tabulate scipy psutil
pip install rustbpe  # Required for tokenizer!
pip install fastapi uvicorn  # For web UI

# Add nanochat to Python path
if ! grep -q "PYTHONPATH.*nanochat" .venv/bin/activate; then
    echo "export PYTHONPATH=\"\$PYTHONPATH:$(pwd)\"" >> .venv/bin/activate
fi

echo ""
echo "=== Setup complete! ==="
echo "PyTorch version:"
python -c "import torch; print(f'  torch {torch.__version__}, CUDA {torch.version.cuda}')"
echo ""
echo "Current storage usage:"
du -sh ~/.cache/nanochat 2>/dev/null || echo "  ~/.cache/nanochat: not created yet"
du -sh ~/nanochat/.venv 2>/dev/null || echo "  venv: checking..."
echo ""
echo "To activate: source .venv/bin/activate"
echo "To run test: sbatch runs/uppmax/train_test.sh"
