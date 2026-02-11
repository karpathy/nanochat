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
echo "This setup will use ~500MB for venv + packages"
echo ""

if [ "$USAGE" -gt 900000 ]; then
    echo "WARNING: You're using >900GB. Consider cleaning up before proceeding."
    echo "Run: du -sh ~/.cache/* ~/.*/ | sort -h"
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

echo ""
echo "=== Setup complete! ==="
echo "Current storage usage:"
du -sh ~/.cache/nanochat 2>/dev/null || echo "  ~/.cache/nanochat: not created yet"
du -sh ~/nanochat/.venv 2>/dev/null || echo "  ~/nanochat/.venv: not found"
echo ""
echo "To activate: source .venv/bin/activate"
