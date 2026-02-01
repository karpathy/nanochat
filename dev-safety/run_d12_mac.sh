#!/bin/bash
set -euo pipefail

# Run the d12 checkpoint on macOS (CPU or MPS)
# This script runs the SFT checkpoint from sft_checkpoints/d12

echo "=========================================="
echo "Running nanochat d12 on macOS"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANOCHAT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$NANOCHAT_DIR/.." && pwd)"

# Set the base directory so nanochat can find the checkpoints
export NANOCHAT_BASE_DIR="$WORKSPACE_DIR"

# Change to nanochat directory for proper module resolution
cd "$NANOCHAT_DIR"

# Check if virtual environment exists and activate it
if [ -d "$NANOCHAT_DIR/.venv" ]; then
    source "$NANOCHAT_DIR/.venv/bin/activate"
else
    echo "Error: Virtual environment not found at $NANOCHAT_DIR/.venv"
    echo "Please run setup_mac_env.sh first:"
    echo "  cd $NANOCHAT_DIR && ./dev-safety/setup_mac_env.sh"
    exit 1
fi

# Verify checkpoint exists
CHECKPOINT_DIR="$WORKSPACE_DIR/sft_checkpoints/d12"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_DIR"
    echo "Please ensure the d12 checkpoint is in the sft_checkpoints directory"
    exit 1
fi

# Detect available device
echo ""
echo "Detecting available compute device..."
python3 << 'EOF'
import torch
if torch.backends.mps.is_available():
    print("✓ MPS (Metal Performance Shaders) is available - will use Apple Silicon GPU")
elif torch.cuda.is_available():
    print("✓ CUDA is available")
else:
    print("ℹ Running on CPU (this will be slower)")
EOF

echo ""
echo "Starting chat with d12 model..."
echo "Checkpoint: $CHECKPOINT_DIR"
echo ""
echo "Commands:"
echo "  - Type 'quit' or 'exit' to end"
echo "  - Type 'clear' to reset conversation"
echo ""

# Run the chat CLI
# Note: checkpoint_manager.py automatically converts bfloat16 to float32 for MPS/CPU
python3 -m scripts.chat_cli \
    --source sft \
    --model-tag d12 \
    --step 850 \
    --device-type mps
