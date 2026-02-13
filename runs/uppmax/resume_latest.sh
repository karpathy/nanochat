#!/bin/bash

# Helper script to resume training from the latest checkpoint
# Usage: ./resume_latest.sh [depth]

DEPTH=${1:-"20"}
CHECKPOINT_DIR="$HOME/.cache/nanochat/base_checkpoints/d${DEPTH}"

echo "=== Checking for checkpoints in: $CHECKPOINT_DIR ==="

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ No checkpoint directory found at: $CHECKPOINT_DIR"
    echo "Starting fresh training instead..."
    sbatch runs/uppmax/train_d${DEPTH}_checkpoint.sh
    exit 0
fi

# Find the latest checkpoint
LATEST_MODEL=$(ls -1 "$CHECKPOINT_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1)

if [ -z "$LATEST_MODEL" ]; then
    echo "❌ No model checkpoints found in: $CHECKPOINT_DIR"
    echo "Starting fresh training instead..."
    sbatch runs/uppmax/train_d${DEPTH}_checkpoint.sh
    exit 0
fi

# Extract step number from filename (e.g., model_001000.pt -> 1000)
LATEST_STEP=$(basename "$LATEST_MODEL" .pt | sed 's/model_0*//')

echo "✅ Found latest checkpoint at step: $LATEST_STEP"
echo "📁 Checkpoint file: $LATEST_MODEL"

# Check if corresponding meta file exists
META_FILE="$CHECKPOINT_DIR/meta_$(printf "%06d" $LATEST_STEP).json"
if [ ! -f "$META_FILE" ]; then
    echo "❌ Missing meta file: $META_FILE"
    echo "Cannot resume from this checkpoint"
    exit 1
fi

echo "✅ Meta file found: $META_FILE"

# Show some info about the checkpoint
echo ""
echo "=== Checkpoint Info ==="
echo "Step: $LATEST_STEP"
echo "Model file size: $(du -h "$LATEST_MODEL" | cut -f1)"
echo "Created: $(stat -c %y "$LATEST_MODEL" 2>/dev/null || stat -f %Sm "$LATEST_MODEL" 2>/dev/null)"

if command -v jq > /dev/null 2>&1; then
    echo "Training progress:"
    jq -r '.step_count // "N/A"' "$META_FILE" 2>/dev/null | sed 's/^/  Steps completed: /'
    jq -r '.total_training_time // "N/A"' "$META_FILE" 2>/dev/null | sed 's/^/  Training time: /' | sed 's/$/s/'
fi

echo ""
echo "🚀 Resuming training from step $LATEST_STEP..."

# Submit the resume job
sbatch runs/uppmax/train_d${DEPTH}_checkpoint.sh $LATEST_STEP

echo "Job submitted! Check status with: squeue -u $USER"