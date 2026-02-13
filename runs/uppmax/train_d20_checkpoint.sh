#!/bin/bash
#SBATCH -A uppmax2025-2-290
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -t 24:00:00
#SBATCH -J nanochat-d20-ckpt
#SBATCH --output=nanochat-d20-ckpt-%j.out
#SBATCH --error=nanochat-d20-ckpt-%j.err

# nanochat depth=20 training with checkpointing (~12-15 hours on L40s)
# Saves checkpoints every 2 hours to enable resuming from failures
# Usage: sbatch train_d20_checkpoint.sh [resume_step]

set -e

echo "=== nanochat depth=20 training with checkpointing ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"

# Resume step (optional argument)
RESUME_STEP=${1:-""}

# Storage check
echo ""
echo "=== Storage Check (before) ==="
du -sh ~ 2>/dev/null | awk '{print "Home total: " $1}'
du -sh ~/.cache/nanochat 2>/dev/null | awk '{print "nanochat cache: " $1}' || echo "nanochat cache: 0"
echo ""

# Load modules (Pelle-specific)
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

# Set up environment
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTHONPATH="$PYTHONPATH:$(pwd)"
mkdir -p $NANOCHAT_BASE_DIR

# Load wandb API key if available
if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat $HOME/.wandb_key)
    echo "wandb API key loaded"
    WANDB_RUN="nanochat-d20-ckpt-$SLURM_JOB_ID"
else
    WANDB_RUN="dummy"
    echo "No wandb key found, logging disabled"
fi

cd ~/nanochat
source .venv/bin/activate

# Download data (20 shards for depth=20, ~2GB) - only if not resuming
if [[ -z "$RESUME_STEP" ]]; then
    echo "=== Downloading data (20 shards) ==="
    python -m nanochat.dataset -n 20
    
    # Train tokenizer (required: base_train loads tokenizer and token_bytes.pt from here)
    echo "=== Training tokenizer ==="
    python -m scripts.tok_train
else
    echo "=== Resuming from step $RESUME_STEP ==="
fi

# Checkpoint directory
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d20"

# Training parameters with aggressive checkpointing
echo "=== Training model (depth=20) with checkpointing ==="
echo "Checkpoint directory: $CHECKPOINT_DIR"

# Create function to handle graceful shutdown
cleanup() {
    echo "Received signal, saving checkpoint and exiting gracefully..."
    # The training script will save on the next iteration
    exit 0
}

# Set up signal handlers for graceful shutdown
trap cleanup TERM INT

# Train depth=20 model with checkpointing every 500 steps (~2 hours)
TRAIN_CMD="python -m scripts.base_train \
    --depth=20 \
    --device-batch-size=8 \
    --run=$WANDB_RUN \
    --save-every=500 \
    --eval-every=100 \
    --core-metric-every=1000 \
    --sample-every=1000"

# Add resume flag if specified
if [[ -n "$RESUME_STEP" ]]; then
    TRAIN_CMD="$TRAIN_CMD --resume-from-step=$RESUME_STEP"
fi

echo "Training command: $TRAIN_CMD"
echo "Checkpoints will be saved every 500 steps"
echo "Starting training at: $(date)"

# Run training
eval $TRAIN_CMD

echo ""
echo "=== Training complete! ==="
echo "Finished at: $(date)"

echo ""
echo "=== Storage Check (after) ==="
du -sh ~ 2>/dev/null | awk '{print "Home total: " $1}'
du -sh ~/.cache/nanochat 2>/dev/null | awk '{print "nanochat cache: " $1}'
du -sh ~/.cache/nanochat/*/ 2>/dev/null || true

echo ""
echo "=== Available checkpoints ==="
if [ -d "$CHECKPOINT_DIR" ]; then
    ls -la "$CHECKPOINT_DIR"/model_*.pt 2>/dev/null | tail -10 || echo "No model checkpoints found"
    echo ""
    echo "To resume from latest checkpoint, find the highest step number and use:"
    echo "sbatch runs/uppmax/train_d20_checkpoint.sh <step_number>"
else
    echo "No checkpoint directory found at $CHECKPOINT_DIR"
fi

echo ""