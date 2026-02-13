#!/bin/bash
#SBATCH -A uppmax2025-2-290
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -t 48:00:00
#SBATCH -J nanochat-d24-ckpt
#SBATCH --output=nanochat-d24-ckpt-%j.out
#SBATCH --error=nanochat-d24-ckpt-%j.err

# nanochat depth=24 training with checkpointing (~24-30 hours on L40s)
# Saves checkpoints every 1.5 hours for longer training runs
# Usage: sbatch train_d24_checkpoint.sh [resume_step]

set -e

echo "=== nanochat depth=24 training with checkpointing ==="
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
    WANDB_RUN="nanochat-d24-ckpt-$SLURM_JOB_ID"
else
    WANDB_RUN="dummy"
    echo "No wandb key found, logging disabled"
fi

cd ~/nanochat
source .venv/bin/activate

# Download data (24 shards for depth=24, ~2.4GB) - only if not resuming
if [[ -z "$RESUME_STEP" ]]; then
    echo "=== Downloading data (24 shards) ==="
    python -m nanochat.dataset -n 24
    
    # Train tokenizer (required: base_train loads tokenizer and token_bytes.pt from here)
    echo "=== Training tokenizer ==="
    python -m scripts.tok_train
else
    echo "=== Resuming from step $RESUME_STEP ==="
fi

# Checkpoint directory
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d24"

# Training parameters with frequent checkpointing for long runs
echo "=== Training model (depth=24) with checkpointing ==="
echo "Checkpoint directory: $CHECKPOINT_DIR"

# Create function to handle graceful shutdown
cleanup() {
    echo "Received signal, saving checkpoint and exiting gracefully..."
    # The training script will save on the next iteration
    exit 0
}

# Set up signal handlers for graceful shutdown
trap cleanup TERM INT

# Train depth=24 model with more frequent checkpointing (every 400 steps ~1.5 hours)
# Reduced batch size for larger model to fit in memory
TRAIN_CMD="python -m scripts.base_train \
    --depth=24 \
    --device-batch-size=6 \
    --run=$WANDB_RUN \
    --save-every=400 \
    --eval-every=100 \
    --core-metric-every=800 \
    --sample-every=800"

# Add resume flag if specified
if [[ -n "$RESUME_STEP" ]]; then
    TRAIN_CMD="$TRAIN_CMD --resume-from-step=$RESUME_STEP"
fi

echo "Training command: $TRAIN_CMD"
echo "Checkpoints will be saved every 400 steps (~1.5 hours)"
echo "Starting training at: $(date)"

# Run training with timeout handling
timeout 172800s $TRAIN_CMD || {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo "Training reached time limit, but this is expected for long runs"
        echo "Use resume_latest.sh to continue training"
    else
        echo "Training exited with code: $EXIT_CODE"
    fi
}

echo ""
echo "=== Training session ended ==="
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
    echo "To resume from latest checkpoint, run:"
    echo "./runs/uppmax/resume_latest.sh 24"
else
    echo "No checkpoint directory found at $CHECKPOINT_DIR"
fi

echo ""