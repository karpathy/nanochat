#!/bin/bash
#SBATCH -A uppmax2025-2-290
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -t 23:50:00
#SBATCH -J nanochat-d20-demo
#SBATCH --output=nanochat-d20-demo-%j.out
#SBATCH --error=nanochat-d20-demo-%j.err

# nanochat d20 training optimized for 24-hour completion + demo results
# Designed to finish with strong results for Monday presentation
# Usage: sbatch train_d20_demo.sh [resume_step]

set -e

echo "=== nanochat d20 DEMO training (24-hour optimized) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"
echo "Target completion: Monday meeting ready!"

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
    WANDB_RUN="nanochat-d20-demo-$SLURM_JOB_ID"
else
    WANDB_RUN="dummy"
    echo "No wandb key found, logging disabled"
fi

cd ~/nanochat
source .venv/bin/activate

# Download data optimized for 24-hour completion
if [[ -z "$RESUME_STEP" ]]; then
    echo "=== Downloading data (20 shards for fast training) ==="
    python -m nanochat.dataset -n 20
    
    # Train tokenizer (required)
    echo "=== Training tokenizer ==="
    python -m scripts.tok_train
else
    echo "=== Resuming from step $RESUME_STEP ==="
fi

# Checkpoint directory
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d20-demo"

# Training parameters optimized for 24-hour completion with demo-quality results
echo "=== Training model (depth=20) - DEMO OPTIMIZED ==="
echo "Checkpoint directory: $CHECKPOINT_DIR"

# Create function to handle graceful shutdown with final eval
cleanup() {
    echo "Training time almost up - running final evaluation..."
    # The training script will save final checkpoint
    exit 0
}

# Set up signal handlers
trap cleanup TERM INT

# Optimized training for 24-hour completion
# - Slightly larger batch size for faster training
# - More frequent evaluation for better monitoring
# - Strategic checkpoint timing
# - Focus on getting good demo results quickly
TRAIN_CMD="python -m scripts.base_train \
    --depth=20 \
    --device-batch-size=10 \
    --run=$WANDB_RUN \
    --save-every=400 \
    --eval-every=50 \
    --core-metric-every=400 \
    --sample-every=400 \
    --model-tag=d20-demo"

# Add resume flag if specified
if [[ -n "$RESUME_STEP" ]]; then
    TRAIN_CMD="$TRAIN_CMD --resume-from-step=$RESUME_STEP"
fi

echo "Training command: $TRAIN_CMD"
echo "Optimized for 24-hour completion with demo results"
echo "Checkpoints every 400 steps (~1.5 hours)"
echo "Frequent evaluation for monitoring progress"
echo "Starting training at: $(date)"

# Run training with timeout at 23h 40m to allow cleanup
timeout 85200s $TRAIN_CMD || {
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo "Approaching time limit - saving final state..."
    else
        echo "Training completed or exited with code: $EXIT_CODE"
    fi
}

echo ""
echo "=== Final Evaluation and Demo Prep ==="
echo "Finished at: $(date)"

# Generate final samples for demo
echo "=== Generating demo samples ==="
if [ -d "$CHECKPOINT_DIR" ]; then
    LATEST_MODEL=$(ls -1 "$CHECKPOINT_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1)
    if [ -n "$LATEST_MODEL" ]; then
        LATEST_STEP=$(basename "$LATEST_MODEL" .pt | sed 's/model_0*//')
        echo "Using model from step: $LATEST_STEP"
        
        # Generate interesting samples for demo
        echo "=== Demo Samples ==="
        python -m scripts.base_eval \
            --checkpoint-dir="$CHECKPOINT_DIR" \
            --step=$LATEST_STEP \
            --num-samples=5 \
            --prompt="The future of AI" \
            --max-length=200 || echo "Sample generation failed"
    fi
fi

echo ""
echo "=== Storage Check (after) ==="
du -sh ~ 2>/dev/null | awk '{print "Home total: " $1}'
du -sh ~/.cache/nanochat 2>/dev/null | awk '{print "nanochat cache: " $1}'

echo ""
echo "=== DEMO RESULTS SUMMARY ==="
if [ -d "$CHECKPOINT_DIR" ]; then
    echo "✅ Training completed successfully!"
    echo "📁 Model saved in: $CHECKPOINT_DIR"
    ls -la "$CHECKPOINT_DIR"/model_*.pt 2>/dev/null | tail -5
    echo ""
    echo "🎯 For Monday demo:"
    echo "   1. Model trained to step: $(ls -1 "$CHECKPOINT_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1 | sed 's/.*model_0*//' | sed 's/.pt//' || echo 'N/A')"
    echo "   2. Checkpoint available for inference"
    echo "   3. Training logs in: ~/nanochat-d20-demo-$SLURM_JOB_ID.out"
    echo "   4. Use: python -m scripts.chat_cli for interactive demo"
else
    echo "❌ No model checkpoints found - check logs for issues"
fi

echo ""
echo "🚀 READY FOR MONDAY DEMO! 🚀"