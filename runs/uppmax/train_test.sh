#!/bin/bash
#SBATCH -A uppmax2025-2-290
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -t 01:00:00
#SBATCH -J nanochat-test
#SBATCH --output=nanochat-%j.out
#SBATCH --error=nanochat-%j.err

# nanochat test run on UPPMAX Pelle (single L40s GPU)
# This is a "hello world" test - depth=8 is very small (~5-10 min)
# Expected storage: ~500MB

set -e

echo "=== nanochat test run on UPPMAX Pelle ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
date

# Storage check
echo ""
echo "=== Storage Check (before) ==="
du -sh ~ 2>/dev/null | awk '{print "Home total: " $1}'
du -sh ~/.cache/nanochat 2>/dev/null | awk '{print "nanochat cache: " $1}' || echo "nanochat cache: 0"
echo ""

# Load modules
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
    WANDB_RUN="nanochat-d8-$SLURM_JOB_ID"
else
    WANDB_RUN="dummy"
    echo "No wandb key found, logging disabled"
fi

# Activate virtual environment
source .venv/bin/activate

# Download minimal dataset (2 shards for quick test, ~200MB)
echo "=== Downloading dataset (2 shards, ~200MB) ==="
python -m nanochat.dataset -n 2

# Train tokenizer on small data
echo "=== Training tokenizer ==="
python -m scripts.tok_train

# Run a quick training test with depth=8 (tiny model, ~5-10 min)
echo "=== Training model (depth=8) ==="
python -m scripts.base_train --depth=8 --device-batch-size=8 --run=$WANDB_RUN

echo ""
echo "=== Storage Check (after) ==="
du -sh ~ 2>/dev/null | awk '{print "Home total: " $1}'
du -sh ~/.cache/nanochat 2>/dev/null | awk '{print "nanochat cache: " $1}'
du -sh ~/.cache/nanochat/*/ 2>/dev/null || true
echo ""

echo "=== Training complete! ==="
date
