#!/bin/bash
#SBATCH -A uppmax2025-2-290
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -t 24:00:00
#SBATCH -J nanochat-d20
#SBATCH --output=nanochat-%j.out
#SBATCH --error=nanochat-%j.err

# nanochat depth=20 training (~12-15 hours on L40s)
# Produces a decent small model capable of basic reasoning

set -e

echo "=== nanochat depth=20 training ==="
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

# Load modules (Pelle-specific)
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

# Set up environment (same as train_test.sh)
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTHONPATH="$PYTHONPATH:$(pwd)"
mkdir -p $NANOCHAT_BASE_DIR

# Load wandb API key if available
if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat $HOME/.wandb_key)
    echo "wandb API key loaded"
    WANDB_RUN="nanochat-d20-$SLURM_JOB_ID"
else
    WANDB_RUN="dummy"
    echo "No wandb key found, logging disabled"
fi

cd ~/nanochat
source .venv/bin/activate

# Download data (20 shards for depth=20, ~2GB)
echo "=== Downloading data (20 shards) ==="
python -m nanochat.dataset -n 20

# Train tokenizer (required: base_train loads tokenizer and token_bytes.pt from here)
echo "=== Training tokenizer ==="
python -m scripts.tok_train

# Train depth=20 model
echo "=== Training model (depth=20) ==="
python -m scripts.base_train --depth=20 --device-batch-size=8 --run=$WANDB_RUN

echo ""
echo "=== Storage Check (after) ==="
du -sh ~ 2>/dev/null | awk '{print "Home total: " $1}'
du -sh ~/.cache/nanochat 2>/dev/null | awk '{print "nanochat cache: " $1}'
du -sh ~/.cache/nanochat/*/ 2>/dev/null || true
echo ""

echo "=== Training complete! ==="
date
