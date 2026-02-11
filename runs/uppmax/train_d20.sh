#!/bin/bash
#SBATCH -A uppmax2025-2-290
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -J nanochat-d20
#SBATCH -o /home/%u/nanochat/nanochat-%j.out
#SBATCH -e /home/%u/nanochat/nanochat-%j.err

# nanochat depth=20 training (~12-15 hours on L40s)
# Produces a decent small model capable of basic reasoning

set -e

echo "=== nanochat depth=20 training ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
date

# Load modules (Pelle-specific)
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

cd ~/nanochat
source .venv/bin/activate

# Set up wandb if key exists
if [ -f "$HOME/.wandb_key" ]; then
    export WANDB_API_KEY=$(cat $HOME/.wandb_key)
    echo "wandb API key loaded"
fi

echo "=== Storage Check (before) ==="
echo "Home: $(du -sh ~ 2>/dev/null | cut -f1)"

# Download data (20 shards for depth=20, ~2GB)
echo "=== Downloading data (20 shards) ==="
python -m nanochat.dataset -n 20

# Train depth=20 model
echo "=== Training model (depth=20) ==="
python -m scripts.base_train --depth=20 --device-batch-size=8

echo "=== Storage Check (after) ==="
echo "Home: $(du -sh ~ 2>/dev/null | cut -f1)"
du -sh ~/.cache/nanochat/*/ 2>/dev/null

echo "=== Training complete! ==="
date
