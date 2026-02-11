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

set -e

echo "=== nanochat test run on UPPMAX Pelle ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
date

# Load modules
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

# Set up environment
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Activate virtual environment
source .venv/bin/activate

# Download minimal dataset (2 shards for quick test)
echo "=== Downloading dataset ==="
python -m nanochat.dataset -n 2

# Train tokenizer on small data
echo "=== Training tokenizer ==="
python -m scripts.tok_train

# Run a quick training test with depth=8 (tiny model, ~5-10 min)
# For a more serious run, increase depth (12, 16, 20, etc.)
echo "=== Training model (depth=8) ==="
python -m scripts.base_train --depth=8 --device-batch-size=8

echo "=== Training complete! ==="
date
