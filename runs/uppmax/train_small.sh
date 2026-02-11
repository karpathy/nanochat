#!/bin/bash
#SBATCH -A uppmax2025-2-290
#SBATCH -p gpu
#SBATCH --gpus=l40s:1
#SBATCH -t 02:00:00
#SBATCH -J nanochat-d12
#SBATCH --output=nanochat-%j.out
#SBATCH --error=nanochat-%j.err

# nanochat small model on UPPMAX Pelle (single L40s GPU)
# depth=12 is a reasonable small model (~30-60 min)

set -e

echo "=== nanochat depth=12 training on UPPMAX Pelle ==="
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
export PYTHONPATH="$PYTHONPATH:$(pwd)"
mkdir -p $NANOCHAT_BASE_DIR

# Activate virtual environment
source .venv/bin/activate

# Download dataset (more shards for better training)
echo "=== Downloading dataset ==="
python -m nanochat.dataset -n 20

# Train tokenizer
echo "=== Training tokenizer ==="
python -m scripts.tok_train

# Evaluate tokenizer
python -m scripts.tok_eval

# Train depth=12 model
echo "=== Training model (depth=12) ==="
python -m scripts.base_train --depth=12 --device-batch-size=8

# Evaluate the model
echo "=== Evaluating model ==="
python -m scripts.base_eval --device-batch-size=8

echo "=== Training complete! ==="
date
