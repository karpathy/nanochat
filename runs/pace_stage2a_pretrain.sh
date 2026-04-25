#!/bin/bash
#SBATCH -N 1
#SBATCH -p ice-gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu-h100|gpu-h200"
#SBATCH --mem-per-gpu=48G
#SBATCH -t 3:55:00
#SBATCH -J nanochat-stage2a
#SBATCH -o runs/logs/stage2a_%j.out
#SBATCH -e runs/logs/stage2a_%j.err

# Stage 2a

set -e
cd "$HOME/scratch/nanochat"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/scratch/nanochat"
mkdir -p runs/logs

WANDB_RUN="${WANDB_RUN:-dummy}"
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d24"
DONE_MARKER="$CHECKPOINT_DIR/.training_complete"

echo "=== Stage 2a: Pretraining (chunk 1) ==="
echo "Base dir: $NANOCHAT_BASE_DIR"
echo "WANDB_RUN: $WANDB_RUN"
echo "Started: $(date)"

source .venv/bin/activate

torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size=16 \
    --save-every=200 \
    --run=$WANDB_RUN

mkdir -p "$CHECKPOINT_DIR"
touch "$DONE_MARKER"
echo "=== Stage 2a complete: $(date) ==="
