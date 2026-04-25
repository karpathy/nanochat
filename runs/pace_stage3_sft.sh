#!/bin/bash
#SBATCH -N 1
#SBATCH -p ice-gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu-h100|gpu-h200"
#SBATCH --mem-per-gpu=48G
#SBATCH -t 3:55:00
#SBATCH -J nanochat-stage3-sft
#SBATCH -o runs/logs/stage3_%j.out
#SBATCH -e runs/logs/stage3_%j.err

# Stage 3

set -e
cd "$HOME/scratch/nanochat"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/scratch/nanochat"
mkdir -p runs/logs

WANDB_RUN="${WANDB_RUN:-dummy}"

echo "=== Stage 3: Eval + SFT ==="
echo "Base dir: $NANOCHAT_BASE_DIR"
echo "WANDB_RUN: $WANDB_RUN"
echo "Started: $(date)"

CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d24"
DONE_MARKER="$CHECKPOINT_DIR/.training_complete"
if [ ! -f "$DONE_MARKER" ]; then
    echo "ERROR: pretraining did not finish — missing $DONE_MARKER"
    echo "Re-run pretrain chunks 2a–2d until the marker is created before running stage 3."
    exit 1
fi

source .venv/bin/activate

torchrun --standalone --nproc_per_node=2 -m scripts.base_eval -- \
    --device-batch-size=16

curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=2 -m scripts.chat_sft -- \
    --device-batch-size=16 \
    --run=$WANDB_RUN

torchrun --standalone --nproc_per_node=2 -m scripts.chat_eval -- -i sft

python -m nanochat.report generate

echo "=== Stage 3 complete: $(date) ==="

