#!/bin/bash
#SBATCH -N 1
#SBATCH -p ice-gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint="gpu-h100|gpu-h200"
#SBATCH --mem-per-gpu=48G
#SBATCH -t 3:55:00
#SBATCH -J nanochat-stage2d
#SBATCH -o runs/logs/stage2d_%j.out
#SBATCH -e runs/logs/stage2d_%j.err

# Stage 2d

set -e
cd "$HOME/scratch/nanochat"

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/scratch/nanochat"
mkdir -p runs/logs

WANDB_RUN="${WANDB_RUN:-dummy}"
XSA="${XSA:-FALSE}"
XSA_ARG=""
[ "$XSA" = "TRUE" ] && XSA_ARG="--xsa"
CHECKPOINT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d24"
DONE_MARKER="$CHECKPOINT_DIR/.training_complete"

echo "=== Stage 2d: Pretraining (chunk 4 / auto-resume) ==="
echo "Base dir: $NANOCHAT_BASE_DIR"
echo "XSA: $XSA"
echo "Started: $(date)"

if [ -f "$DONE_MARKER" ]; then
    echo "Training already complete (marker: $DONE_MARKER). Nothing to do."
    echo "=== Stage 2d skipped: $(date) ==="
    exit 0
fi

source .venv/bin/activate

LAST_STEP=$(python -c "
import glob, os, sys
files = glob.glob('${CHECKPOINT_DIR}/model_*.pt')
if not files:
    print(0); sys.exit(0)
print(max(int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in files))
")

if [ "$LAST_STEP" -eq 0 ]; then
    echo "No checkpoint found — starting from scratch"
    torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
        --depth=24 \
        --target-param-data-ratio=8 \
        --device-batch-size=16 \
        --save-every=200 \
        $XSA_ARG \
        --run=$WANDB_RUN
else
    echo "Resuming from step $LAST_STEP"
    torchrun --standalone --nproc_per_node=2 -m scripts.base_train -- \
        --depth=24 \
        --target-param-data-ratio=8 \
        --device-batch-size=16 \
        --save-every=200 \
        --resume-from-step=$LAST_STEP \
        $XSA_ARG \
        --run=$WANDB_RUN
fi

mkdir -p "$CHECKPOINT_DIR"
touch "$DONE_MARKER"
echo "=== Stage 2d complete: $(date) ==="
