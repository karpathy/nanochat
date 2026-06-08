#!/bin/bash
#SBATCH --partition=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:7
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --signal=B:USR1@900
#SBATCH -o nanochat4b-%j.out
#SBATCH -e nanochat4b-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/u201/l39gu/projects/nanochat}"

# Keep the run directory stable across submissions so AUTO_RESUME can pick up
# the newest checkpoint after preemption or a time-limit stop.
export RUN_ROOT="${RUN_ROOT:-/u201/l39gu/nanoknow-climbmix/nanochat-runs}"
export RUN_NAME="${RUN_NAME:-d36_4b_climbmix}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$RUN_ROOT/$RUN_NAME}"
export CLIMBMIX_DATA_DIR="${CLIMBMIX_DATA_DIR:-/u201/l39gu/projects/climbmix-400b-shuffle}"
mkdir -p "$NANOCHAT_BASE_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ';')"
    echo "Allocated GPUs: $GPU_NAMES"
    if [ "${ALLOW_NON_H200_OR_BLACKWELL:-0}" != "1" ] &&
       [[ "$GPU_NAMES" != *"H200"* ]] &&
       [[ "$GPU_NAMES" != *"Blackwell"* ]] &&
       [[ "$GPU_NAMES" != *"B200"* ]] &&
       [[ "$GPU_NAMES" != *"B300"* ]]; then
        echo "Refusing non-H200/non-Blackwell allocation. Set ALLOW_NON_H200_OR_BLACKWELL=1 to override."
        exit 2
    fi
    if [[ "$GPU_NAMES" == *"Blackwell"* ]] && [[ "${EXTRA_BASE_TRAIN_ARGS:-}" != *"--window-pattern"* ]]; then
        export EXTRA_BASE_TRAIN_ARGS="${EXTRA_BASE_TRAIN_ARGS:-} --window-pattern=L"
        echo "Blackwell allocation detected; using --window-pattern=L for SDPA compatibility."
    fi
fi

export MODEL_TAG="${MODEL_TAG:-d36_4b_climbmix}"
export MODEL_DEPTH="${MODEL_DEPTH:-36}"
export TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-12}"
export DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
export DATASET_SHARDS="${DATASET_SHARDS:-1000}"
export DATASET_WORKERS="${DATASET_WORKERS:-8}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-7}}"
export SAVE_EVERY="${SAVE_EVERY:-250}"
export KEEP_LAST_CHECKPOINTS="${KEEP_LAST_CHECKPOINTS:-4}"
export AUTO_RESUME="${AUTO_RESUME:-1}"
export WANDB_RUN="${WANDB_RUN:-dummy}"

bash runs/train_4b.sh
