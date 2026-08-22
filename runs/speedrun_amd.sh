#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning) on AMD ROCm GPUs.
# It is designed to work across single-GPU or multi-GPU AMD setups (e.g. Radeon AI PRO R9700 / Instinct MI300).

# 1) Example launch (full GPT-2 grade model):
# bash runs/speedrun_amd.sh
# 2) Example launch (small fast model, ~5h on slower GPUs):
# bash runs/speedrun_amd.sh small
# 3) Example launch in a screen session:
# screen -L -Logfile runs/speedrun_amd.log -S speedrun_amd bash runs/speedrun_amd.sh small
# 4) Example launch with custom GPU count and wandb logging:
# NPROC_PER_NODE=2 WANDB_RUN=speedrun_amd bash runs/speedrun_amd.sh small

# Model size: "full" (default GPT-2 grade) or "small" (fast 6-layer model)
MODEL_SIZE="${1:-full}"
if [ "$MODEL_SIZE" != "small" ] && [ "$MODEL_SIZE" != "full" ]; then
    echo "Unknown model size: $MODEL_SIZE. Expected 'small' or 'full' (default)."
    exit 1
fi

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv (AMD / ROCm)

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install repo dependencies with the amd extra
uv sync --extra amd --prerelease=allow
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# Auto-detect valid discrete AMD GPUs (filtering out integrated CPU graphics/APUs)
if [ -z "$HIP_VISIBLE_DEVICES" ]; then
    VALID_GPUS=$(python -c "import torch; valid = [str(i) for i in range(torch.cuda.device_count()) if not any(k in torch.cuda.get_device_name(i).lower() for k in ['processor', 'ryzen', 'apu', 'intel'])]; print(','.join(valid))" 2>/dev/null)
    if [ -n "$VALID_GPUS" ]; then
        export HIP_VISIBLE_DEVICES=$VALID_GPUS
    fi
fi

# Auto-detect number of AMD GPUs available (unless explicitly overridden)
if [ -z "$NPROC_PER_NODE" ]; then
    NPROC_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 1)" 2>/dev/null || echo 1)
fi
echo "Running ($MODEL_SIZE model) on $NPROC_PER_NODE AMD GPU(s)..."

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (recommended):
# 1) Make sure to first log in to wandb: `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars, so 8 shards = ~2B chars (~800MB compressed on disk)
python -m nanochat.dataset -n 8

if [ "$MODEL_SIZE" != "small" ]; then
    # Immediately kick off downloading remaining shards in background while tokenizer trains
    # Approximately 150 shards are needed for GPT-2 capability pretraining, add 20 for padding.
    python -m nanochat.dataset -n 170 &
    DATASET_DOWNLOAD_PID=$!
fi

# Train the tokenizer with vocab size 2**15 = 32768
python -m scripts.tok_train --max-chars=2000000000
# Evaluate the tokenizer (compression ratio report)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

if [ "$MODEL_SIZE" == "small" ]; then
    # Small 6-layer model (adapted from runcpu.sh, fast on low-tier/mid-tier GPUs)
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train \
        --depth=6 \
        --head-dim=64 \
        --window-pattern=L \
        --max-seq-len=512 \
        --device-batch-size=32 \
        --total-batch-size=16384 \
        --eval-every=200 \
        --eval-tokens=524288 \
        --core-metric-every=-1 \
        --sample-every=100 \
        --num-iterations=7500 \
        --run=$WANDB_RUN

    # Evaluate the base model (quick evaluation)
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- \
        --device-batch-size=32 \
        --split-tokens=16384 \
        --max-per-task=16
else
    echo "Waiting for dataset download to complete..."
    wait $DATASET_DOWNLOAD_PID

    # d24 model (target data:params ratio = 8 for fast GPT-2 baseline)
    # Note: --window-pattern=L is set for SDPA performance on AMD ROCm (full attention context across layers)
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train \
        --depth=24 \
        --target-param-data-ratio=8 \
        --device-batch-size=16 \
        --window-pattern=L \
        --run=$WANDB_RUN

    # Evaluate the base model: CORE metric, BPB on train/val, and sample generations
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- --device-batch-size=16
fi

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)

if [ "$MODEL_SIZE" == "small" ]; then
    # SFT for small model (with warmup and lower initial LR for training stability)
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft \
        --eval-every=200 \
        --chatcore-every=1000 \
        --eval-tokens=524288 \
        --num-iterations=4000 \
        --warmup-ratio=0.2 \
        --init-lr-frac=0.00003 \
        --run=$WANDB_RUN
else
    # Full SFT
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
fi

# Run chat evaluation
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# Chat with the model over CLI! (Leave out the -p to chat interactively)
# python -m scripts.chat_cli -p "Why is the sky blue?"
