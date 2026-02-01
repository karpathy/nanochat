#!/bin/bash
set -euo pipefail

# D12 launch script for pretraining + eval + SFT + eval + report
#
# Design goals (diffs vs runs/speedrun.sh):
# - Depth: d12 (GPT-1 size)
# - GPUs: 2 (CUDA devices 0,1)
# - Per-device batch size: 128 (to keep total batch size at 524,288 tokens with 2 GPUs)
# - Dataset: 70 shards (moderate footprint for a 110M-ish model)
# - Optimizer tweaks: use the d12 "optimizer sweep best combo" from dev/LOG.md
#   * embedding_lr=0.38, matrix_lr=0.027, weight_decay=0.13
#   * unembedding_lr stays default (0.004)
#   * scalar_lr remains default (0.5) because the sweep did not tie it to the best combo
# - SFT: mirror the same optimizer settings where the SFT script exposes them
#   (chat_sft does NOT expose scalar_lr or x0_beta1, so those cannot be matched)
#
# Notes on batch size math:
# - tokens per rank = device_batch_size * max_seq_len = 128 * 2048 = 262,144
# - world tokens   = tokens per rank * world_size    = 262,144 * 2 = 524,288
# - grad accum     = total_batch_size / world tokens = 1
#
# Notes on data shards:
# - nanochat uses the LAST shard as val split; with 70 shards this means 69 train + 1 val.
# - the dataloader cycles indefinitely over shards, so smaller shard counts repeat data sooner.

export OMP_NUM_THREADS=1
# Store all artifacts in the parent directory by default (checkpoints, data, reports, etc.)
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$(cd .. && pwd)}"
mkdir -p "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# User environment variables (fill in as needed)
# export WANDB_PROJECT="nanochat"
# export WANDB_ENTITY="your-wandb-entity"
# export WANDB_API_KEY="..."
# export WANDB_MODE="online"   # or "offline"
# export HF_TOKEN="..."        # if you need private HF access

# -----------------------------------------------------------------------------
# Python venv setup with uv (local ./.venv)

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

if [ ! -d ".venv" ]; then
    uv venv
    uv sync --extra gpu
    echo "Created .venv and installed dependencies."
    echo "Activate it, login to wandb if needed, then re-run this script."
    exit 0
fi

# Keep dependencies up to date on subsequent runs
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# If WANDB_RUN is unset, default to dummy to disable wandb logging.
WANDB_RUN="${WANDB_RUN:-dummy}"

# -----------------------------------------------------------------------------
# Report reset (writes header with system info and timestamp)
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer + dataset

# Download 70 shards of pretraining data (69 train + 1 val)
python -m nanochat.dataset -n 70

# Train + eval tokenizer (default uses ~2B chars)
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Hardware config

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

# -----------------------------------------------------------------------------
# Hyperparameters (pretrain + SFT)

DEPTH=12
MODEL_TAG="${MODEL_TAG:-d12}"
DEVICE_BATCH_SIZE=128
TOTAL_BATCH_SIZE=524288
MAX_SEQ_LEN=2048
TARGET_PARAM_DATA_RATIO=10.5

# d12 optimizer sweep best combo (dev/LOG.md)
EMBEDDING_LR=0.38
MATRIX_LR=0.027
WEIGHT_DECAY=0.13
UNEMBEDDING_LR=0.004
SCALAR_LR=0.5

# -----------------------------------------------------------------------------
# Base model (pretraining)

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth="$DEPTH" \
    --run="$WANDB_RUN" \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
    --embedding-lr="$EMBEDDING_LR" \
    --unembedding-lr="$UNEMBEDDING_LR" \
    --matrix-lr="$MATRIX_LR" \
    --weight-decay="$WEIGHT_DECAY" \
    --scalar-lr="$SCALAR_LR"

# Evaluate the base model
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE"

# -----------------------------------------------------------------------------
# SFT (teach conversation, tools, etc.)

# Download synthetic identity conversations
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# SFT run: mirror pretrain optimizer settings where supported
# (chat_sft does NOT expose scalar_lr/x0_beta1, so those cannot be matched here.)
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
    --run="$WANDB_RUN" \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --total-batch-size="$TOTAL_BATCH_SIZE" \
    --max-seq-len="$MAX_SEQ_LEN" \
    --embedding-lr="$EMBEDDING_LR" \
    --unembedding-lr="$UNEMBEDDING_LR" \
    --matrix-lr="$MATRIX_LR" \
    --weight-decay="$WEIGHT_DECAY"

# Evaluate the SFT model
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# Generate the full report
python -m nanochat.report generate
