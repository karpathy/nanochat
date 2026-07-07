#!/bin/bash

# Multi-node training script for distributed training across multiple servers.
# Usage example for 2 nodes:
# Node 0: MASTER_ADDR=10.0.0.1 NODE_RANK=0 NNODES=2 bash runs/multinode.sh
# Node 1: MASTER_ADDR=10.0.0.1 NODE_RANK=1 NNODES=2 bash runs/multinode.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# NCCL Configuration (Critical for multi-node)
# Force NCCL to use the correct network interface
export NCCL_SOCKET_IFNAME=bond0
# Optional: Enable debug logging to diagnose connection issues
# export NCCL_DEBUG=INFO

# -----------------------------------------------------------------------------
# Multi-node Configuration
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-9321}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

# Function to handle kill signals
cleanup() {
    echo "Stopping script... Killing child processes."
    # Kill the background dataset download if it exists
    if [ -n "$DATASET_DOWNLOAD_PID" ]; then
        kill $DATASET_DOWNLOAD_PID 2>/dev/null
    fi
    # Kill torchrun and other python processes started by this shell
    pkill -P $$ 
    exit 1
}
trap cleanup SIGINT SIGTERM

echo "Starting node $NODE_RANK of $NNODES connected to $MASTER_ADDR:$MASTER_PORT using $GPUS_PER_NODE GPUs."

# -----------------------------------------------------------------------------
# Setup
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Data Preparation (Runs on all nodes to ensure local data availability)
# If using a shared filesystem, you might want to wrap this in: if [ "$NODE_RANK" == "0" ]; then ... fi

if [ "$NODE_RANK" == "0" ]; then
    python -m nanochat.report reset
fi

# Download initial data
python -m nanochat.dataset -n 8

# Download rest in background
echo "[$(date)] Starting background dataset download..."
python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!
echo "[$(date)] Dataset download PID: $DATASET_DOWNLOAD_PID"

# Train tokenizer (might be redundant on workers but ensures consistency)
python -m scripts.tok_train
python -m scripts.tok_eval

echo "[$(date)] Checking download status before waiting..."
if kill -0 $DATASET_DOWNLOAD_PID 2>/dev/null; then
    echo "Process $DATASET_DOWNLOAD_PID is still active."
    echo "Parquet files found so far: $(ls $NANOCHAT_BASE_DIR/base_data/*.parquet 2>/dev/null | wc -l)"
else
    echo "Process $DATASET_DOWNLOAD_PID has already finished."
fi

echo "Waiting for dataset download..."
wait $DATASET_DOWNLOAD_PID
echo "[$(date)] Dataset download completed/verified."

# -----------------------------------------------------------------------------
# Distributed Training
# Using pre-defined distributed args instead of --standalone

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m scripts.base_train -- \
    --depth=26 \
    --target-param-data-ratio=8.5 \
    --device-batch-size=16 \
    --fp8 \
    --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# Evaluation
# Run eval on all nodes (distributed eval) or just master depending on implementation.
# Typically eval is lightweight enough for just master or distributed parallel.
# Assuming distributed eval support in base_eval:

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m scripts.base_eval -- \
    --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT
# SFT also benefits from distributed training

curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m scripts.chat_sft -- \
    --device-batch-size=16 \
    --run=$WANDB_RUN

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# Report (Only Master)
if [ "$NODE_RANK" == "0" ]; then
    python -m nanochat.report generate
fi
