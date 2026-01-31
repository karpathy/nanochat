#!/bin/bash

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.
# This version auto-detects the PyTorch flavour (CUDA/ROCm/CPU) and number of GPUs.

# 1) Example launch (simplest):
# bash autorun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile autorun.log -S autorun bash autorun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=autorun screen -L -Logfile autorun.log -S autorun bash autorun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Auto-detect PyTorch flavour
# Source the GPU configuration utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/gpu_config_utils.sh"

TORCH_FLAVOUR=$(detect_torch_flavour)

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH if it was just installed
    export PATH="$HOME/.local/bin:$PATH"
fi
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# install the repo dependencies with the detected PyTorch flavour
echo "Installing dependencies for PyTorch flavour: $TORCH_FLAVOUR"
uv sync --extra $TORCH_FLAVOUR

# Verify the backend after installation
verify_backend_after_install

NUM_GPUS=$(detect_num_gpus_pytorch)
# Ensure at least 1 process (fallback to 1 if no GPUs detected)
if [ "$NUM_GPUS" -eq 0 ]; then
    NUM_GPUS=1
fi
export NPROC_PER_NODE="${NPROC_PER_NODE:-$NUM_GPUS}"
echo "Using PyTorch-detected GPU count: $NUM_GPUS, NPROC_PER_NODE=$NPROC_PER_NODE"

# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash autorun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 370 is the right number here
python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
python -m scripts.tok_train --max-chars=2000000000 --vocab-size=65536
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. Also, the new DataLoader wastes about 35% of tokens to cropping
# so 240 / (1 - 0.35) = 370 shards are needed.
# At ~100MB/shard, this downloads ~37GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# pretrain the d20 model
# Set optimal environment variables based on detected PyTorch flavour
set_backend_env_vars "$TORCH_FLAVOUR"

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=20 --target-param-data-ratio=20 --run=$WANDB_RUN
# evaluate the model on a larger chunk of train/val data and draw some samples
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
# evaluate the model on CORE tasks
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run midtraining and eval the model
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# train sft and re-eval right away (should see a small bump)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)

# run reinforcement learning
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN
# eval the RL model only on GSM8K
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
