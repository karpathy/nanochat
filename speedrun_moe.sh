#!/bin/bash

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# bash speedrun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat-moe
export USER=""
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$USER/.cache/nanochat-moe"
export NANOCHAT_DATA_DIR="$USER/.cache/nanochat-moe/base_data"
mkdir -p $NANOCHAT_BASE_DIR
mkdir -p $NANOCHAT_DATA_DIR



# Use tokenizer from nanochat (not nanochat-moe)
# Create a symlink to nanochat's tokenizer directory if it doesn't exist
NANOCHAT_TOKENIZER_DIR="$USER/.cache/nanochat-moe/tokenizer"
MOE_TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
if [ -d "$NANOCHAT_TOKENIZER_DIR" ] && [ ! -e "$MOE_TOKENIZER_DIR" ]; then
    echo "Creating symlink to nanochat tokenizer: $MOE_TOKENIZER_DIR -> $NANOCHAT_TOKENIZER_DIR"
    ln -s "$NANOCHAT_TOKENIZER_DIR" "$MOE_TOKENIZER_DIR"
elif [ ! -d "$NANOCHAT_TOKENIZER_DIR" ]; then
    echo "Warning: nanochat tokenizer directory not found at $NANOCHAT_TOKENIZER_DIR"
    echo "You may need to train the tokenizer first using nanochat's tok_train.py"
fi

# # -----------------------------------------------------------------------------
# # China mirror configuration (环境镜像配置)

# # Configure pip mirror
# mkdir -p ~/.pip
# cat > ~/.pip/pip.conf << 'EOF'
# [global]
# index-url = https://pypi.tuna.tsinghua.edu.cn/simple
# trusted-host = pypi.tuna.tsinghua.edu.cn
# timeout = 1000

# [install]
# trusted-host = pypi.tuna.tsinghua.edu.cn
# EOF

# # Configure Rust mirror
# export RUSTUP_DIST_SERVER=https://rsproxy.cn
# export RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
# SHELL_RC="$HOME/.bashrc"
# if [[ "$OSTYPE" == "darwin"* ]]; then
#     SHELL_RC="$HOME/.zshrc"
# fi
# if ! grep -q "RUSTUP_DIST_SERVER" "$SHELL_RC" 2>/dev/null; then
#     cat >> "$SHELL_RC" << 'EOF'

# # Rust 镜像配置
# export RUSTUP_DIST_SERVER=https://rsproxy.cn
# export RUSTUP_UPDATE_ROOT=https://rsproxy.cn/rustup
# EOF
# fi

# # Configure Cargo mirror
# mkdir -p ~/.cargo
# cat > ~/.cargo/config << 'EOF'
# [source.crates-io]
# replace-with = 'rsproxy-sparse'

# [source.rsproxy-sparse]
# registry = "sparse+https://rsproxy.cn/index/"

# [net]
# git-fetch-with-cli = true
# EOF

# # Configure HuggingFace mirror
# if ! grep -q "HF_ENDPOINT" "$SHELL_RC" 2>/dev/null; then
#     echo 'export HF_ENDPOINT=https://hf-mirror.com' >> "$SHELL_RC"
# fi
# export HF_ENDPOINT=https://hf-mirror.com

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
if ! command -v uv &> /dev/null; then
    pip3 install uv -i https://pypi.tuna.tsinghua.edu.cn/simple
fi
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies with China mirror
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source "${USER}/nanochat/.venv/bin/activate"

# # -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
WANDB_RUN=moe
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# # During the course of the run, we will be writing markdown reports to the report/
# # directory in the base dir. This command clears it out and writes a header section
# # with a bunch of system info and a timestamp that marks the start of the run.
# python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo (if not already installed)
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://rsproxy.cn/rustup-init.sh | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# Build the rustbpe Tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download the first ~2B characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 240 is the right number here
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
python -m scripts.tok_train --max_chars=2000000000
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)

# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. At ~100MB/shard, this downloads ~24GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
# echo "Waiting for dataset download to complete..."
# wait $DATASET_DOWNLOAD_PID


MODEL_DIM=${MODEL_DIM:-384}
GLOBAL_BS=${GLOBAL_BS:-480}
MIN_LR=${MIN_LR:-6e-5}
LEARNING_RATE=${LEARNING_RATE:-6e-4}
DEPTH=${DEPTH:-${N_LAYER:-6}}
MODEL_TAG=${MODEL_TAG:-d${DEPTH}_min_lr${MIN_LR}_max_lr${LEARNING_RATE}}
# # Number of processes/GPUs to use
# NPROC_PER_NODE=${NPROC_PER_NODE:-8}
# Number of processes/GPUs to use
# Auto-detect number of GPUs: prefer CUDA_VISIBLE_DEVICES, then nvidia-smi, then python torch fallback
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # Count entries in CUDA_VISIBLE_DEVICES (comma-separated list)
    NPROC_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
else
    if command -v nvidia-smi &>/dev/null; then
        NPROC_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    else
        if command -v python3 &>/dev/null; then
            NPROC_PER_NODE=$(python3 - <<'PY'
import sys
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)
        else
            NPROC_PER_NODE=1
        fi
    fi
fi
# Ensure at least 1
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
if [ "$NPROC_PER_NODE" -lt 1 ]; then
    NPROC_PER_NODE=1
fi
# Master port for distributed training (default: 29500)
# Set this to avoid port conflicts when running multiple torchrun tasks simultaneously
# Example: MASTER_PORT=29501 bash speedrun.sh
MASTER_PORT=${MASTER_PORT:-29501}
LOG_TAG=${LOG_TAG:-$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-$NANOCHAT_BASE_DIR/log/${MODEL_TAG}_${LOG_TAG}.log}
# # # pretrain the d20 model
MASTER_PORT=$MASTER_PORT torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train >> "$LOG_FILE" 2>&1

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_sft_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
# curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# MID_LR=${MID_LR:-3e-4}
# LOAD_FROM_MODEL_TAG=${LOAD_FROM_MODEL_TAG:-d6_min_lr0.0002_max_lr0.002}
# WANDB_RUN=moe_mid_${MID_LR}_${LOAD_FROM_MODEL_TAG}

# # run midtraining and eval the model
# MASTER_PORT=$MASTER_PORT torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN \
#     --model_tag=$LOAD_FROM_MODEL_TAG --learning_rate=$MID_LR \
#     --device_batch_size=8 --max_seq_len=1024 --total_batch_size=524288 
# MASTER_PORT=$MASTER_PORT torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN \
#     --model_tag=d12_e16_lr2e-4_double_dmodel \
#     --learning_rate=$MID_LR --num_epochs=1 \
#     --device_batch_size=8 --max_seq_len=1024 --total_batch_size=524288 
    # --disable_aux_loss=True --disable_router_z_loss=True \
# MASTER_PORT=$MASTER_PORT torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts_moe.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)
SFT_LR=${SFT_LR:-9e-6}
LOAD_FROM_MID_MODEL_TAG=${LOAD_FROM_MID_MODEL_TAG:-d6_lr0.0003_modeld6_min_lr0.0002_max_lr0.002}
WANDB_RUN=moe_sft_${SFT_LR}_${LOAD_FROM_MID_MODEL_TAG}
# train sft and re-eval right away (should see a small bump)

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN --learning_rate=$SFT_LR --init_lr_frac=1.0 \
    --model_tag=$LOAD_FROM_MID_MODEL_TAG

# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN --learning_rate=3e-4 --init_lr_frac=1.0 --num_epochs=4
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web
# -----------------------------------------------------------------------------
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)
RL_LR=${RL_LR:-9e-6}
LOAD_FROM_SFT_MODEL_TAG=${LOAD_FROM_SFT_MODEL_TAG:-d6_lr5e-05_init1.0_modeld6_lr0.0003_modeld6_min_lr0.0002_max_lr0.002}
WANDB_RUN=moe_rl_${RL_LR}_${LOAD_FROM_SFT_MODEL_TAG}

# run reinforcement learning
MASTER_PORT=$MASTER_PORT torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN \
    --model_tag=$LOAD_FROM_SFT_MODEL_TAG 

# eval the RL model only on GSM8K
# MASTER_PORT=$MASTER_PORT torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate