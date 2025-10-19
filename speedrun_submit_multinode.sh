#!/bin/bash
#SBATCH --account nvr_lpr_llm
#SBATCH --partition interactive,batch_short,batch_block1,backfill
#SBATCH --job-name=nanochat_multinode_d20    
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/nanochat_1node_d20-%j.out
#SBATCH --mem=0
#SBATCH --exclusive

# This script is the "Best ChatGPT clone that $100 can buy",
# It is designed to run in ~4 hours on 8XH100 node at $3/GPU/hour.

# 1) Example launch (simplest):
# bash speedrun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

set -x  # Enable debug output

DATA_NAME=climbmix
export DATA_DIR=/lustre/fsw/portfolios/nvr/users/sdiao/nanochat/data/$DATA_NAME
export MATRIX_LR=0.02

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/nanochat_cache"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Multi-node defaults from Slurm environment

export GPUS_PER_NODE=${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}
export NNODES=${NNODES:-${SLURM_NNODES:-2}}
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=${MASTER_PORT:-29500}
export RDZV_ENDPOINT=$MASTER_ADDR:$MASTER_PORT
export NCCL_ASYNC_ERROR_HANDLING=1

# -----------------------------------------------------------------------------
# Python venv setup with uv

# # install uv (if not already installed)
# command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# # create a .venv local virtual environment (if it doesn't exist)
# [ -d "$HOME/nanochat_cache/.venv" ] || uv venv
# # install the repo dependencies
# uv sync
# # activate venv so that `python` uses the project's venv instead of system python
# source $HOME/nanochat_cache/.venv/bin/activate

# 1️⃣ 创建或重建 venv（--clear 会先清空旧内容）
uv venv "$HOME/nanochat_cache/.venv" --clear

# 2️⃣ 激活虚拟环境
source "$HOME/nanochat_cache/.venv/bin/activate"

# 3️⃣ 安装依赖（uv 会自动识别项目 pyproject.toml）
cd /lustre/fs1/portfolios/nvr/projects/nvr_lpr_llm/users/sdiao/nanochat
uv sync --active


# -----------------------------------------------------------------------------
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
# if [ -z "$WANDB_RUN" ]; then
#     # by default use "dummy" : it's handled as a special case, skips logging to wandb
#     WANDB_RUN=dummy
# fi
export WANDB_API_KEY="ec7a9c0701d404122e4fc5c7c7518ed17f5b03ca"
export WANDB_RUN=${DATA_NAME}_d20_1node_matrixlr${MATRIX_LR}_${SLURM_JOB_ID}

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset --exp_name=$WANDB_RUN

# -----------------------------------------------------------------------------
# Tokenizer

# Install Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

echo "VIRTUAL_ENV: $VIRTUAL_ENV"
echo "CONDA_PREFIX: $CONDA_PREFIX"
unset CONDA_PREFIX

# Build the rustbpe Tokenizer
# uv run 
maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download the first ~2B characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
python -m nanochat.dataset -n 8 --data_dir=$DATA_DIR
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 240 is the right number here
python -m nanochat.dataset -n 240 --data_dir=$DATA_DIR &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
# Use unique tokenizer name based on dataset
TOKENIZER_NAME="tokenizer_${DATA_NAME}"
python -m scripts.tok_train --max_chars=2000000000 --data_dir=$DATA_DIR --tokenizer_name=$TOKENIZER_NAME
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval --tokenizer_name=$TOKENIZER_NAME

# -----------------------------------------------------------------------------
# Base model (pretraining)

# Download the eval_bundle from s3 to evaluate CORE metric during training (~162MB)
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. At ~100MB/shard, this downloads ~24GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Warm up venv on all nodes (ensures env is available everywhere)
srun --ntasks=$NNODES --ntasks-per-node=1 bash --noprofile --norc -lc 'source $HOME/nanochat_cache/.venv/bin/activate; python -c "import torch; print(torch.cuda.device_count())"'

# pretrain the d20 model (multi-node)
srun --ntasks=$NNODES --ntasks-per-node=1 bash --noprofile --norc -lc 'source $HOME/nanochat_cache/.venv/bin/activate; torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --rdzv_endpoint=$RDZV_ENDPOINT --rdzv_id=$SLURM_JOB_ID --node_rank=$SLURM_NODEID -m scripts.base_train -- --depth=20 --run=$WANDB_RUN --data_dir=$DATA_DIR --matrix_lr=$MATRIX_LR'
# evaluate the model on a larger chunk of train/val data and draw some samples (multi-node)
srun --ntasks=$NNODES --ntasks-per-node=1 bash --noprofile --norc -lc 'source $HOME/nanochat_cache/.venv/bin/activate; torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --rdzv_endpoint=$RDZV_ENDPOINT --rdzv_id=$SLURM_JOB_ID --node_rank=$SLURM_NODEID -m scripts.base_loss --data_dir=$DATA_DIR'
# evaluate the model on CORE tasks (multi-node)
srun --ntasks=$NNODES --ntasks-per-node=1 bash --noprofile --norc -lc 'source $HOME/nanochat_cache/.venv/bin/activate; torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --rdzv_endpoint=$RDZV_ENDPOINT --rdzv_id=$SLURM_JOB_ID --node_rank=$SLURM_NODEID -m scripts.base_eval'

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# run midtraining and eval the model (multi-node)
srun --ntasks=$NNODES --ntasks-per-node=1 bash --noprofile --norc -lc 'source $HOME/nanochat_cache/.venv/bin/activate; torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --rdzv_endpoint=$RDZV_ENDPOINT --rdzv_id=$SLURM_JOB_ID --node_rank=$SLURM_NODEID -m scripts.mid_train -- --run=$WANDB_RUN'
srun --ntasks=$NNODES --ntasks-per-node=1 bash --noprofile --norc -lc 'source $HOME/nanochat_cache/.venv/bin/activate; torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --rdzv_endpoint=$RDZV_ENDPOINT --rdzv_id=$SLURM_JOB_ID --node_rank=$SLURM_NODEID -m scripts.chat_eval -- -i mid'

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# train sft and re-eval right away (should see a small bump) (multi-node)
srun --ntasks=$NNODES --ntasks-per-node=1 bash --noprofile --norc -lc 'source $HOME/nanochat_cache/.venv/bin/activate; torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --rdzv_endpoint=$RDZV_ENDPOINT --rdzv_id=$SLURM_JOB_ID --node_rank=$SLURM_NODEID -m scripts.chat_sft -- --run=$WANDB_RUN'
srun --ntasks=$NNODES --ntasks-per-node=1 bash --noprofile --norc -lc 'source $HOME/nanochat_cache/.venv/bin/activate; torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --rdzv_endpoint=$RDZV_ENDPOINT --rdzv_id=$SLURM_JOB_ID --node_rank=$SLURM_NODEID -m scripts.chat_eval -- -i sft'

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)

# run reinforcement learning
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=$WANDB_RUN
# eval the RL model only on GSM8K
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate --exp_name=$WANDB_RUN
