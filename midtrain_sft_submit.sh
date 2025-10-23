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


set -x  # Enable debug output

export DATA_NAME=nemotron # nemotron # smoltalk
export BASE_NAME=climbmix_8_2_d20_1node_matrixlr0.02_2314630 #climbmix_9_1_d20_1node_matrixlr0.02_2308728 #climbmix_8_2_d20_1node_matrixlr0.02_2309730 #climbmix_5_5_d20_1node_matrixlr0.02_2309731 #climbmix_2_8_d20_1node_matrixlr0.02_2309732 #climbmix_1_9_d20_1node_matrixlr0.02_2309733 #climbmix_d20_1node_matrixlr0.02_2298334 # fineweb_d20_1node # climbmix_d20_1node_matrixlr0.02_2298334 # nemotron-cc-hq_d20_1node_matrixlr0.02_2298371 # smollm_d20_1node_matrixlr0.02_2298373

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


# 1️⃣ 创建或重建 venv（--clear 会先清空旧内容）
[ -d ".venv" ] || uv venv "$HOME/nanochat_cache/.venv" # --clear

# 2️⃣ 激活虚拟环境
source "$HOME/nanochat_cache/.venv/bin/activate"

# 3️⃣ 安装依赖（uv 会自动识别项目 pyproject.toml）
cd /lustre/fs1/portfolios/nvr/projects/nvr_lpr_llm/users/sdiao/nanochat
uv sync --active

export WANDB_API_KEY="ec7a9c0701d404122e4fc5c7c7518ed17f5b03ca"
export WANDB_RUN=data_${DATA_NAME}_base_${BASE_NAME}_${SLURM_JOB_ID}

# python -m nanochat.report reset --exp_name=$WANDB_RUN

# -----------------------------------------------------------------------------
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# run midtraining and eval the model (multi-node)
# mid_train loads from base_checkpoints/$WANDB_RUN and saves to mid_checkpoints/$WANDB_RUN
srun --ntasks=$NNODES --ntasks-per-node=1 bash --noprofile --norc -lc 'source $HOME/nanochat_cache/.venv/bin/activate; torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --rdzv_endpoint=$RDZV_ENDPOINT --rdzv_id=$SLURM_JOB_ID --node_rank=$SLURM_NODEID -m scripts.mid_train -- --run=$WANDB_RUN --model_tag=$BASE_NAME --dataset_choice=$DATA_NAME'
srun --ntasks=$NNODES --ntasks-per-node=1 bash --noprofile --norc -lc 'source $HOME/nanochat_cache/.venv/bin/activate; torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --rdzv_endpoint=$RDZV_ENDPOINT --rdzv_id=$SLURM_JOB_ID --node_rank=$SLURM_NODEID -m scripts.chat_eval -- -i mid --model-tag=$WANDB_RUN'

# -----------------------------------------------------------------------------
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# train sft and re-eval right away (should see a small bump) (multi-node)
# chat_sft loads from mid_checkpoints/$WANDB_RUN and saves to chatsft_checkpoints/$WANDB_RUN
srun --ntasks=$NNODES --ntasks-per-node=1 bash --noprofile --norc -lc 'source $HOME/nanochat_cache/.venv/bin/activate; torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --rdzv_endpoint=$RDZV_ENDPOINT --rdzv_id=$SLURM_JOB_ID --node_rank=$SLURM_NODEID -m scripts.chat_sft -- --run=$WANDB_RUN --model_tag=$WANDB_RUN --dataset_choice=$DATA_NAME'
srun --ntasks=$NNODES --ntasks-per-node=1 bash --noprofile --norc -lc 'source $HOME/nanochat_cache/.venv/bin/activate; torchrun --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --rdzv_endpoint=$RDZV_ENDPOINT --rdzv_id=$SLURM_JOB_ID --node_rank=$SLURM_NODEID -m scripts.chat_eval -- -i sft --model-tag=$WANDB_RUN'

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