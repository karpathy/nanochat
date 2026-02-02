#!/bin/bash

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

( cat ./nanochat/gpt.py; cat ./nanochat/optim.py; cat ./nanochat/dataloader.py; cat ./scripts/base_train.py; echo -e "\n\n===== TRAINING OUTPUT =====\n\n"; OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=24 \
    --run=d24-feb01 \
    --model-tag=d24_feb01 \
    --device-batch-size=16 \
    --sample-every=-1 \
    --save-every=-1 \
    --core-metric-max-per-task=-1 \
    --core-metric-every=3000 \
    --target-param-data-ratio=12 ) \
  2>&1 | tee ./logs/speedrun_d24_feb01-rope_chunk_mlp_lr_1x2x.log
