#!/bin/bash

# Showing an example run for exercising some of the code paths on the CPU (or MPS on Macbooks)
# Run as:
# bash dev/runcpu.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.
# This is also why I hide this script away in dev/

# all the setup stuff
export OMP_NUM_THREADS=1
NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Memory-based configuration for macOS
# Detect system memory (in GB) or allow manual override
if [ -z "$MEMORY_SIZE" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        MEMORY_SIZE=$(sysctl hw.memsize | awk '{print int($2/1024/1024/1024)}')
        echo "Auto-detected macOS memory: ${MEMORY_SIZE}GB"
    else
        # Linux fallback - assume conservative
        MEMORY_SIZE=16
        echo "Non-macOS system, using conservative: ${MEMORY_SIZE}GB"
    fi
fi

# Calculate optimal batch sizes based on available memory
# Note: total_batch_size must be divisible by (device_batch_size * max_seq_len)
# With max_seq_len=1024: device_batch_size * 1024 must divide total_batch_size
if [ $MEMORY_SIZE -ge 128 ]; then
    DEVICE_BATCH_SIZE=16
    TOTAL_BATCH_SIZE=16384    # 16 * 1024 = 16384
    EVAL_TOKENS=16384
    SPLIT_TOKENS=16384
    echo "Memory profile: 128GB+ (High performance)"
elif [ $MEMORY_SIZE -ge 64 ]; then
    DEVICE_BATCH_SIZE=8
    TOTAL_BATCH_SIZE=8192     # 8 * 1024 = 8192
    EVAL_TOKENS=8192
    SPLIT_TOKENS=8192
    echo "Memory profile: 64GB (Good performance)"
elif [ $MEMORY_SIZE -ge 32 ]; then
    DEVICE_BATCH_SIZE=4
    TOTAL_BATCH_SIZE=4096     # 4 * 1024 = 4096
    EVAL_TOKENS=4096
    SPLIT_TOKENS=4096
    echo "Memory profile: 32GB (Moderate performance)"
else
    DEVICE_BATCH_SIZE=1
    TOTAL_BATCH_SIZE=1024     # 1 * 1024 = 1024
    EVAL_TOKENS=2048
    SPLIT_TOKENS=2048
    echo "Memory profile: <32GB (Conservative)"
fi

echo "Using: device_batch_size=$DEVICE_BATCH_SIZE, total_batch_size=$TOTAL_BATCH_SIZE"
echo ""
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $NANOCHAT_BASE_DIR
fi

# wipe the report
python -m nanochat.report reset

# train tokenizer on ~1B characters
python -m nanochat.dataset -n 4
python -m scripts.tok_train --max_chars=1000000000
python -m scripts.tok_eval

# train a very small 4 layer model on the CPU/MPS
# batch sizes are now optimized based on available memory
# we only run 50 steps of optimization (bump this to get better results)
python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=1024 \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --total_batch_size=$TOTAL_BATCH_SIZE \
    --eval_every=50 \
    --eval_tokens=$EVAL_TOKENS \
    --core_metric_every=50 \
    --core_metric_max_per_task=12 \
    --sample_every=50 \
    --num_iterations=50
python -m scripts.base_loss --device_batch_size=$DEVICE_BATCH_SIZE --split_tokens=$SPLIT_TOKENS
python -m scripts.base_eval --max-per-task=5

# midtraining
python -m scripts.mid_train \
    --max_seq_len=1024 \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --eval_every=50 \
    --eval_tokens=$EVAL_TOKENS \
    --total_batch_size=$TOTAL_BATCH_SIZE \
    --num_iterations=100
# eval results will be terrible, this is just to execute the code paths.
# note that we lower the execution memory limit to 1MB to avoid warnings on smaller systems
python -m scripts.chat_eval -i mid --max-new-tokens=128 --max-problems=20

# SFT
python -m scripts.chat_sft \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --target_examples_per_step=$((DEVICE_BATCH_SIZE * 2)) \
    --num_iterations=100 \
    --eval_steps=4 \
    --eval_metrics_max_problems=16

# Chat CLI
# python -m scripts.chat_cli -p "Why is the sky blue?"

# Chat Web
# python -m scripts.chat_web

python -m nanochat.report generate
