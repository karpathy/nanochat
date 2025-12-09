#!/bin/bash

# Attention mechanism architecture comparison experiment (CPU/small-scale version)
# Compare MHA, GQA, MLA three architectures
# Parameters reference dev/runcpu.sh, suitable for quick experiments on CPU or single GPU

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Use CPU-friendly small parameters
DEPTH=8                    # Small model, 4 layers
MAX_SEQ_LEN=1024          # Short sequence length
DEVICE_BATCH_SIZE=8       # Small batch size
TOTAL_BATCH_SIZE=1024     # Total batch size
NUM_ITERS=200             # Train 50 steps (can increase to 100-200 for more obvious effects)
EVAL_EVERY=50             # Evaluate every 50 steps
EVAL_TOKENS=4096          # Number of tokens for evaluation
CORE_METRIC_EVERY=50      # Core metric evaluation frequency
CORE_METRIC_MAX=12        # Maximum samples per task for core metrics
SAMPLE_EVERY=50           # Sampling frequency

# To see more obvious training effects, can increase to 200-500 steps
# NUM_ITERS=200

echo "================================"
echo "Attention Mechanism Comparison Experiment"
echo "Model scale: depth=$DEPTH"
echo "Training steps: $NUM_ITERS"
echo "================================"

echo ""
echo "================================"
echo "Experiment 1: MHA Baseline (Standard Multi-Head Attention)"
echo "================================"
python -m scripts.base_train \
    --depth=$DEPTH \
    --max_seq_len=$MAX_SEQ_LEN \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --total_batch_size=$TOTAL_BATCH_SIZE \
    --num_iterations=$NUM_ITERS \
    --eval_every=$EVAL_EVERY \
    --eval_tokens=$EVAL_TOKENS \
    --core_metric_every=$CORE_METRIC_EVERY \
    --core_metric_max_per_task=$CORE_METRIC_MAX \
    --sample_every=$SAMPLE_EVERY \
    --run=attn_mha \
    --model_tag=attn_mha


echo ""
echo "================================"
echo "Experiment 3: MLA (Multi-Head Latent Attention)"
echo "================================"
# d_latent = n_embd / 4, for depth=4, n_embd=256, so d_latent=64
MLA_ENABLED=1 MLA_D_LATENT=64 \
python -m scripts.base_train \
    --depth=$DEPTH \
    --max_seq_len=$MAX_SEQ_LEN \
    --device_batch_size=$DEVICE_BATCH_SIZE \
    --total_batch_size=$TOTAL_BATCH_SIZE \
    --num_iterations=$NUM_ITERS \
    --eval_every=$EVAL_EVERY \
    --eval_tokens=$EVAL_TOKENS \
    --core_metric_every=$CORE_METRIC_EVERY \
    --core_metric_max_per_task=$CORE_METRIC_MAX \
    --sample_every=$SAMPLE_EVERY \
    --run=attn_mla \
    --model_tag=attn_mla
