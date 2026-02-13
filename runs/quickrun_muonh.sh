#!/bin/bash

# Quickrun: GPT-Gamma + MuonH (Hyperball)
# - Parameterized RMSNorm (learnable gamma)
# - Per-block projection scalars
# - Hyperball or Muon for matrix params
#
# Examples:
#   bash runs/quickrun_muonh.sh
#   WANDB_RUN=exp1 bash runs/quickrun_muonh.sh
#   FP8=1 FP8_RECIPE=tensorwise bash runs/quickrun_muonh.sh
#   DEPTH=16 bash runs/quickrun_muonh.sh

set -e

# -----------------------------------------------------------------------------
# Config

DEPTH="${DEPTH:-26}"
NUM_SHARDS="${NUM_SHARDS:-370}"      # default for d24 @ ratio~11
TARGET_RATIO="${TARGET_RATIO:-10.5}"
WINDOW_PATTERN="${WINDOW_PATTERN:-SSSL}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"  # -1 = auto-compute optimal (Power Lines paper)

NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
if [ "$NPROC_PER_NODE" -eq 0 ]; then
    NPROC_PER_NODE=1
fi

# Optimizer
MATRIX_OPTIMIZER="${MATRIX_OPTIMIZER:-hyperball}"
SCALAR_LR="${SCALAR_LR:-0.5}"
MATRIX_LR="${MATRIX_LR:-0.02}"
WARMDOWN_RATIO="${WARMDOWN_RATIO:-0.3}"
MATRIX_WARMDOWN_RATIO="${MATRIX_WARMDOWN_RATIO:-1.0}"

# AdamW
EMBEDDING_LR="${EMBEDDING_LR:-0.3}"
UNEMBEDDING_LR="${UNEMBEDDING_LR:-0.004}"
NORM_LR="${NORM_LR:-0.1}"

# Wandb
export WANDB_ENTITY="${WANDB_ENTITY:-xingyu20}"
export WANDB_PROJECT="${WANDB_PROJECT:-nanochat}"
WANDB_RUN="${WANDB_RUN:-muonh_d${DEPTH}_ratio${TARGET_RATIO}_feb_11_no_gamma}"
MODEL_TAG="${MODEL_TAG:-d${DEPTH}_gamma_muonh}"

# FP8 (default enabled)c
FP8="${FP8:-1}"
FP8_ARGS=""
if [ "${FP8:-0}" -eq 1 ]; then
    FP8_RECIPE="${FP8_RECIPE:-tensorwise}"
    FP8_ARGS="--fp8 --fp8-recipe=${FP8_RECIPE}"
fi

# -----------------------------------------------------------------------------
# Paths and cache

export OMP_NUM_THREADS=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export NANOCHAT_BASE_DIR="$PROJECT_ROOT/cache"
export TORCHINDUCTOR_CACHE_DIR="$NANOCHAT_BASE_DIR/torch_inductor"
export TRITON_CACHE_DIR="$NANOCHAT_BASE_DIR/triton"
export TMPDIR="$NANOCHAT_BASE_DIR/tmp"
mkdir -p "$NANOCHAT_BASE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"

# -----------------------------------------------------------------------------
# Print summary

echo "=============================================="
echo "Quickrun (GPT-Gamma + MuonH)"
echo "=============================================="
echo "Project root:      $PROJECT_ROOT"
echo "Cache dir:         $NANOCHAT_BASE_DIR"
echo "Depth:             $DEPTH"
echo "Num shards:        $NUM_SHARDS"
echo "Target ratio:      $TARGET_RATIO"
echo "Window pattern:    $WINDOW_PATTERN"
echo "Num GPUs:          $NPROC_PER_NODE"
echo "Device batch size: $DEVICE_BATCH_SIZE"
echo "Total batch size:  $TOTAL_BATCH_SIZE"
echo "Matrix optimizer:  $MATRIX_OPTIMIZER"
echo "Matrix LR:         $MATRIX_LR"
echo "Norm LR:           $NORM_LR"
echo "Adam LRs:          embedding=$EMBEDDING_LR, unembedding=$UNEMBEDDING_LR, scalar=$SCALAR_LR"
echo "Warmdown ratio:    adam=$WARMDOWN_RATIO, matrix=$MATRIX_WARMDOWN_RATIO"
echo "Wandb run:         $WANDB_RUN"
echo "Model tag:         $MODEL_TAG"
if [ "${FP8:-0}" -eq 1 ]; then
    echo "FP8:               enabled ($FP8_RECIPE)"
fi
echo "=============================================="

cd "$PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Python venv

if [ ! -d ".venv" ]; then
    echo "Setting up Python environment..."
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    uv venv
    uv sync --extra gpu
fi
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Data + tokenizer

echo ""
echo "Downloading $NUM_SHARDS data shards..."
python -m nanochat.dataset -n "$NUM_SHARDS"

echo ""
TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
if [ -f "$TOKENIZER_DIR/token_bytes.pt" ]; then
    echo "Tokenizer already exists at $TOKENIZER_DIR, skipping training."
else
    echo "Training tokenizer..."
    python -m scripts.tok_train --max-chars=500000000 --vocab-size=32768
fi

# -----------------------------------------------------------------------------
# Train

echo ""
echo "Starting pretraining (depth=$DEPTH)..."

TRAIN_ARGS=(
    --depth=$DEPTH
    --run=$WANDB_RUN
    --model-tag=$MODEL_TAG
    --window-pattern=$WINDOW_PATTERN
    --target-param-data-ratio=$TARGET_RATIO
    --device-batch-size=$DEVICE_BATCH_SIZE
    --total-batch-size=$TOTAL_BATCH_SIZE
    --matrix-optimizer=$MATRIX_OPTIMIZER
    --matrix-lr=$MATRIX_LR
    --warmdown-ratio=$WARMDOWN_RATIO
    --matrix-warmdown-ratio=$MATRIX_WARMDOWN_RATIO
    --embedding-lr=$EMBEDDING_LR
    --unembedding-lr=$UNEMBEDDING_LR
    --norm-lr=$NORM_LR
    --scalar-lr=$SCALAR_LR
    --core-metric-every=${CORE_METRIC_EVERY:-2000}
    --sample-every=${SAMPLE_EVERY:--1}
    --save-every=${SAVE_EVERY:--1}
)

if [ "$NPROC_PER_NODE" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
        "${TRAIN_ARGS[@]}" $FP8_ARGS
else
    python -m scripts.base_train \
        "${TRAIN_ARGS[@]}" $FP8_ARGS
fi

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo "Checkpoint saved to: $NANOCHAT_BASE_DIR/base_checkpoints/${MODEL_TAG}/"