#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

# Base dir hosts base checkpoints + shared data.
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${WORKSPACE_DIR}}"
export SAFETY_CURATION_DIR="${SAFETY_CURATION_DIR:-${NANOCHAT_BASE_DIR}/safety_response_curation}"
export SAFETY_FINETUNE_DIR="${SAFETY_FINETUNE_DIR:-${NANOCHAT_BASE_DIR}/safety_finetune_runs}"
ORIG_BASE_DIR="$NANOCHAT_BASE_DIR"
EXTRA_JSONL="${EXTRA_JSONL:-${SAFETY_CURATION_DIR}/step4_sft.jsonl}"

mkdir -p "$SAFETY_FINETUNE_DIR"

# Sanity check: extra data must exist and be non-empty
if [ ! -s "$EXTRA_JSONL" ]; then
  echo "Extra JSONL not found or empty: $EXTRA_JSONL"
  echo "Set EXTRA_JSONL or SAFETY_CURATION_DIR to the correct location."
  exit 1
fi
EXTRA_LINES="$(wc -l < "$EXTRA_JSONL" | tr -d ' ')"
echo "Using extra JSONL: $EXTRA_JSONL ($EXTRA_LINES lines)"

# Ensure identity conversations are available in the base dir.
if [ ! -f "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" ]; then
  curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" \
    https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
fi

# Hardware config
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

# Hyperparameters (match d12 defaults unless overridden)
MODEL_TAG="${MODEL_TAG:-d12}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-128}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
EMBEDDING_LR="${EMBEDDING_LR:-0.38}"
UNEMBEDDING_LR="${UNEMBEDDING_LR:-0.004}"
MATRIX_LR="${MATRIX_LR:-0.027}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.13}"

WANDB_RUN="${WANDB_RUN:-dummy}"

# SFT with extra curated safety data
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
  --run "$WANDB_RUN" \
  --model-tag "$MODEL_TAG" \
  --device-batch-size "$DEVICE_BATCH_SIZE" \
  --total-batch-size "$TOTAL_BATCH_SIZE" \
  --max-seq-len "$MAX_SEQ_LEN" \
  --embedding-lr "$EMBEDDING_LR" \
  --unembedding-lr "$UNEMBEDDING_LR" \
  --matrix-lr "$MATRIX_LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --extra-jsonl "$EXTRA_JSONL" \
  --output-base-dir "$SAFETY_FINETUNE_DIR"

# Evaluate the new SFT checkpoints from the finetune runs directory
TOKENIZER_SRC="${ORIG_BASE_DIR}/tokenizer"
TOKENIZER_DST="${SAFETY_FINETUNE_DIR}/tokenizer"
if [ ! -d "$TOKENIZER_DST" ]; then
  if [ ! -d "$TOKENIZER_SRC" ]; then
    echo "Tokenizer not found at $TOKENIZER_SRC. Run tokenizer training first (e.g., dev-safety/d12_train.sh) or set NANOCHAT_BASE_DIR to a location that already has tokenizer artifacts."
    exit 1
  fi
  cp -R "$TOKENIZER_SRC" "$TOKENIZER_DST"
fi

NANOCHAT_BASE_DIR="$SAFETY_FINETUNE_DIR" \
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- -i sft -g "$MODEL_TAG"
