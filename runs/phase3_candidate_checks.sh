#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$PWD"

DEPTH="${DEPTH:-32}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
STEPS="${STEPS:-6}"
INPUT_MODE="${INPUT_MODE:-repeated}"
EXECUTION_MODE="${EXECUTION_MODE:-compiled}"

echo "== AdamW sanity check =="
python dev/mlx_training_check.py \
    --depth "$DEPTH" \
    --device-batch-size "$DEVICE_BATCH_SIZE" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --warmup-steps "$WARMUP_STEPS" \
    --steps "$STEPS" \
    --input-mode "$INPUT_MODE" \
    --execution-mode "$EXECUTION_MODE" \
    --matrix-optimizer adamw \
    --log-prefix phase3_script_check_adamw

echo
echo "== Muon candidate sanity check (attn_only, ns=2, float16) =="
python dev/mlx_training_check.py \
    --depth "$DEPTH" \
    --device-batch-size "$DEVICE_BATCH_SIZE" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --warmup-steps "$WARMUP_STEPS" \
    --steps "$STEPS" \
    --input-mode "$INPUT_MODE" \
    --execution-mode "$EXECUTION_MODE" \
    --matrix-optimizer muon \
    --muon-block-groups attn_only \
    --muon-ns-steps 2 \
    --muon-orthogonalize-dtype float16 \
    --log-prefix phase3_script_check_muon_attn_only_ns2_f16