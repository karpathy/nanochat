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
WARMUP_STEPS="${WARMUP_STEPS:-2}"
STEPS="${STEPS:-32}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-8}"
INPUT_MODE="${INPUT_MODE:-repeated}"
EXECUTION_MODE="${EXECUTION_MODE:-compiled}"

echo "== Muon candidate longer session (attn_only, ns=2, float16) =="
python dev/mlx_training_session.py \
    --depth "$DEPTH" \
    --device-batch-size "$DEVICE_BATCH_SIZE" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --warmup-steps "$WARMUP_STEPS" \
    --steps "$STEPS" \
    --progress-interval "$PROGRESS_INTERVAL" \
    --input-mode "$INPUT_MODE" \
    --execution-mode "$EXECUTION_MODE" \
    --matrix-optimizer muon \
    --muon-block-groups attn_only \
    --muon-ns-steps 2 \
    --muon-orthogonalize-dtype float16 \
    --log-prefix phase3_script_session_muon_attn_only_ns2_f16