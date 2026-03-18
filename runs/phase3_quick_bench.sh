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
STEPS="${STEPS:-2}"
INPUT_MODE="${INPUT_MODE:-repeated}"
EXECUTION_MODE="${EXECUTION_MODE:-compiled}"

run_case() {
    local label="$1"
    shift

    echo "== $label =="
    python dev/benchmark_mlx_reference.py \
        --depth "$DEPTH" \
        --device-batch-size "$DEVICE_BATCH_SIZE" \
        --max-seq-len "$MAX_SEQ_LEN" \
        --warmup-steps "$WARMUP_STEPS" \
        --steps "$STEPS" \
        --input-mode "$INPUT_MODE" \
        --execution-mode "$EXECUTION_MODE" \
        "$@"
    echo
}

run_case "AdamW baseline" \
    --matrix-optimizer adamw \
    --log-prefix phase3_script_adamw

run_case "Muon full baseline" \
    --matrix-optimizer muon \
    --log-prefix phase3_script_muon_full

run_case "Muon full tuned (ns=3)" \
    --matrix-optimizer muon \
    --muon-ns-steps 3 \
    --log-prefix phase3_script_muon_ns3

run_case "Muon candidate (attn_only, ns=2, float16)" \
    --matrix-optimizer muon \
    --muon-block-groups attn_only \
    --muon-ns-steps 2 \
    --muon-orthogonalize-dtype float16 \
    --log-prefix phase3_script_muon_attn_only_ns2_f16