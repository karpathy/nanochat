#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$PWD"

MANIFEST="${1:-${MANIFEST:-runs/mlx_exports/phase2_d4_l_mps_step20.json}}"
PROMPT="${PROMPT:-The chemical formula of water is}"

python dev/benchmark_swift_vs_python.py \
    --manifest "$MANIFEST" \
    --prompt "$PROMPT" \
    --max-new-tokens "${MAX_NEW_TOKENS:-32}" \
    --warmup-runs "${WARMUP_RUNS:-2}" \
    --timed-runs "${TIMED_RUNS:-5}" \
    --swift-device "${SWIFT_DEVICE:-both}"