#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$PWD"

python dev/benchmark_swift_vs_python.py \
    --manifest runs/mlx_exports/phase2_d4_l_mps_step20.json \
    --prompt "The chemical formula of water is" \
    --max-new-tokens "${MAX_NEW_TOKENS:-32}" \
    --warmup-runs "${WARMUP_RUNS:-2}" \
    --timed-runs "${TIMED_RUNS:-5}" \
    --swift-device "${SWIFT_DEVICE:-both}"