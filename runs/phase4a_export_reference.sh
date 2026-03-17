#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$PWD"

DEPTH="${DEPTH:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
OUTPUT_STEM="${OUTPUT_STEM:-mlx_reference_d${DEPTH}}"

python dev/export_mlx_safetensors.py \
    --depth "$DEPTH" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --init-from-pytorch-reference \
    --output-stem "$OUTPUT_STEM"

echo
echo "Export written to runs/mlx_exports/${OUTPUT_STEM}.json and .safetensors"