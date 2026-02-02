#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${WORKSPACE_DIR}}"

OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE_DIR}/safety_eval_data}"
COCONOT_SAMPLE_SIZE="${COCONOT_SAMPLE_SIZE:-50}"
ARENA_SAMPLE_SIZE="${ARENA_SAMPLE_SIZE:-50}"
SEED="${SEED:-13}"
TRAIN_JSONL="${TRAIN_JSONL:-}" # optional override

cmd=(
  python3 "$ROOT_DIR/dev-safety/safety_eval/step1_prepare_data.py"
  --output-dir "$OUTPUT_DIR"
  --coconot-sample-size "$COCONOT_SAMPLE_SIZE"
  --arena-sample-size "$ARENA_SAMPLE_SIZE"
  --seed "$SEED"
)

if [[ -n "$TRAIN_JSONL" ]]; then
  cmd+=(--train-jsonl "$TRAIN_JSONL")
fi

"${cmd[@]}"
