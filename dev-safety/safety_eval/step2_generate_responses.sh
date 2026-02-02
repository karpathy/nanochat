#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${WORKSPACE_DIR}}"

DATA_DIR="${DATA_DIR:-${WORKSPACE_DIR}/safety_eval_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE_DIR}/eval_generations}"

COCONOT_FILE="${COCONOT_FILE:-${DATA_DIR}/coconot_eval_50.jsonl}"
ARENA_FILE="${ARENA_FILE:-${DATA_DIR}/arena_eval_50.jsonl}"

BASELINE_CHECKPOINT_DIR="${BASELINE_CHECKPOINT_DIR:-sft_checkpoints/d12}"
SAFETY_CHECKPOINT_DIR="${SAFETY_CHECKPOINT_DIR:-safety_finetune_runs_20x/sft_checkpoints/d12}"

MODEL_A_NAME="${MODEL_A_NAME:-baseline_d12}"
MODEL_B_NAME="${MODEL_B_NAME:-safety_ft_20x_d12}"

TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_K="${TOP_K:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SEED="${SEED:-13}"
DEVICE_TYPE="${DEVICE_TYPE:-}"
DTYPE="${DTYPE:-bfloat16}"

run_model() {
  local input_file="$1"
  local model_name="$2"
  local checkpoint_dir="$3"

  python3 "$ROOT_DIR/dev-safety/safety_eval/step2_generate_responses.py" \
    --input "$input_file" \
    --output-root "$OUTPUT_ROOT" \
    --model-name "$model_name" \
    --checkpoint-dir "$checkpoint_dir" \
    --temperature "$TEMPERATURE" \
    --top-k "$TOP_K" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --seed "$SEED" \
    --dtype "$DTYPE" \
    ${DEVICE_TYPE:+--device-type "$DEVICE_TYPE"}
}

run_model "$COCONOT_FILE" "$MODEL_A_NAME" "$BASELINE_CHECKPOINT_DIR"
run_model "$ARENA_FILE" "$MODEL_A_NAME" "$BASELINE_CHECKPOINT_DIR"

run_model "$COCONOT_FILE" "$MODEL_B_NAME" "$SAFETY_CHECKPOINT_DIR"
run_model "$ARENA_FILE" "$MODEL_B_NAME" "$SAFETY_CHECKPOINT_DIR"
