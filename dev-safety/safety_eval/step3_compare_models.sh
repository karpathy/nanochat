#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${WORKSPACE_DIR}}"
export BASE_URL="${BASE_URL:-https://integrate.api.nvidia.com/v1}"

GENERATIONS_ROOT="${GENERATIONS_ROOT:-${WORKSPACE_DIR}/eval_generations}"
RESULTS_ROOT="${RESULTS_ROOT:-${WORKSPACE_DIR}/safety_eval_results}"
DATASETS="${DATASETS:-coconot_eval_50,arena_eval_50}"

MODEL_A_NAME="${MODEL_A_NAME:-baseline_d12}"
MODEL_B_NAME="${MODEL_B_NAME:-safety_ft_20x_d12}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"
API_KEY_ENV="${API_KEY_ENV:-API_KEY}"
POLICY_FILE="${POLICY_FILE:-}"
WORKERS="${WORKERS:-4}"
RETRIES="${RETRIES:-2}"
RETRY_BACKOFF="${RETRY_BACKOFF:-1.5}"
MAX_PROMPTS="${MAX_PROMPTS:-}"

cmd=(
  python3 "$ROOT_DIR/dev-safety/safety_eval/step3_compare_models.py"
  --model-a-name "$MODEL_A_NAME"
  --model-b-name "$MODEL_B_NAME"
  --datasets "$DATASETS"
  --generations-root "$GENERATIONS_ROOT"
  --output-root "$RESULTS_ROOT"
  --judge-model "$JUDGE_MODEL"
  --base-url "$BASE_URL"
  --api-key-env "$API_KEY_ENV"
  --workers "$WORKERS"
  --retries "$RETRIES"
  --retry-backoff "$RETRY_BACKOFF"
)

if [[ -n "$POLICY_FILE" ]]; then
  cmd+=(--policy-file "$POLICY_FILE")
fi

if [[ -n "$MAX_PROMPTS" ]]; then
  cmd+=(--max-prompts "$MAX_PROMPTS")
fi

"${cmd[@]}"
