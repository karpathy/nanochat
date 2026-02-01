#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-${WORKSPACE_DIR}}"
export SAFETY_CURATION_DIR="${SAFETY_CURATION_DIR:-${NANOCHAT_BASE_DIR}/safety_response_curation}"
export BASE_URL="${BASE_URL:-https://integrate.api.nvidia.com/v1}"
# export API_KEY="..."

mkdir -p "$SAFETY_CURATION_DIR"

INPUT_FILE="${INPUT_FILE:-${SAFETY_CURATION_DIR}/step2_generations.jsonl}"
OUTPUT_NAME="${OUTPUT_NAME:-step3_verifications.jsonl}"
WORKERS="${WORKERS:-4}"

python3 "$ROOT_DIR/dev-safety/response_curation/step3_verify_responses.py" \
  --input "$INPUT_FILE" \
  --output-dir "$SAFETY_CURATION_DIR" \
  --output-name "$OUTPUT_NAME" \
  --workers "$WORKERS" \
  --base-url "$BASE_URL"
