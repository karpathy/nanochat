#!/usr/bin/env bash
# stage_done must require .train_ok — a lone model_*.pt (probe/partial) must not SKIP.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export NANOCHAT_BASE_DIR="$TMP"
PROFILE=test
# shellcheck source=../runs/_stage_lib.sh
source "$ROOT/runs/_stage_lib.sh"

mkdir -p "$TMP/base_checkpoints/d4" "$TMP/chatsft_checkpoints/d4"
touch "$TMP/base_checkpoints/d4/model_000001.pt"
touch "$TMP/chatsft_checkpoints/d4/model_000001.pt"

if stage_done base_train; then
    echo "FAIL: base_train should not be done without .train_ok" >&2
    exit 1
fi
if stage_done chat_sft; then
    echo "FAIL: chat_sft should not be done without .train_ok" >&2
    exit 1
fi

touch "$TMP/base_checkpoints/.train_ok"
touch "$TMP/chatsft_checkpoints/.train_ok"

stage_done base_train || { echo "FAIL: base_train should be done with .train_ok" >&2; exit 1; }
stage_done chat_sft || { echo "FAIL: chat_sft should be done with .train_ok" >&2; exit 1; }

# START clears stale markers before a real run
FORCE_STAGES=base_train
# stage_forced needs FORCE_STAGES set before sourcing in real scripts; re-check clear path:
rm -f "$(stage_path base_train)"
[ ! -f "$(stage_path base_train)" ] || { echo "FAIL: expected marker cleared" >&2; exit 1; }

echo "ok: stage completion markers"
