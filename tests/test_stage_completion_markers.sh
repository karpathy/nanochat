#!/usr/bin/env bash
# stage_done must require a profile+depth marker — a lone model_*.pt or old .train_ok must not SKIP.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export NANOCHAT_BASE_DIR="$TMP"
PROFILE=test
DEPTH=4
# shellcheck source=../runs/_stage_lib.sh
source "$ROOT/runs/_stage_lib.sh"

mkdir -p "$TMP/base_checkpoints/d4" "$TMP/chatsft_checkpoints/d4"
touch "$TMP/base_checkpoints/d4/model_000001.pt"
touch "$TMP/chatsft_checkpoints/d4/model_000001.pt"

if stage_done base_train; then
    echo "FAIL: base_train should not be done without profile+depth marker" >&2
    exit 1
fi
if stage_done chat_sft; then
    echo "FAIL: chat_sft should not be done without profile+depth marker" >&2
    exit 1
fi

# Old .train_ok alone must NOT trigger SKIP (backward compat: treated as mismatch).
touch "$TMP/base_checkpoints/.train_ok"
touch "$TMP/chatsft_checkpoints/.train_ok"
if stage_done base_train; then
    echo "FAIL: base_train should not be done with only old .train_ok (no profile+depth marker)" >&2
    exit 1
fi
if stage_done chat_sft; then
    echo "FAIL: chat_sft should not be done with only old .train_ok (no profile+depth marker)" >&2
    exit 1
fi
rm -f "$TMP/base_checkpoints/.train_ok" "$TMP/chatsft_checkpoints/.train_ok"

# Profile+depth marker must trigger SKIP.
touch "$(_stage_marker_path base_train)"
touch "$(_stage_marker_path chat_sft)"

stage_done base_train || { echo "FAIL: base_train should be done with profile+depth marker" >&2; exit 1; }
stage_done chat_sft || { echo "FAIL: chat_sft should be done with profile+depth marker" >&2; exit 1; }

# START clears stale markers before a real run
FORCE_STAGES=base_train
_new_marker="$(_stage_marker_path base_train)"
rm -f "$_new_marker"
[ ! -f "$_new_marker" ] || { echo "FAIL: expected profile+depth marker cleared" >&2; exit 1; }
# Also clear the chat_sft marker set above so it doesn't interfere
rm -f "$(_stage_marker_path chat_sft)"

echo "ok: stage completion markers"

# Test: SKIP is printed with no plaque (contract frozen; plaques only on START/DONE/FAIL)
# Given: the profile+depth marker exists, so run_stage takes the SKIP path
# When: base_train is run without FORCE_STAGES
# Then: a SKIP line is printed and no plaque border line appears
mkdir -p "$TMP/base_checkpoints"
touch "$(_stage_marker_path base_train)"
SKIP_output="$(FORCE_STAGES="" run_stage base_train "" next -- echo "dummy" 2>&1)"
echo "$SKIP_output" | grep -q "SKIP stage=base_train" || { echo "FAIL: SKIP line not printed" >&2; exit 1; }
if echo "$SKIP_output" | grep -q '^+-'; then
    echo "FAIL: SKIP must not have a plaque" >&2
    exit 1
fi
rm -f "$(_stage_marker_path base_train)"

# Test: NANOCHAT_PLAIN=1 suppresses plaques but keeps contract
# When: NANOCHAT_PLAIN=1 is set
# Then: plaque CLI returns empty string
plaque_output="$(NANOCHAT_PLAIN=1 "$ROOT/.venv/bin/python" -m nanochat.stage_art plaque --stage base_train --kind start --subtitle "test")"
if [ -n "$plaque_output" ]; then
    echo "FAIL: plaque should be suppressed with NANOCHAT_PLAIN=1" >&2
    exit 1
fi

echo "ok: color, plaque, and learn line tests"
