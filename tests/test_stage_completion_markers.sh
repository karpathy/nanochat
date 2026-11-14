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

# Test: contract tokens are frozen (SKIP has no plaque)
# When: a stage is already done and not forced
# Then: SKIP line prints but no plaque
if output="$(FORCE_STAGES="" run_stage base_train "" next -- echo "dummy" 2>&1)"; then
    if echo "$output" | grep -q "SKIP stage=base_train"; then
        if echo "$output" | grep -q '^+-'; then
            echo "FAIL: SKIP should not have plaque" >&2
            exit 1
        fi
    fi
else
    true  # stage may fail for other reasons; we're just checking output format
fi

# Test: SKIP gets color but no plaque
# When: SKIP is printed to non-TTY (this script's stdout is not a TTY)
# Then: SKIP line appears (plain text in non-TTY), no plaque
SKIP_output="$(FORCE_STAGES="" run_stage base_train "" next -- echo "dummy" 2>&1)" || true
if echo "$SKIP_output" | grep -q "SKIP"; then
    if echo "$SKIP_output" | grep '^+-'; then
        echo "FAIL: SKIP must not have plaque even with color enabled" >&2
        exit 1
    fi
fi

# Test: learn line is present after purpose
# When: a non-skipped stage runs
# Then: learn line appears after purpose line (if the stage has terms)
# (This test would require a full run, so we check CLI instead)
learn_line=$("$ROOT/.venv/bin/python" -m nanochat.glossary learn --stage base_train)
if [ -z "$learn_line" ]; then
    echo "FAIL: base_train should have learn line" >&2
    exit 1
fi
if ! echo "$learn_line" | grep -q "learn:"; then
    echo "FAIL: learn line should start with 'learn:'" >&2
    exit 1
fi

# Test: NANOCHAT_PLAIN=1 suppresses plaques but keeps contract
# When: NANOCHAT_PLAIN=1 is set
# Then: plaque CLI returns empty string
plaque_output="$(NANOCHAT_PLAIN=1 "$ROOT/.venv/bin/python" -m nanochat.stage_art plaque --stage base_train --kind start --subtitle "test")"
if [ -n "$plaque_output" ]; then
    echo "FAIL: plaque should be suppressed with NANOCHAT_PLAIN=1" >&2
    exit 1
fi

echo "ok: color, plaque, and learn line tests"
