#!/bin/bash

# Modal nanochat runner. Two modes, picked with MODE (default: cpu):
#   MODE=cpu  runs/runcpu.sh on a heavy CPU machine (64 cores target, no GPU, no GPU payment gate)
#   MODE=gpu  runs/speedrun.sh on 8×H100 (needs a GPU payment method and entitlement)
# One command: bash runs/runmodal.sh          (CPU)
#              MODE=gpu bash runs/runmodal.sh (GPU, later)
# Prerequisites: Modal account, `modal setup`, local `modal` CLI.
# Skip the prompt with RUNMODAL_YES=1.

set -euo pipefail

MODE="${MODE:-cpu}"
case "$MODE" in
    cpu|gpu) ;;
    *) echo "ERROR: MODE must be cpu or gpu (got '$MODE')" >&2; exit 1 ;;
esac

# Fail-fast: require modal on PATH
if ! command -v modal &>/dev/null; then
    echo "ERROR: 'modal' not found on PATH. Install with: pip install modal (or uv tool run modal)" >&2
    echo "Then run 'modal setup'." >&2
    exit 1
fi

# Guard: refuse if another run on the shared Volume is already active.
# JSON, not the table: the table wraps long names when piped and hides them.
_LIVE_CHECK='import json,sys
print("\n".join(a["app_id"] for a in json.load(sys.stdin) if a.get("description") == "nanochat-speedrun" and a.get("stopped_at") is None))'
if _APP_JSON=$(modal app list --json 2>/dev/null) && _RUNNING_APP=$(printf '%s' "$_APP_JSON" | python3 -c "$_LIVE_CHECK"); then
    :
else
    echo "WARNING: could not check for running Modal apps; proceeding without the concurrency guard." >&2
    _RUNNING_APP=""
fi
if [ -n "$_RUNNING_APP" ] && [ -z "${RUNMODAL_ALLOW_CONCURRENT:-}" ]; then
    echo "" >&2
    echo "ERROR: A nanochat Modal app appears to be running already." >&2
    echo "  Wait for it to finish, or set RUNMODAL_ALLOW_CONCURRENT=1 to override." >&2
    echo "" >&2
    exit 1
fi

# Print the mode-specific header with cost and time
if [ "$MODE" = "cpu" ]; then
    cat <<'EOF'
============================================================================
 nanochat on Modal — CPU mode (runcpu profile, small model, whole pipeline)

 Machine: heavy CPU container, up to 64 cores / 128 GiB, no GPU
 Cost:    scales with cores. Check Modal's current CPU price before running.
          The first run measures real time; the cost is printed there too.
 Time:    unmeasured. Keep laptop awake; use screen/tmux (see docs/RUNNING.md)

 On completion: tokenizer + chat checkpoint downloaded to ~/.cache/nanochat-modal/cpu
   NANOCHAT_BASE_DIR=~/.cache/nanochat-modal/cpu python3 -m scripts.chat_cli
============================================================================
EOF
else
    cat <<'EOF'
============================================================================
 nanochat on Modal — GPU mode (speedrun profile, 8×H100)

 Cost: ~$40–60 (~$3/GPU/hr × 8 GPUs × ~1.5–2h). First run higher (image bake).
 Time: ~1.5–2h training + prefetch + image build. Keep laptop awake.

 On completion: tokenizer + chat checkpoint downloaded to ~/.cache/nanochat-modal/gpu
   NANOCHAT_BASE_DIR=~/.cache/nanochat-modal/gpu python3 -m scripts.chat_cli
============================================================================
EOF
fi

# Spend confirmation (unless RUNMODAL_YES=1)
if [ -z "${RUNMODAL_YES:-}" ]; then
    echo ""
    read -p "Proceed with MODE=$MODE? [y/N] " -r proceed
    if [[ ! "$proceed" =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

# Launch Modal app
echo ""
echo "Starting Modal run (runs/modal_run.py, mode=$MODE)..."
echo ""
# Every run gets an id and an archive folder: the full log is kept locally next to the Volume
# copy (runs/<run-id>/run.json and events.jsonl). The code revision ties results to the code.
RUN_ID="$(date -u +%Y%m%d-%H%M%S)-$MODE"
CODE_REV="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
ARCHIVE_ROOT="${NANOCHAT_MODAL_ARCHIVE:-$HOME/.cache/nanochat-modal/runs}"
ARCHIVE_DIR="$ARCHIVE_ROOT/$RUN_ID"
mkdir -p "$ARCHIVE_DIR"
echo "Run id: $RUN_ID (code $CODE_REV). Local log: $ARCHIVE_DIR/run.log"
echo ""
NANOCHAT_MODAL_MODE="$MODE" modal run runs/modal_run.py --mode "$MODE" \
    --run-id "$RUN_ID" --code-rev "$CODE_REV" 2>&1 | tee "$ARCHIVE_DIR/run.log"
