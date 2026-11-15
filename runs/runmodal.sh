#!/bin/bash

# Modal nanochat runner. Two modes, picked with MODE (default: cpu):
#   MODE=cpu  runs/runcpu.sh on a heavy CPU machine (64 cores target, no GPU, no GPU payment gate)
#   MODE=gpu  runs/speedrun.sh on 8×H100 (needs a GPU payment method and entitlement)
# Runs detached by default (ATTACHED=1 to stream here).
# One command: bash runs/runmodal.sh          (CPU)
#              MODE=gpu bash runs/runmodal.sh (GPU, later)
#              bash runs/runmodal.sh finish <run-id>  (download artifacts after run stops)
# Prerequisites: Modal account, `modal setup`, local `modal` CLI.
# Skip the prompt with RUNMODAL_YES=1.

set -euo pipefail

# Source Modal config (NANOCHAT_MODAL_VOLUME)
SCRIPTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPTDIR/_modal_config.sh"

cmd="${1:-launch}"
case "$cmd" in
    finish)
        # Download artifacts after a run stops
        RUN_ID="${2:-}"
        if [ -z "$RUN_ID" ]; then
            echo "ERROR: usage: bash runs/runmodal.sh finish <run-id>" >&2
            exit 1
        fi
        exec bash "$SCRIPTDIR/modal_peek.sh" finish-download "$RUN_ID"
        ;;
    launch)
        # Fall through to main launch logic below
        ;;
    *)
        echo "ERROR: unknown command: $cmd" >&2
        exit 1
        ;;
esac

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
if [ -n "$_RUNNING_APP" ] && [ "${RUNMODAL_ALLOW_CONCURRENT:-}" != "1" ]; then
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
 Time:    unmeasured. Run detached. Use screen/tmux to keep terminal alive.

 On completion: artifacts on Volume. Download with: bash runs/runmodal.sh finish <run-id>
   Then chat: NANOCHAT_BASE_DIR=~/.cache/nanochat-modal/cpu python3 -m scripts.chat_cli
============================================================================
EOF
else
    cat <<'EOF'
============================================================================
 nanochat on Modal — GPU mode (speedrun profile, 8×H100)

 Cost: ~$40–60 (~$3/GPU/hr × 8 GPUs × ~1.5–2h). First run higher (image bake).
 Time: ~1.5–2h training + prefetch + image build. Run detached.

 On completion: artifacts on Volume. Download with: bash runs/runmodal.sh finish <run-id>
   Then chat: NANOCHAT_BASE_DIR=~/.cache/nanochat-modal/gpu python3 -m scripts.chat_cli
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

# Detached launch (unless ATTACHED=1)
DETACH_FLAG="--detach"
if [ -z "${ATTACHED:-}" ]; then
    echo "Running detached: the run continues if this terminal disconnects."
    echo "Use ATTACHED=1 to stream here, or run: bash runs/modal_peek.sh logs"
    echo ""
else
    DETACH_FLAG=""
    echo "Running attached: logs stream here until the run finishes or is cancelled."
    echo ""
fi

# Launch with mode prefix
RUN_LOG="$ARCHIVE_DIR/run.log"
set +e
NANOCHAT_MODAL_MODE="$MODE" NANOCHAT_MODAL_VOLUME="$NANOCHAT_MODAL_VOLUME" modal run \
    runs/modal_run.py --mode "$MODE" --run-id "$RUN_ID" --code-rev "$CODE_REV" \
    $DETACH_FLAG 2>&1 | tee "$RUN_LOG"
MODAL_STATUS=${PIPESTATUS[0]}
set -e

APP_ID="$(sed -n 's/.*Modal app id: \(ap-[A-Za-z0-9]*\).*/\1/p' "$RUN_LOG" | sed -n 1p)"
if [ -n "$APP_ID" ]; then
    echo "$APP_ID $MODE" > "$ARCHIVE_DIR/app.id"
    echo "App ID saved to: $ARCHIVE_DIR/app.id" >&2
else
    echo "WARNING: could not find the Modal app id in $RUN_LOG." >&2
    echo "  Find it with: bash runs/modal_peek.sh status" >&2
fi

if [ "$MODAL_STATUS" -ne 0 ]; then
    echo "modal run exited with status $MODAL_STATUS. See $RUN_LOG" >&2
    exit "$MODAL_STATUS"
fi

if [ -n "$DETACH_FLAG" ]; then
    touch "$ARCHIVE_DIR/PENDING_DOWNLOAD"
    echo "To download artifacts: bash runs/runmodal.sh finish $RUN_ID" >&2
else
    bash "$SCRIPTDIR/modal_peek.sh" finish-download "$RUN_ID"
fi
