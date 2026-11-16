#!/bin/bash

# Modal volume commit hook: wait-for-ack protocol after stage success.
# Called by _stage_lib.sh run_stage via NANOCHAT_POST_STAGE_HOOK.
# Contract: delete heartbeat, write request, poll for commit daemon ack.
# Daemon (in modal_run.py) watches request → runs vol.commit() → removes request on success.
# Hook limits: 120s idle (no heartbeat update) or 600s overall per stage.

set -euo pipefail

case "${1:-}" in
    -h|--help|help)
        echo "modal_hook.sh: post-stage commit hook. Not for manual use."
        echo "It runs from _stage_lib.sh (via NANOCHAT_POST_STAGE_HOOK) inside a Modal run."
        echo "It writes a commit request and waits for the daemon in runs/modal_run.py."
        exit 0
        ;;
esac

BASE_DIR="${NANOCHAT_BASE_DIR:?NANOCHAT_BASE_DIR is not set. This hook runs from _stage_lib.sh inside a Modal run; run 'bash runs/modal_hook.sh help' for usage.}"
HEARTBEAT_FILE="$BASE_DIR/.modal_commit_heartbeat"
REQUEST_FILE="$BASE_DIR/.modal_commit_request"
FAILED_FILE="$BASE_DIR/.modal_commit_failed"
IDLE_LIMIT=120  # seconds since last heartbeat update
OVERALL_CEILING=600  # seconds per stage
POLL_INTERVAL=2  # seconds between checks

# Delete any stale heartbeat (from prior stage)
rm -f "$HEARTBEAT_FILE"

# Delete any stale failed file (from prior stage)
rm -f "$FAILED_FILE"

# Write request to wake up the daemon
mkdir -p "$BASE_DIR"
touch "$REQUEST_FILE"

# Poll loop
start_time=$(date +%s)

while true; do
    now=$(date +%s)
    elapsed=$((now - start_time))

    # Check if daemon succeeded (request file removed)
    if [ ! -f "$REQUEST_FILE" ]; then
        exit 0
    fi

    # Check if daemon failed
    if [ -f "$FAILED_FILE" ]; then
        echo "ERROR: daemon reported commit failure (failed file present)" >&2
        exit 1
    fi

    # Check overall ceiling
    if [ "$elapsed" -gt "$OVERALL_CEILING" ]; then
        echo "ERROR: commit hook timeout (overall ceiling $OVERALL_CEILING s exceeded; elapsed=$elapsed)" >&2
        exit 1
    fi

    # Check idle limit (no heartbeat update). Before the first heartbeat exists,
    # idle counts from hook start, so a daemon that dies immediately still fails fast.
    if [ -f "$HEARTBEAT_FILE" ]; then
        last_seen=$(stat -c %Y "$HEARTBEAT_FILE" 2>/dev/null || stat -f %m "$HEARTBEAT_FILE" 2>/dev/null || echo "$start_time")
    else
        last_seen=$start_time
    fi
    idle=$((now - last_seen))
    if [ "$idle" -gt "$IDLE_LIMIT" ]; then
        echo "ERROR: commit hook timeout (idle limit $IDLE_LIMIT s exceeded; no heartbeat update; elapsed=$elapsed)" >&2
        exit 1
    fi

    sleep "$POLL_INTERVAL"
done
