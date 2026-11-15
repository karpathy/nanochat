#!/bin/bash

# Peek at Modal runs from the terminal. Read-only: it lists, reads logs and downloads small
# files (config, report, manifest). It never stops or changes anything (stop an app by hand
# with `modal app stop <app-id>`).
#
# Runs: each CPU run has a run id (UTC timestamp + mode). Its full log is kept locally in
# ~/.cache/nanochat-modal/runs/<run-id>/run.log; its config, report and manifest live on the
# Volume under runs/<run-id>/. Checkpoints are archived in runs/<run-id>/artifacts on the Volume.
#
# If you leave out an app or container id, the script picks the current run: the newest
# nanochat app that is running (preferring one with tasks), else the newest nanochat app.
#
# Usage:
#   bash runs/modal_peek.sh status                    running apps + active containers
#   bash runs/modal_peek.sh apps [N]                  last N Modal apps, incl. stopped (default 10)
#   bash runs/modal_peek.sh past                      archived runs: local folders and Volume runs
#   bash runs/modal_peek.sh report [run-id]           run record of a run: config, resources, status (default: latest)
#   bash runs/modal_peek.sh compare                   one table of every run on the Volume
#   bash runs/modal_peek.sh pull [run-id]             copy a run's config, report, manifest locally
#   bash runs/modal_peek.sh info [app-id]             lifecycle, functions, state of an app
#   bash runs/modal_peek.sh logs [app-id] [N] [-f]    last N log lines of an app (default 100); -f streams
#   bash runs/modal_peek.sh container [ta-id] [N] [-f]  last N log lines of a container; -f streams
#   bash runs/modal_peek.sh cost [PERIOD]             billing summary, e.g. "this month" or "last month"

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

if ! command -v modal &>/dev/null; then
    echo "ERROR: 'modal' not found on PATH. Install with: uv tool install modal" >&2
    exit 1
fi

# Source Modal config (NANOCHAT_MODAL_VOLUME)
SCRIPTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPTDIR/_modal_config.sh"

VOLUME="$NANOCHAT_MODAL_VOLUME"
LOCAL_RUNS="${NANOCHAT_MODAL_ARCHIVE:-$HOME/.cache/nanochat-modal/runs}"

usage() {
    sed -n 3,24p "$0" | sed 's/^# \{0,1\}//'
}

# Print the run ids on the Volume, oldest first. Prints nothing if there are none yet.
volume_run_ids() {
    modal volume ls "$VOLUME" runs --json 2>/dev/null | python3 -c '
import json, sys
try:
    entries = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
for e in sorted(entries, key=lambda e: e["filename"]):
    if e.get("type") == "dir":
        print(e["filename"].rsplit("/", 1)[-1])
' || true
}

# Print the id of the newest run: the latest on the Volume, or the latest local folder.
latest_run() {
    local ids
    ids=$(volume_run_ids | tail -n 1)
    if [ -z "$ids" ] && [ -d "$LOCAL_RUNS" ]; then
        ids=$(ls -1 "$LOCAL_RUNS" 2>/dev/null | sort | tail -n 1)
    fi
    if [ -z "$ids" ]; then
        echo "No archived runs yet. A CPU run writes one when it starts." >&2
        return 1
    fi
    echo "$ids"
}

# Fetch one file of a run from the Volume to stdout.
volume_file() {
    modal volume get "$VOLUME" "runs/$1/$2" -
}

# Print "<app-id>\t<state>\t<created_at>" for the current nanochat run. Exit 1 if none.
pick_app() {
    modal app list --json | python3 -c '
import json, sys
apps = [a for a in json.load(sys.stdin) if a["description"].startswith("nanochat")]
running = [a for a in apps if a["state"] == "ephemeral"]
busy = [a for a in running if str(a["tasks"]) != "0"]
pool = busy or running or apps
if not pool:
    sys.exit(1)
best = max(pool, key=lambda a: a["created_at"])
print(best["app_id"], best["state"], best["created_at"], sep="\t")
'
}

# Print the id of the first container belonging to app $1. Exit 1 if none.
pick_container() {
    modal container list --json | python3 -c '
import json, sys
app = sys.argv[1]
found = [c for c in json.load(sys.stdin) if c["app_id"] == app]
if not found:
    sys.exit(1)
print(found[0]["container_id"])
' "$1"
}

# Resolve the app id: use $1 if given, else the current run (and say which one).
resolve_app() {
    if [ -n "${1:-}" ]; then
        echo "$1"
        return
    fi
    local line
    line=$(pick_app) || { echo "No nanochat app found. Pass an app id, or run: bash runs/modal_peek.sh apps" >&2; exit 1; }
    local app state created
    IFS=$'\t' read -r app state created <<<"$line"
    echo "Using app $app (state: $state, created $created). Pass an app id to choose another." >&2
    echo "$app"
}

cmd="${1:-status}"
shift || true

# -f anywhere in the args means "follow": stream logs live instead of a snapshot.
follow=0
rest=()
for arg in "$@"; do
    if [ "$arg" = "-f" ]; then follow=1; else rest+=("$arg"); fi
done
set -- ${rest[@]+"${rest[@]}"}

case "$cmd" in
    finish-download)
        # Download artifacts after run stops: called from runmodal.sh finish
        RUN_ID="${1:-}"
        if [ -z "$RUN_ID" ]; then
            echo "ERROR: usage: bash runs/modal_peek.sh finish-download <run-id>" >&2
            exit 1
        fi
        ARCHIVE_DIR="$LOCAL_RUNS/$RUN_ID"
        APP_ID_FILE="$ARCHIVE_DIR/app.id"
        if [ ! -f "$APP_ID_FILE" ]; then
            echo "ERROR: $APP_ID_FILE not found" >&2
            exit 1
        fi
        read -r APP_ID MODE < "$APP_ID_FILE"

        # Check app state: must be stopped
        APP_STATE=$(modal app list --json | python3 -c "
import json, sys
try:
    for a in json.load(sys.stdin):
        if a.get('app_id') == '$APP_ID':
            print(a.get('state', 'unknown'))
            sys.exit(0)
    print('not_found')
except:
    print('error')
" 2>/dev/null || echo "error")

        if [ "$APP_STATE" = "error" ] || [ "$APP_STATE" = "not_found" ]; then
            echo "ERROR: could not find app $APP_ID in modal app list" >&2
            exit 1
        fi
        if [ "$APP_STATE" != "stopped" ]; then
            echo "ERROR: app $APP_ID is not stopped (state: $APP_STATE)" >&2
            echo "Check logs: bash runs/modal_peek.sh logs $APP_ID" >&2
            exit 1
        fi

        # Check if chatsft_checkpoints exists on Volume (mode-prefixed path)
        if ! modal volume ls "$VOLUME" "/$MODE/chatsft_checkpoints" 2>/dev/null | grep -q .; then
            # Fallback: check legacy root path (runs launched before the mode-prefix migration)
            if modal volume ls "$VOLUME" "chatsft_checkpoints" 2>/dev/null | grep -q .; then
                echo "WARNING: found chatsft_checkpoints at Volume root, not at /$MODE/ — this run was launched before the mode-prefix migration." >&2
                echo "  Move the artifacts: modal volume cp $VOLUME /chatsft_checkpoints $VOLUME /$MODE/chatsft_checkpoints" >&2
                echo "  Then re-run: bash runs/runmodal.sh finish $RUN_ID" >&2
            else
                echo "ERROR: /$MODE/chatsft_checkpoints not found on Volume $VOLUME" >&2
            fi
            exit 1
        fi

        # Download artifacts (uv run so modal is on the Python path regardless of active venv)
        echo "Downloading artifacts from Volume $VOLUME, mode $MODE..."
        if ! (cd "$(dirname "$SCRIPTDIR")" && NANOCHAT_MODAL_VOLUME="$VOLUME" \
              uv run --frozen python -c "
import sys, os
from runs.modal_run import _download_artifacts
_download_artifacts('$MODE')
"); then
            echo "ERROR: download failed" >&2
            exit 1
        fi

        # Remove marker
        rm -f "$ARCHIVE_DIR/PENDING_DOWNLOAD"
        echo "Download complete. Artifacts in: ~/.cache/nanochat-modal/$MODE"
        ;;

    status)
        echo "== Running apps =="
        modal app list
        echo
        echo "== Active containers =="
        modal container list
        ;;
    apps)
        modal app list --limit "${1:-10}"
        ;;
    past)
        echo "== Local archive ($LOCAL_RUNS) =="
        if [ -d "$LOCAL_RUNS" ] && [ -n "$(ls -A "$LOCAL_RUNS" 2>/dev/null)" ]; then
            for d in $(ls -1 "$LOCAL_RUNS" | sort); do
                rev=$(head -n 1 "$LOCAL_RUNS/$d/run.log" 2>/dev/null | sed -n 's/.*(code \([^)]*\)).*/\1/p')
                echo "  $d  code=${rev:-unknown}"
            done
        else
            echo "  (none)"
        fi
        echo
        echo "== On the Volume ($VOLUME/runs) =="
        ids=$(volume_run_ids)
        if [ -n "$ids" ]; then
            echo "$ids" | sed 's/^/  /'
        else
            echo "  (none)"
        fi
        ;;
    report)
        run="${1:-$(latest_run)}"
        volume_file "$run" run.json || { echo "No run record for $run on the Volume." >&2; exit 1; }
        ;;
    compare)
        ids=$(volume_run_ids)
        if [ -z "$ids" ]; then
            echo "No runs on the Volume yet." >&2
            exit 1
        fi
        # Copy each run's run.json + events.jsonl into a temp root, then use the one recorder
        # table (nanochat/run_record.py) so local and Modal runs read the same way.
        tmp_root=$(mktemp -d)
        trap 'rm -rf "$tmp_root"' EXIT
        for run in $ids; do
            mkdir -p "$tmp_root/$run"
            for f in run.json events.jsonl; do
                volume_file "$run" "$f" > "$tmp_root/$run/$f" 2>/dev/null || rm -f "$tmp_root/$run/$f"
            done
        done
        (cd "$(dirname "$0")/.." && python3 -m nanochat.run_record compare "$tmp_root")
        ;;
    pull)
        run="${1:-$(latest_run)}"
        dest="$LOCAL_RUNS/$run"
        mkdir -p "$dest"
        for f in run.json events.jsonl manifest.json; do
            volume_file "$run" "$f" > "$dest/$f" 2>/dev/null || rm -f "$dest/$f"
        done
        echo "Pulled $run into $dest (run record, events, manifest). Checkpoints stay on the Volume."
        ls -1 "$dest"
        ;;
    info)
        app=$(resolve_app "${1:-}")
        modal app info "$app"
        ;;
    logs)
        # Args: [app-id] [N]. A leading ap- id picks the app; otherwise the arg is N.
        if [[ "${1:-}" == ap-* ]]; then
            app=$(resolve_app "$1"); n="${2:-100}"
        else
            app=$(resolve_app ""); n="${1:-100}"
        fi
        if [ "$follow" = 1 ]; then
            modal app logs "$app" -f
        else
            modal app logs "$app" | tail -n "$n"
        fi
        ;;
    container)
        # Args: [ta-id] [N]. A leading ta- id picks the container; otherwise the current run's.
        if [[ "${1:-}" == ta-* ]]; then
            ctr="$1"; n="${2:-100}"
        else
            app=$(resolve_app "")
            ctr=$(pick_container "$app") || { echo "No active container for app $app (the run may have finished)." >&2; exit 1; }
            echo "Using container $ctr." >&2
            n="${1:-100}"
        fi
        if [ "$follow" = 1 ]; then
            modal container logs "$ctr" -f
        else
            modal container logs "$ctr" | tail -n "$n"
        fi
        ;;
    cost)
        modal billing summary --for "${1:-this month}"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "unknown command: $cmd" >&2
        usage >&2
        exit 2
        ;;
esac
