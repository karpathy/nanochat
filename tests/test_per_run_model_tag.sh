#!/usr/bin/env bash
# Per-run model tag: the picker resumes only the newest unfinished run of the same depth,
# never a finished or other-depth run, and skips plain files; markers scope under the tag.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export NANOCHAT_BASE_DIR="$TMP"
PROFILE=runcpu
DEPTH=6
# shellcheck source=../runs/_stage_lib.sh
source "$ROOT/runs/_stage_lib.sh"

R="$NANOCHAT_BASE_DIR/runs"
mkdir -p "$R/20260101-000000-cpu-d6" "$R/20260102-000000-cpu-d6" "$R/20260103-000000-cpu-d12"
touch "$R/plainfile-cpu-d6"
touch "$R/20260102-000000-cpu-d6/.done"

# Most recent depth-6 run is 0102 and it is finished: a new id is issued. The older unfinished
# 0101 run is NOT resumed automatically.
got="$(pick_run_id "cpu-d6")"
case "$got" in
    *-cpu-d6) ;;
    *) echo "FAIL: expected a fresh cpu-d6 id, got '$got'" >&2; exit 1 ;;
esac
[ "$got" != "20260101-000000-cpu-d6" ] || { echo "FAIL: older unfinished run resumed automatically" >&2; exit 1; }

# Depth 12: its most recent run is unfinished, so it is resumed.
got="$(pick_run_id "cpu-d12")"
[ "$got" = "20260103-000000-cpu-d12" ] || { echo "FAIL: depth-12 picked '$got'" >&2; exit 1; }

# A most-recent unfinished depth-6 run (interrupted or stopped) is resumed.
mkdir -p "$R/20260110-000000-cpu-d6"
got="$(pick_run_id "cpu-d6")"
[ "$got" = "20260110-000000-cpu-d6" ] || { echo "FAIL: interrupted most-recent run not resumed, got '$got'" >&2; exit 1; }
rm -rf "$R/20260110-000000-cpu-d6"

# Keep only the newest 3 finished runs' checkpoints; unfinished runs are never pruned.
mkdir -p "$NANOCHAT_BASE_DIR/base_checkpoints/20260104-000000-cpu-d6" "$NANOCHAT_BASE_DIR/base_checkpoints/20260105-000000-cpu-d6" "$NANOCHAT_BASE_DIR/base_checkpoints/20260106-000000-cpu-d6" "$NANOCHAT_BASE_DIR/base_checkpoints/20260107-000000-cpu-d6" "$NANOCHAT_BASE_DIR/base_checkpoints/20260103-000000-cpu-d6"
for id in 20260104-000000-cpu-d6 20260105-000000-cpu-d6 20260106-000000-cpu-d6 20260107-000000-cpu-d6; do mark_run_done "$R/$id"; done
mkdir -p "$R/20260103-000000-cpu-d6" 2>/dev/null || true
prune_old_runs "$R" "cpu-d6" 3
[ ! -d "$NANOCHAT_BASE_DIR/base_checkpoints/20260104-000000-cpu-d6" ] || { echo "FAIL: oldest finished run not pruned" >&2; exit 1; }
[ -d "$NANOCHAT_BASE_DIR/base_checkpoints/20260105-000000-cpu-d6" ] || { echo "FAIL: kept run pruned" >&2; exit 1; }
[ -d "$NANOCHAT_BASE_DIR/base_checkpoints/20260103-000000-cpu-d6" ] || { echo "FAIL: unfinished run pruned" >&2; exit 1; }

# Markers are scoped under the tag when set, and unscoped (shared) when unset.
NANOCHAT_MODEL_TAG="20260101-000000-cpu-d6"
scoped="$(_stage_marker_path base_train)"
[ "$scoped" = "$TMP/base_checkpoints/20260101-000000-cpu-d6/.stage_base_train_runcpu_d6" ] \
    || { echo "FAIL: scoped marker path '$scoped'" >&2; exit 1; }
unset NANOCHAT_MODEL_TAG
shared="$(_stage_marker_path base_train)"
[ "$shared" = "$TMP/base_checkpoints/.stage_base_train_runcpu_d6" ] \
    || { echo "FAIL: shared marker path '$shared'" >&2; exit 1; }

echo "ok: per-run model tag"
