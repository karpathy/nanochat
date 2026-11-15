#!/usr/bin/env bash
# Atomic run lock: prevents two concurrent runs on the same NANOCHAT_BASE_DIR.
# Uses mkdir (atomic on POSIX) + run-ID file + heartbeat file; no flock (not on stock macOS).
# Source this file after _stage_lib.sh; it must not source _stage_lib.sh itself.
#
# Usage: source runs/_run_lock.sh
# Effect: acquires the lock or exits non-zero with a message.
# The lock is released automatically via EXIT/INT/TERM trap.
#
# Staleness: if the heartbeat file is older than _STALE_SECS (600 s) the holder
# is considered dead (covers cancelled Modal containers where kill -0 was useless).
# Reclaiming a dead holder's lock is gated by a second mkdir (<lock>.reclaim).
# Only the process holding that gate may re-check the heartbeat and replace the lock.
# A missing heartbeat file is treated as "live-unknown" (holder mid-acquire) and refused.

set -euo pipefail

_LOCK_DIR="${NANOCHAT_BASE_DIR:?must be set}/.run.lock"
_LOCK_RUN_ID_FILE="${_LOCK_DIR}/run_id"
_LOCK_HEARTBEAT_FILE="${_LOCK_DIR}/heartbeat"
_RECLAIM_DIR="${_LOCK_DIR}.reclaim"
_RECLAIM_HELD=0
# Both thresholds are overridable via env vars so tests can run fast.
_STALE_SECS="${NANOCHAT_LOCK_STALE_SECS:-600}"
_HB_REFRESH_INTERVAL="${NANOCHAT_LOCK_HB_INTERVAL:-60}"
_HB_PID=""
# Per-process instance token (hostname:pid:start-time); set by _claim_run_lock.
_LOCK_INSTANCE_TOKEN=""

# _LOCK_RUN_ID: use NANOCHAT_RUN_ID from env if set; fall back to PID-timestamp so
# ownership checks work even when no explicit run ID is passed.
_LOCK_RUN_ID="${NANOCHAT_RUN_ID:-$$-$(date +%s)}"

_read_run_id() {
    local run_id_file="$1"
    if [ -f "${run_id_file}" ]; then
        cat "${run_id_file}"
    else
        echo ""
    fi
}

_dir_mtime_epoch() {
    # Linux/GNU stat (-c %Y) first; BSD/macOS stat (-f %m) as fallback.
    # Order matches runs/modal_hook.sh (Modal runtime is Linux).
    # Redirect each loser's stdout to /dev/null — GNU stat prints a filesystem
    # dump to stdout before exiting 1, which would corrupt the captured value.
    stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo 0
}

_is_lock_stale() {
    # Returns 0 (true) when the lock looks stale.
    # Returns 1 (false) when the lock looks live or state is unreadable.
    # Heartbeat present: compare its recorded epoch to now.
    # Heartbeat missing (crashed mid-acquire): fall back to the lock dir mtime.
    # A missing heartbeat that is itself old enough is stale — prevents an
    # unrecoverable stuck lock when a crash leaves an empty lock dir.
    local last_beat now age
    if [ -f "${_LOCK_HEARTBEAT_FILE}" ]; then
        last_beat=$(cat "${_LOCK_HEARTBEAT_FILE}" 2>/dev/null) || return 1
    else
        last_beat=$(_dir_mtime_epoch "${_LOCK_DIR}") || return 1
    fi
    now=$(date +%s)
    age=$(( now - last_beat ))
    [ "${age}" -ge "${_STALE_SECS}" ]
}

_update_heartbeat() {
    date +%s > "${_LOCK_HEARTBEAT_FILE}"
}

_is_same_id_stale() {
    # Returns 0 (true) when the lock looks stale for same-run-ID reclaim.
    # Threshold = 2 * _HB_REFRESH_INTERVAL (a live holder refreshes within one interval).
    # Falls back to dir mtime when heartbeat is missing (crashed mid-acquire).
    local last_beat now age threshold
    if [ -f "${_LOCK_HEARTBEAT_FILE}" ]; then
        last_beat=$(cat "${_LOCK_HEARTBEAT_FILE}" 2>/dev/null) || return 1
    else
        last_beat=$(_dir_mtime_epoch "${_LOCK_DIR}") || return 1
    fi
    now=$(date +%s)
    age=$(( now - last_beat ))
    threshold=$(( _HB_REFRESH_INTERVAL * 2 ))
    [ "${age}" -ge "${threshold}" ]
}

_start_heartbeat_refresher() {
    # Runs in the background; writes the heartbeat every _HB_REFRESH_INTERVAL seconds.
    # Checks that the parent process is still alive before each write; exits when the
    # parent is gone (covers kill -9 where EXIT traps do not fire — BLOCKER 1 fix).
    # Redirect stdout/stderr to /dev/null so the background subshell and its sleep
    # children do not hold inherited pipe fds open (which would delay communicate()
    # EOF when tests capture output via subprocess.run(capture_output=True)).
    local _parent_pid=$$
    (
        exec >/dev/null 2>/dev/null
        while true; do
            sleep "${_HB_REFRESH_INTERVAL}"
            kill -0 "${_parent_pid}" 2>/dev/null || exit 0
            _update_heartbeat 2>/dev/null || true
        done
    ) &
    _HB_PID=$!
}

_drop_reclaim_gate() {
    if [ "${_RECLAIM_HELD}" = "1" ]; then
        rmdir "${_RECLAIM_DIR}" 2>/dev/null || true
        _RECLAIM_HELD=0
    fi
}

_release_run_lock() {
    # Kill the background refresher first so it cannot write after we remove the lock dir.
    # wait ensures the zombie is reaped and bash can exit without stalling on a background child.
    if [ -n "${_HB_PID:-}" ]; then
        kill "${_HB_PID}" 2>/dev/null || true
        wait "${_HB_PID}" 2>/dev/null || true
    fi
    _drop_reclaim_gate
    # Ownership check via instance token (hostname:pid:start-time), which is unique per
    # process instance even when two restarts share the same NANOCHAT_RUN_ID.
    # Never delete a lock claimed by a different process instance (BLOCKER 2 fix).
    local stored_token
    stored_token=$(cat "${_LOCK_DIR}/instance_token" 2>/dev/null) || stored_token=""
    if [ -n "${_LOCK_INSTANCE_TOKEN:-}" ] && [ "${stored_token}" = "${_LOCK_INSTANCE_TOKEN}" ]; then
        rm -rf "${_LOCK_DIR}"
    fi
}

# Caller must own the lock directory. Records run ID + per-process instance token +
# heartbeat, starts the background refresher, and installs traps.
_claim_run_lock() {
    echo "${_LOCK_RUN_ID}" > "${_LOCK_RUN_ID_FILE}"
    # Instance token is unique per process even when two restarts share a run ID.
    _LOCK_INSTANCE_TOKEN="${HOSTNAME:-unknown}:$$:$(date +%s)"
    echo "${_LOCK_INSTANCE_TOKEN}" > "${_LOCK_DIR}/instance_token"
    _update_heartbeat
    _start_heartbeat_refresher
    # NOTE: these traps overwrite any trap the caller set for EXIT/INT/TERM.
    # Callers must not install competing traps after sourcing _run_lock.sh.
    trap '_release_run_lock' EXIT
    trap '_release_run_lock; exit 130' INT
    trap '_release_run_lock; exit 143' TERM
}

_acquire_run_lock() {
    if mkdir "${_LOCK_DIR}" 2>/dev/null; then
        _claim_run_lock
        return 0
    fi

    # Lock directory exists: is the holder still active?
    # Volume-relative lock path for operator remove commands (e.g. cpu/.run.lock or gpu/.run.lock).
    local _vol_lock_path
    _vol_lock_path="$(basename "${NANOCHAT_BASE_DIR}")/.run.lock"

    # Same-run-ID reclaim: a preempted container that restarts can reclaim its own lock,
    # but only after the previous instance looks dead. A live holder refreshes the heartbeat
    # every _HB_REFRESH_INTERVAL s; after kill -9 the BLOCKER 1 fix stops the refresher,
    # so the heartbeat goes stale within one interval. Threshold = 2 * _HB_REFRESH_INTERVAL
    # gives a safe margin: live → always fresh; killed → stale after ~2 intervals (120 s default).
    # This prevents a second process with the same NANOCHAT_RUN_ID from stealing a live lock.
    local held_run_id
    held_run_id="$(_read_run_id "${_LOCK_RUN_ID_FILE}")"
    if [ "${held_run_id:-}" = "${_LOCK_RUN_ID}" ]; then
        if ! _is_same_id_stale; then
            echo "ERROR: same run ID ${_LOCK_RUN_ID} is already active on ${NANOCHAT_BASE_DIR} (heartbeat is fresh — previous instance still running)." >&2
            exit 1
        fi
        # Previous instance looks dead: reclaim via the standard gate.
        if ! mkdir "${_RECLAIM_DIR}" 2>/dev/null; then
            echo "ERROR: another run is reclaiming the lock on ${NANOCHAT_BASE_DIR}." >&2
            echo "  If no run is active, remove it with: rm -rf \"${_RECLAIM_DIR}\"" >&2
            exit 1
        fi
        _RECLAIM_HELD=1
        trap '_release_run_lock' EXIT
        trap '_release_run_lock; exit 130' INT
        trap '_release_run_lock; exit 143' TERM
        local own_tmp="${_LOCK_DIR}.stale.$$"
        if mv "${_LOCK_DIR}" "${own_tmp}" 2>/dev/null; then
            rm -rf "${own_tmp}"
        fi
        if mkdir "${_LOCK_DIR}" 2>/dev/null; then
            _claim_run_lock
            _drop_reclaim_gate
            return 0
        fi
        echo "ERROR: could not reclaim own lock on ${NANOCHAT_BASE_DIR}." >&2
        exit 1
    fi

    if ! _is_lock_stale; then
        echo "ERROR: another run is already active on ${NANOCHAT_BASE_DIR} (run ${held_run_id:-unknown})." >&2
        echo "  If that run has ended, remove the stale lock with:" >&2
        echo "    modal volume rm -r nanochat-modal ${_vol_lock_path}" >&2
        exit 1
    fi

    # Holder looks stale (heartbeat > ${_STALE_SECS} s old).
    # Take the reclaim gate so only one process does this at a time.
    if ! mkdir "${_RECLAIM_DIR}" 2>/dev/null; then
        echo "ERROR: another run is reclaiming the lock on ${NANOCHAT_BASE_DIR}." >&2
        echo "  If no run is active, remove it with: rm -rf \"${_RECLAIM_DIR}\"" >&2
        exit 1
    fi
    _RECLAIM_HELD=1
    # Install EXIT/INT/TERM traps now so any failure below still releases the gate.
    trap '_release_run_lock' EXIT
    trap '_release_run_lock; exit 130' INT
    trap '_release_run_lock; exit 143' TERM

    # Re-check under the gate: the heartbeat may have been refreshed since we first read it.
    # _is_lock_stale falls back to dir mtime when no heartbeat, so a fresh lock dir
    # (new holder mid-acquire) will correctly appear non-stale without a separate guard.
    if ! _is_lock_stale; then
        echo "ERROR: another run is already active on ${NANOCHAT_BASE_DIR} (heartbeat is fresh)." >&2
        exit 1
    fi

    local stale_tmp="${_LOCK_DIR}.stale.$$"
    if mv "${_LOCK_DIR}" "${stale_tmp}" 2>/dev/null; then
        rm -rf "${stale_tmp}"
    fi
    if mkdir "${_LOCK_DIR}" 2>/dev/null; then
        _claim_run_lock
        _drop_reclaim_gate
        return 0
    fi

    local winner_run_id
    winner_run_id="$(_read_run_id "${_LOCK_RUN_ID_FILE}")"
    echo "ERROR: another run is already active on ${NANOCHAT_BASE_DIR} (run ${winner_run_id:-?})." >&2
    exit 1
}

_acquire_run_lock
