#!/usr/bin/env bash
# Atomic run lock: prevents two concurrent runs on the same NANOCHAT_BASE_DIR.
# Uses mkdir (atomic on POSIX) + PID file; no flock (not on stock macOS).
# Source this file after _stage_lib.sh; it must not source _stage_lib.sh itself.
#
# Usage: source runs/_run_lock.sh
# Effect: acquires the lock or exits non-zero with a message.
# The lock is released automatically via EXIT/INT/TERM trap.

set -euo pipefail

_LOCK_DIR="${NANOCHAT_BASE_DIR}/.run.lock"
_LOCK_PID_FILE="${_LOCK_DIR}/pid"

_release_run_lock() {
    rm -rf "${_LOCK_DIR}"
}

_read_pid_file() {
    local pid_file="$1"
    if [ -f "${pid_file}" ]; then
        cat "${pid_file}"
    else
        echo ""
    fi
}

_acquire_run_lock() {
    local stale_tmp
    stale_tmp="${NANOCHAT_BASE_DIR}/.run.lock.stale.$$"

    if mkdir "${_LOCK_DIR}" 2>/dev/null; then
        echo $$ > "${_LOCK_PID_FILE}"
        trap '_release_run_lock' EXIT
        trap '_release_run_lock; exit 130' INT
        trap '_release_run_lock; exit 143' TERM
        return 0
    fi

    # Lock directory exists — check if the holder is still alive.
    local held_pid
    held_pid="$(_read_pid_file "${_LOCK_PID_FILE}")"
    if [ -n "${held_pid}" ] && kill -0 "${held_pid}" 2>/dev/null; then
        echo "ERROR: another run is already active on ${NANOCHAT_BASE_DIR} (PID ${held_pid})." >&2
        echo "  If that run has ended, remove the stale lock with:" >&2
        echo "    rm -rf \"${_LOCK_DIR}\"" >&2
        exit 1
    fi

    # Holder is dead — reclaim atomically: rename the stale dir before anyone else does.
    if mv "${_LOCK_DIR}" "${stale_tmp}" 2>/dev/null; then
        rm -rf "${stale_tmp}"
        if mkdir "${_LOCK_DIR}" 2>/dev/null; then
            echo $$ > "${_LOCK_PID_FILE}"
            trap '_release_run_lock' EXIT
            trap '_release_run_lock; exit 130' INT
            trap '_release_run_lock; exit 143' TERM
            return 0
        fi
    fi

    # Race: another process won the reclaim.
    local winner_pid
    winner_pid="$(_read_pid_file "${_LOCK_PID_FILE}")"
    echo "ERROR: another run is already active on ${NANOCHAT_BASE_DIR} (PID ${winner_pid:-?})." >&2
    exit 1
}

_acquire_run_lock
