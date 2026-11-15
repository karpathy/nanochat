#!/usr/bin/env bash
# Atomic run lock: prevents two concurrent runs on the same NANOCHAT_BASE_DIR.
# Uses mkdir (atomic on POSIX) + PID file; no flock (not on stock macOS).
# Source this file after _stage_lib.sh; it must not source _stage_lib.sh itself.
#
# Usage: source runs/_run_lock.sh
# Effect: acquires the lock or exits non-zero with a message.
# The lock is released automatically via EXIT/INT/TERM trap.
#
# Reclaiming a dead holder's lock is gated by a second mkdir (<lock>.reclaim).
# Only the process holding that gate may re-check the PID and replace the lock.
# A missing PID file is treated as "live-unknown" (holder mid-acquire) and refused.

set -euo pipefail

_LOCK_DIR="${NANOCHAT_BASE_DIR:?must be set}/.run.lock"
_LOCK_PID_FILE="${_LOCK_DIR}/pid"
_RECLAIM_DIR="${_LOCK_DIR}.reclaim"
_RECLAIM_HELD=0

_read_pid_file() {
    local pid_file="$1"
    if [ -f "${pid_file}" ]; then
        cat "${pid_file}"
    else
        echo ""
    fi
}

_drop_reclaim_gate() {
    if [ "${_RECLAIM_HELD}" = "1" ]; then
        rmdir "${_RECLAIM_DIR}" 2>/dev/null || true
        _RECLAIM_HELD=0
    fi
}

_release_run_lock() {
    _drop_reclaim_gate
    # Ownership check: never delete a lock that another process has taken over.
    if [ "$(_read_pid_file "${_LOCK_PID_FILE}")" = "$$" ]; then
        rm -rf "${_LOCK_DIR}"
    fi
}

# Caller must own the lock directory. Records our PID and installs the traps.
_claim_run_lock() {
    echo $$ > "${_LOCK_PID_FILE}"
    trap '_release_run_lock' EXIT
    trap '_release_run_lock; exit 130' INT
    trap '_release_run_lock; exit 143' TERM
}

_acquire_run_lock() {
    if mkdir "${_LOCK_DIR}" 2>/dev/null; then
        _claim_run_lock
        return 0
    fi

    # Lock directory exists: is the holder still alive?
    # Treat a missing or empty PID file as "live-unknown" (holder mid-acquire, not yet written PID).
    local held_pid
    held_pid="$(_read_pid_file "${_LOCK_PID_FILE}")"
    if [ -z "${held_pid}" ]; then
        echo "ERROR: lock dir exists but no PID recorded (holder may be mid-acquire). Try again in a moment." >&2
        echo "  If no run is active, remove it with: rm -rf \"${_LOCK_DIR}\"" >&2
        exit 1
    fi
    if kill -0 "${held_pid}" 2>/dev/null; then
        echo "ERROR: another run is already active on ${NANOCHAT_BASE_DIR} (PID ${held_pid})." >&2
        echo "  If that run has ended, remove the stale lock with:" >&2
        echo "    rm -rf \"${_LOCK_DIR}\"" >&2
        exit 1
    fi

    # Holder looks dead. Take the reclaim gate so only one process does this at a time.
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

    # Re-check under the gate: the PID may have been rewritten since we first read it.
    # Also re-apply the live-unknown guard: a new holder may have written its PID while we
    # were taking the reclaim gate.
    held_pid="$(_read_pid_file "${_LOCK_PID_FILE}")"
    if [ -z "${held_pid}" ]; then
        echo "ERROR: lock dir exists but no PID recorded after taking reclaim gate. Try again in a moment." >&2
        exit 1
    fi
    if kill -0 "${held_pid}" 2>/dev/null; then
        echo "ERROR: another run is already active on ${NANOCHAT_BASE_DIR} (PID ${held_pid})." >&2
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

    local winner_pid
    winner_pid="$(_read_pid_file "${_LOCK_PID_FILE}")"
    echo "ERROR: another run is already active on ${NANOCHAT_BASE_DIR} (PID ${winner_pid:-?})." >&2
    exit 1
}

_acquire_run_lock
