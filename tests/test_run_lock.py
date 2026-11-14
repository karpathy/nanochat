"""
Author: Mani Sarkar
Created: 2026-11-14
Scope: runs/_run_lock.sh — atomic mkdir-based run lock.
       Cases: acquire fresh, second acquire fails, stale PID reclaimed, EXIT trap releases.
"""
import os
import subprocess
import sys
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_LOCK = os.path.join(REPO_ROOT, "runs", "_run_lock.sh")


def _run_lock_script(base_dir: str, extra_env: dict | None = None) -> str:
    """Return a one-shot bash script that sources _run_lock.sh then exits cleanly."""
    env_lines = "\n".join(f'export {k}="{v}"' for k, v in (extra_env or {}).items())
    return f"""#!/usr/bin/env bash
set -euo pipefail
export NANOCHAT_BASE_DIR="{base_dir}"
{env_lines}
source "{RUN_LOCK}"
echo "ACQUIRED"
"""


def _run(script: str, timeout: int = 5) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True, text=True, timeout=timeout,
    )


class TestRunLock:
    """
    Scenario: Atomic run lock prevents concurrent executions on the same NANOCHAT_BASE_DIR.
    Slice: runs/_run_lock.sh
    """

    def test_fresh_acquire_succeeds(self, tmp_path):
        """
        Scenario: No lock exists; first acquire should succeed.
        Slice: _acquire_run_lock happy path
        ### Given: an empty base dir
        ### When: _run_lock.sh is sourced
        ### Then: exits 0 and prints ACQUIRED; lock dir is gone after the script ends (EXIT trap)
        """
        r = _run(_run_lock_script(str(tmp_path)))
        assert r.returncode == 0, r.stderr
        assert "ACQUIRED" in r.stdout
        # EXIT trap must have cleaned up
        assert not os.path.exists(tmp_path / ".run.lock")

    def test_second_acquire_fails(self, tmp_path):
        """
        Scenario: Lock is held by an alive process; second acquire must exit non-zero.
        Slice: _acquire_run_lock live-PID rejection
        ### Given: .run.lock/pid contains this process's PID (simulated by writing our own PID)
        ### When: a second sourcing attempt runs
        ### Then: exits 1 and prints ERROR
        """
        lock_dir = tmp_path / ".run.lock"
        lock_dir.mkdir()
        (lock_dir / "pid").write_text(str(os.getpid()))

        r = _run(_run_lock_script(str(tmp_path)))
        assert r.returncode == 1
        assert "ERROR" in r.stderr

    def test_stale_pid_is_reclaimed(self, tmp_path):
        """
        Scenario: Lock exists but the PID it records is dead; acquire should reclaim it.
        Slice: _acquire_run_lock stale-reclaim branch
        ### Given: .run.lock/pid contains a PID that is not alive (use 99999 or similar)
        ### When: _run_lock.sh is sourced
        ### Then: exits 0 and prints ACQUIRED
        """
        lock_dir = tmp_path / ".run.lock"
        lock_dir.mkdir()
        (lock_dir / "pid").write_text("99999")  # almost certainly dead

        r = _run(_run_lock_script(str(tmp_path)))
        assert r.returncode == 0, r.stderr
        assert "ACQUIRED" in r.stdout

    def test_lock_released_after_exit(self, tmp_path):
        """
        Scenario: After a script sourcing _run_lock.sh exits, the lock directory is gone.
        Slice: EXIT trap in _acquire_run_lock
        ### Given: a successful acquisition
        ### When: the subprocess exits
        ### Then: .run.lock directory no longer exists
        """
        lock_dir = tmp_path / ".run.lock"
        r = _run(_run_lock_script(str(tmp_path)))
        assert r.returncode == 0
        assert not lock_dir.exists()

    def test_missing_pid_file_treated_as_stale(self, tmp_path):
        """
        Scenario: Lock dir exists but pid file is absent (crash with no pid write); reclaim it.
        Slice: _acquire_run_lock edge-case: empty lock dir
        ### Given: .run.lock/ exists but is empty
        ### When: acquire runs
        ### Then: exits 0 (reclaims the orphaned dir)
        """
        lock_dir = tmp_path / ".run.lock"
        lock_dir.mkdir()
        # no pid file written

        r = _run(_run_lock_script(str(tmp_path)))
        assert r.returncode == 0, r.stderr
        assert "ACQUIRED" in r.stdout
