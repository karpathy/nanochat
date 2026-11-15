"""
Author: Mani Sarkar
Created: 2026-11-14
Scope: runs/_run_lock.sh — atomic mkdir-based run lock.
       Cases: acquire fresh, second acquire fails, stale PID reclaimed, EXIT trap releases.
"""
import os
import subprocess

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
        check=False, capture_output=True, text=True, timeout=timeout,
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

    def test_missing_pid_file_treated_as_live_unknown(self, tmp_path):
        """
        Scenario: Lock dir exists but pid file is absent; treated as live-unknown to close the
        mid-acquire window (holder may have done mkdir but not yet written PID).
        Slice: _acquire_run_lock edge-case: empty lock dir
        ### Given: .run.lock/ exists but contains no pid file
        ### When: acquire runs
        ### Then: exits 1 with a message indicating the PID is not yet recorded
        """
        lock_dir = tmp_path / ".run.lock"
        lock_dir.mkdir()
        # no pid file written

        r = _run(_run_lock_script(str(tmp_path)))
        assert r.returncode == 1
        assert "no PID recorded" in r.stderr


class TestRunLockReclaimAndSafety:
    """
    Scenario: Reclaim is gated by a second mkdir; an empty base dir is refused.
    Slice: runs/_run_lock.sh (review blockers S2, B5)
    """

    def test_empty_base_dir_is_refused(self, tmp_path):
        """
        Scenario: NANOCHAT_BASE_DIR is set but empty; the lock must not resolve to /.run.lock.
        Slice: S2 guard
        ### Given: NANOCHAT_BASE_DIR=""
        ### When: _run_lock.sh is sourced
        ### Then: exits non-zero with 'must be set' and nothing is created under /
        """
        r = _run(_run_lock_script(""))
        assert r.returncode != 0
        assert "must be set" in r.stderr
        assert "ACQUIRED" not in r.stdout

    def test_reclaim_gate_held_by_another_run_blocks_reclaim(self, tmp_path):
        """
        Scenario: Stale lock, but another process holds the reclaim gate.
        Slice: B5 single-claimant gate
        ### Given: .run.lock/pid is dead and .run.lock.reclaim exists
        ### When: _run_lock.sh is sourced
        ### Then: exits 1 with 'reclaiming' and the stale lock dir is left in place
        """
        lock_dir = tmp_path / ".run.lock"
        lock_dir.mkdir()
        (lock_dir / "pid").write_text("99999")
        (tmp_path / ".run.lock.reclaim").mkdir()

        r = _run(_run_lock_script(str(tmp_path)))
        assert r.returncode == 1
        assert "reclaiming" in r.stderr
        assert (lock_dir / "pid").exists()

    def test_reclaim_gate_is_released_after_successful_reclaim(self, tmp_path):
        """
        Scenario: Stale lock is reclaimed successfully.
        Slice: B5 gate cleanup
        ### Given: .run.lock/pid is dead, no gate present
        ### When: _run_lock.sh is sourced and exits
        ### Then: exits 0 and neither .run.lock nor .run.lock.reclaim remains
        """
        lock_dir = tmp_path / ".run.lock"
        lock_dir.mkdir()
        (lock_dir / "pid").write_text("99999")

        r = _run(_run_lock_script(str(tmp_path)))
        assert r.returncode == 0, r.stderr
        assert not (tmp_path / ".run.lock").exists()
        assert not (tmp_path / ".run.lock.reclaim").exists()

    def test_live_holder_is_never_reclaimed_and_leaves_no_gate(self, tmp_path):
        """
        Scenario: Lock is held by a live process; no reclaim is attempted.
        Slice: B5 — live holder must never be taken over
        ### Given: .run.lock/pid holds a live PID (this test process)
        ### When: _run_lock.sh is sourced
        ### Then: exits 1, the live lock is untouched, and no .reclaim dir is created
        """
        lock_dir = tmp_path / ".run.lock"
        lock_dir.mkdir()
        (lock_dir / "pid").write_text(str(os.getpid()))

        r = _run(_run_lock_script(str(tmp_path)))
        assert r.returncode == 1
        assert (lock_dir / "pid").read_text() == str(os.getpid())
        assert not (tmp_path / ".run.lock.reclaim").exists()

    def test_lock_not_released_when_pid_overwritten(self, tmp_path):
        """
        Scenario: Lock PID is overwritten mid-run with a different live PID; EXIT trap
                  must NOT delete the lock (it now belongs to a different process).
        Slice: _release_run_lock ownership check
        ### Given: _run_lock.sh acquires the lock (writes its own PID), then the PID
                   file is overwritten with a different live PID before the script exits
        ### When: the script exits normally (EXIT trap fires)
        ### Then: .run.lock dir still exists — the lock belongs to the other PID
        """
        # This test falsifies the ownership guard: deleting the
        # `if [ "$(_read_pid_file ...)" = "$$" ]` check in _release_run_lock causes
        # the lock to be removed unconditionally, making this assertion fail.
        other_pid = os.getpid()  # Python's PID — definitely live, definitely != bash's $$
        script = f"""#!/usr/bin/env bash
set -euo pipefail
export NANOCHAT_BASE_DIR="{tmp_path}"
source "{RUN_LOCK}"
echo "{other_pid}" > "{tmp_path}/.run.lock/pid"
# EXIT trap fires here: sees PID {other_pid} != $$, must NOT rm the lock
"""
        r = _run(script)
        assert r.returncode == 0, r.stderr
        lock_dir = tmp_path / ".run.lock"
        assert lock_dir.exists(), "lock was released despite PID mismatch — ownership check missing"
