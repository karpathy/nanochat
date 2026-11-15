"""
Author: Mani Sarkar
Created: 2026-11-14
Scope: runs/_run_lock.sh — atomic mkdir-based run lock.
       Cases: acquire fresh, second acquire fails, stale heartbeat reclaimed,
       live-unknown guard, EXIT trap releases, heartbeat staleness (B3).
"""
import os
import subprocess
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_LOCK = os.path.join(REPO_ROOT, "runs", "_run_lock.sh")

_STALE_SECS = 600  # must match _run_lock.sh


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


def _write_fresh_lock(lock_dir, run_id: str = "some-run") -> None:
    """Create a lock directory with a fresh (current) heartbeat and run_id."""
    lock_dir.mkdir(exist_ok=True)
    (lock_dir / "run_id").write_text(run_id)
    (lock_dir / "heartbeat").write_text(str(int(time.time())))


def _write_stale_lock(lock_dir, run_id: str = "old-run") -> None:
    """Create a lock directory with a stale heartbeat (> 600 s old)."""
    lock_dir.mkdir(exist_ok=True)
    (lock_dir / "run_id").write_text(run_id)
    stale_ts = str(int(time.time()) - _STALE_SECS - 60)
    (lock_dir / "heartbeat").write_text(stale_ts)


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

    def test_second_acquire_fails_when_heartbeat_is_fresh(self, tmp_path):
        """
        Scenario: Lock is held with a fresh heartbeat; second acquire must exit non-zero.
        Slice: _acquire_run_lock live-heartbeat rejection
        ### Given: .run.lock/ with a current-timestamp heartbeat
        ### When: a second sourcing attempt runs
        ### Then: exits 1 and prints ERROR
        """
        lock_dir = tmp_path / ".run.lock"
        _write_fresh_lock(lock_dir, run_id="live-run")

        r = _run(_run_lock_script(str(tmp_path)))
        assert r.returncode == 1
        assert "ERROR" in r.stderr

    def test_stale_heartbeat_is_reclaimed(self, tmp_path):
        """
        Scenario: Lock exists but heartbeat is > 600 s old; acquire should reclaim it.
        Slice: _acquire_run_lock stale-reclaim branch
        ### Given: .run.lock/ with a heartbeat 660 s in the past
        ### When: _run_lock.sh is sourced
        ### Then: exits 0 and prints ACQUIRED
        """
        lock_dir = tmp_path / ".run.lock"
        _write_stale_lock(lock_dir)

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

    def test_missing_heartbeat_treated_as_live_unknown(self, tmp_path):
        """
        Scenario: Lock dir exists but heartbeat file is absent; treated as live-unknown to
        close the mid-acquire window (holder may have done mkdir but not yet written heartbeat).
        Slice: _acquire_run_lock edge-case: empty lock dir
        ### Given: .run.lock/ exists but contains no heartbeat file
        ### When: acquire runs
        ### Then: exits 1 with a message indicating the heartbeat is not yet recorded
        """
        lock_dir = tmp_path / ".run.lock"
        lock_dir.mkdir()
        # no heartbeat written

        r = _run(_run_lock_script(str(tmp_path)))
        assert r.returncode == 1
        assert "no heartbeat recorded" in r.stderr


class TestRunLockReclaimAndSafety:
    """
    Scenario: Reclaim is gated by a second mkdir; an empty base dir is refused.
    Slice: runs/_run_lock.sh (review blockers S2, B5, B3)
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
        ### Given: .run.lock/ has a stale heartbeat and .run.lock.reclaim exists
        ### When: _run_lock.sh is sourced
        ### Then: exits 1 with 'reclaiming' and the stale lock dir is left in place
        """
        lock_dir = tmp_path / ".run.lock"
        _write_stale_lock(lock_dir)
        (tmp_path / ".run.lock.reclaim").mkdir()

        r = _run(_run_lock_script(str(tmp_path)))
        assert r.returncode == 1
        assert "reclaiming" in r.stderr
        assert (lock_dir / "heartbeat").exists()

    def test_reclaim_gate_is_released_after_successful_reclaim(self, tmp_path):
        """
        Scenario: Stale lock is reclaimed successfully.
        Slice: B5 gate cleanup
        ### Given: .run.lock/ has a stale heartbeat, no gate present
        ### When: _run_lock.sh is sourced and exits
        ### Then: exits 0 and neither .run.lock nor .run.lock.reclaim remains
        """
        lock_dir = tmp_path / ".run.lock"
        _write_stale_lock(lock_dir)

        r = _run(_run_lock_script(str(tmp_path)))
        assert r.returncode == 0, r.stderr
        assert not (tmp_path / ".run.lock").exists()
        assert not (tmp_path / ".run.lock.reclaim").exists()

    def test_live_heartbeat_holder_is_never_reclaimed_and_leaves_no_gate(self, tmp_path):
        """
        Scenario: Lock is held with a fresh heartbeat; no reclaim is attempted.
        Slice: B5 — live holder must never be taken over
        ### Given: .run.lock/ with a current-timestamp heartbeat and run_id "live-run"
        ### When: _run_lock.sh is sourced
        ### Then: exits 1, the live lock is untouched, and no .reclaim dir is created
        """
        lock_dir = tmp_path / ".run.lock"
        _write_fresh_lock(lock_dir, run_id="live-run")

        r = _run(_run_lock_script(str(tmp_path)))
        assert r.returncode == 1
        assert (lock_dir / "heartbeat").exists()
        assert not (tmp_path / ".run.lock.reclaim").exists()

    def test_lock_not_released_when_run_id_overwritten(self, tmp_path):
        """
        Scenario: Lock run_id is overwritten mid-run with a different value; EXIT trap
                  must NOT delete the lock (it now belongs to a different holder).
        Slice: _release_run_lock ownership check
        ### Given: _run_lock.sh acquires the lock (writes its own run_id), then the run_id
                   file is overwritten with a different value before the script exits
        ### When: the script exits normally (EXIT trap fires)
        ### Then: .run.lock dir still exists — the lock belongs to the other run_id
        """
        script = f"""#!/usr/bin/env bash
set -euo pipefail
export NANOCHAT_BASE_DIR="{tmp_path}"
source "{RUN_LOCK}"
# Overwrite run_id with a different value AFTER acquiring
echo "other-run-id" > "{tmp_path}/.run.lock/run_id"
# EXIT trap fires here: sees run_id "other-run-id" != our _LOCK_RUN_ID, must NOT rm the lock
"""
        r = _run(script)
        assert r.returncode == 0, r.stderr
        lock_dir = tmp_path / ".run.lock"
        assert lock_dir.exists(), "lock was released despite run_id mismatch — ownership check missing"


class TestRunLockHeartbeat:
    """
    Scenario: Staleness is determined by heartbeat age, not by kill -0 on a container PID.
    Slice: runs/_run_lock.sh B3 — heartbeat-based staleness (600 s threshold)
    """

    def test_fresh_lock_is_not_reclaimed(self, tmp_path):
        """
        Scenario: A run holds the lock with a fresh heartbeat; a second process must be refused.
        Slice: B3 fresh heartbeat blocks acquire
        ### Given: .run.lock/ with heartbeat written just now
        ### When: a second sourcing attempt runs
        ### Then: exits 1 — fresh heartbeat means the holder is considered alive
        """
        lock_dir = tmp_path / ".run.lock"
        _write_fresh_lock(lock_dir, run_id="active-run")

        r = _run(_run_lock_script(str(tmp_path)))
        assert r.returncode == 1
        assert "ERROR" in r.stderr

    def test_stale_heartbeat_lock_is_reclaimed(self, tmp_path):
        """
        Scenario: Heartbeat is > 600 s old (cancelled Modal container); lock must be reclaimed.
        Slice: B3 stale heartbeat triggers reclaim
        ### Given: .run.lock/ with heartbeat written 660 s ago
        ### When: a new sourcing attempt runs
        ### Then: exits 0 — stale heartbeat means old holder is gone
        """
        lock_dir = tmp_path / ".run.lock"
        _write_stale_lock(lock_dir, run_id="dead-run")

        r = _run(_run_lock_script(str(tmp_path)))
        assert r.returncode == 0, r.stderr
        assert "ACQUIRED" in r.stdout

    def test_fresh_heartbeat_with_different_run_id_not_reclaimed(self, tmp_path):
        """
        Scenario: Lock shows a different run-ID but its heartbeat is fresh; must NOT reclaim.
        This catches the case where a new container restarted with a different PID but the
        previous run's heartbeat is still being refreshed by another process.
        Slice: B3 — run-ID difference alone is not sufficient to reclaim
        ### Given: .run.lock/ with run_id "other-run" and a fresh (current) heartbeat
        ### When: a new run tries to acquire with a different NANOCHAT_RUN_ID
        ### Then: exits 1 — the fresh heartbeat indicates an active holder regardless of run_id
        """
        lock_dir = tmp_path / ".run.lock"
        _write_fresh_lock(lock_dir, run_id="other-run")

        r = _run(_run_lock_script(str(tmp_path), extra_env={"NANOCHAT_RUN_ID": "new-run"}))
        assert r.returncode == 1
        assert "ERROR" in r.stderr
