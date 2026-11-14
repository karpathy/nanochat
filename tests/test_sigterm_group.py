"""SIGTERM stop of the training process group: the whole group must stop, not just bash.

Run: python -m pytest tests/test_sigterm_group.py -v
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runs"))

import modal_run


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_given_bash_with_grandchild_when_terminate_group_then_grandchild_stops(tmp_path):
    # Given a bash wrapper whose child (a grandchild of the Popen) is still running
    pid_file = tmp_path / "child.pid"
    proc = subprocess.Popen(
        ["bash", "-c", f"sleep 300 & echo $! > {pid_file}; wait"],
        start_new_session=True,
    )
    for _ in range(50):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        time.sleep(0.05)
    child_pid = int(pid_file.read_text().strip())

    # When the group is stopped
    modal_run._terminate_group(proc)
    proc.wait(timeout=10)

    # Then the grandchild is gone too
    for _ in range(50):
        if not _alive(child_pid):
            break
        time.sleep(0.05)
    assert not _alive(child_pid)
