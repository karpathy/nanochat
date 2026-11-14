"""Periodic Volume commit during training: due-time, failure retry, and daemon shutdown.

Run: python -m pytest tests/test_periodic_commit.py -v
"""
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runs"))

import modal_run


class _FakeVolume:
    def __init__(self, fail: bool = False) -> None:
        self.commits = 0
        self.fail = fail

    def commit(self) -> None:
        if self.fail:
            raise RuntimeError("volume unavailable")
        self.commits += 1


@pytest.fixture
def retry_secs(monkeypatch) -> int:
    monkeypatch.setattr(modal_run, "PERIODIC_COMMIT_RETRY_SECS", 30)
    return 30


def test_given_interval_not_elapsed_when_periodic_commit_then_no_commit():
    vol = _FakeVolume()
    last = modal_run._periodic_commit(vol, last_commit=1000.0, now=1299.0, every_secs=300)
    assert vol.commits == 0
    assert last == 1000.0


def test_given_interval_elapsed_when_periodic_commit_then_commits_and_resets_clock():
    vol = _FakeVolume()
    last = modal_run._periodic_commit(vol, last_commit=1000.0, now=1300.0, every_secs=300)
    assert vol.commits == 1
    assert last == 1300.0


def test_given_failing_commit_when_periodic_commit_then_retries_after_retry_window(retry_secs):
    vol = _FakeVolume(fail=True)
    last = modal_run._periodic_commit(vol, last_commit=1000.0, now=1300.0, every_secs=300)
    assert vol.commits == 0
    assert modal_run._periodic_commit(vol, last_commit=last, now=1329.0, every_secs=300) == last
    vol.fail = False
    retried = modal_run._periodic_commit(vol, last_commit=last, now=1330.0, every_secs=300)
    assert vol.commits == 1
    assert retried == 1330.0


def test_given_short_interval_when_periodic_commit_then_commits_each_interval():
    vol = _FakeVolume()
    last = 0.0
    for now in range(0, 1201, 60):
        last = modal_run._periodic_commit(vol, last_commit=last, now=float(now), every_secs=60)
    assert vol.commits == 20


def test_daemon_stops_when_stop_event_set(tmp_path):
    """
    Scenario: _commit_daemon exits its loop when stop_event is set.
    Slice: modal_run._commit_daemon shutdown path
    ### Given: a daemon thread and a stop event
    ### When: stop_event is set immediately after the thread starts
    ### Then: the thread terminates within 5 seconds
    """
    # ### Given
    vol = _FakeVolume()
    stop_event = threading.Event()
    t = threading.Thread(
        target=modal_run._commit_daemon,
        args=(vol, str(tmp_path), 60.0, stop_event),
        daemon=True,
    )
    # ### When
    t.start()
    stop_event.set()
    # ### Then
    t.join(timeout=5)
    assert not t.is_alive()
