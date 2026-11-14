"""Periodic Volume commit during training: due-time and failure retry.

Run: python -m pytest tests/test_periodic_commit.py -v
"""
import sys
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
def interval(monkeypatch):
    monkeypatch.setattr(modal_run, "PERIODIC_COMMIT_SECS", 300)
    monkeypatch.setattr(modal_run, "PERIODIC_COMMIT_RETRY_SECS", 30)
    return 300


def test_given_interval_not_elapsed_when_periodic_commit_then_no_commit(interval):
    vol = _FakeVolume()
    last = modal_run._periodic_commit(vol, last_commit=1000.0, now=1299.0)
    assert vol.commits == 0
    assert last == 1000.0


def test_given_interval_elapsed_when_periodic_commit_then_commits_and_resets_clock(interval):
    vol = _FakeVolume()
    last = modal_run._periodic_commit(vol, last_commit=1000.0, now=1300.0)
    assert vol.commits == 1
    assert last == 1300.0


def test_given_failing_commit_when_periodic_commit_then_retries_after_retry_window(interval):
    vol = _FakeVolume(fail=True)
    last = modal_run._periodic_commit(vol, last_commit=1000.0, now=1300.0)
    assert vol.commits == 0
    # Next attempt is due at now + (RETRY - 0) = 1330, i.e. 30 s later, not 300 s.
    assert modal_run._periodic_commit(vol, last_commit=last, now=1329.0) == last
    vol.fail = False
    retried = modal_run._periodic_commit(vol, last_commit=last, now=1330.0)
    assert vol.commits == 1
    assert retried == 1330.0


def test_given_many_intervals_when_periodic_commit_then_commits_once_per_interval(interval):
    vol = _FakeVolume()
    last = 0.0
    for now in range(0, 1201, 60):
        last = modal_run._periodic_commit(vol, last_commit=last, now=float(now))
    # Due at 300, 600, 900, 1200 — four commits in twenty minutes.
    assert vol.commits == 4
