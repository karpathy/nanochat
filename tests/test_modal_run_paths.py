"""
Author: nanochat Modal slice (Claude)
Created: 2026-11-15
Scope: runs/modal_run.py artefact archive and commit-daemon paths on the mode-prefixed Volume.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
if str(RUNS_DIR) not in sys.path:
    sys.path.insert(0, str(RUNS_DIR))

import modal_run


@pytest.fixture
def volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / "vol"
    monkeypatch.setattr(modal_run, "BASE_DIR", str(base))
    monkeypatch.setattr(modal_run, "vol", Mock())
    return base


def test_archive_reads_the_mode_folder_not_the_volume_root(volume: Path) -> None:
    # Given: a new run's tokenizer under cpu/, and legacy unprefixed data at the root
    (volume / "cpu" / "tokenizer").mkdir(parents=True)
    (volume / "cpu" / "tokenizer" / "tokenizer.pkl").write_bytes(b"new")
    (volume / "tokenizer").mkdir()
    (volume / "tokenizer" / "legacy.pkl").write_bytes(b"old")

    # When: the run's artefacts are archived
    modal_run._archive_artifacts("run-1", "cpu")

    # Then: only the mode folder's files are in the manifest
    manifest = json.loads((volume / "runs" / "run-1" / "manifest.json").read_text())
    paths = [entry["path"] for entry in manifest["files"]]
    assert "tokenizer/tokenizer.pkl" in paths
    assert not any("legacy" in path for path in paths)


def test_training_commit_daemon_watches_the_mode_folder(volume: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: stage scripts run in the cpu mode folder, with no real process or Volume commit
    (volume / "cpu").mkdir(parents=True)
    started: dict[str, tuple] = {}

    class FakeThread:
        def __init__(self, target, args, daemon):
            started["args"] = args

        def start(self) -> None:
            return None

    class FakeProc:
        def wait(self) -> int:
            return 0

    monkeypatch.setenv("NANOCHAT_MODAL_MODE", "cpu")
    monkeypatch.setattr(modal_run.os, "environ", dict(os.environ))
    monkeypatch.setattr(modal_run.threading, "Thread", FakeThread)
    monkeypatch.setattr(modal_run.subprocess, "Popen", lambda *a, **k: FakeProc())
    monkeypatch.setattr(modal_run.os, "chdir", lambda path: None)
    monkeypatch.setattr(modal_run.signal, "signal", lambda *a, **k: None)
    monkeypatch.setattr(modal_run, "vol", Mock())
    monkeypatch.setattr(modal_run, "_archive_artifacts", lambda *a, **k: None)

    # When: a stage runs
    modal_run._run_training("runs/runcpu.sh", {})

    # Then: the commit daemon is pointed at the mode folder, where the hook writes its files
    assert started["args"][1] == f"{volume}/cpu"
