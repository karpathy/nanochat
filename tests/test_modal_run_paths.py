"""
Author: nanochat Modal slice (Claude)
Created: 2026-11-15
Scope: runs/modal_run.py artefact archive, commit-daemon paths, and
       _download_artifacts() recursive checkpoint walk on the mode-prefixed Volume.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
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


class FakeVolume:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def listdir(self, path: str, *, recursive: bool = False) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(path=p, size=len(b), type=modal_run.modal.volume.FileEntryType.FILE)
            for p, b in self.files.items()
            if p.startswith(f"{path}/")
        ]

    def read_file(self, path: str):
        yield self.files[path]


def _chat_volume_files() -> dict[str, bytes]:
    return {
        "cpu/tokenizer/tokenizer.pkl": b"tok",
        "cpu/tokenizer/token_bytes.pt": b"bytes",
        "cpu/chatsft_checkpoints/.train_ok": b"",
        "cpu/chatsft_checkpoints/d6/model_000100.pt": b"old-weights",
        "cpu/chatsft_checkpoints/d6/meta_000100.json": b"{}",
        "cpu/chatsft_checkpoints/d6/model_001499.pt": b"new-weights",
        "cpu/chatsft_checkpoints/d6/meta_001499.json": b'{"step": 1499}',
        "cpu/chatsft_checkpoints/d6/optim_001499_rank0.pt": b"optimizer",
    }


def test_download_artifacts_copies_only_newest_chat_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Scenario: The Volume holds every saved checkpoint. Chat needs the tokenizer and the newest model.
    Slice: _download_artifacts() selects files instead of copying whole folders.
    ### Given: a Volume with two checkpoints and an optim_* file for the newest step
    ### When: _download_artifacts("cpu") runs
    ### Then: the newest model and meta are copied; the old checkpoint and optim_* are not
    """
    # ### Given
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(modal_run.modal.Volume, "from_name", lambda name: FakeVolume(_chat_volume_files()))
    cache_dir = tmp_path / ".cache" / "nanochat-modal" / "cpu"

    # ### When
    modal_run._download_artifacts("cpu")

    # ### Then
    assert (cache_dir / "tokenizer" / "tokenizer.pkl").read_bytes() == b"tok"
    assert (cache_dir / "chatsft_checkpoints" / "d6" / "model_001499.pt").read_bytes() == b"new-weights"
    assert (cache_dir / "chatsft_checkpoints" / "d6" / "meta_001499.json").exists()
    assert not (cache_dir / "chatsft_checkpoints" / "d6" / "model_000100.pt").exists()
    assert not (cache_dir / "chatsft_checkpoints" / "d6" / "optim_001499_rank0.pt").exists()


def test_fetch_file_skips_a_same_size_copy(tmp_path: Path) -> None:
    # ### Given: a local file already present with the expected size
    local = tmp_path / "model_001499.pt"
    local.write_bytes(b"abcd")
    volume = FakeVolume({"cpu/x/model_001499.pt": b"WXYZ"})

    # ### When / Then: nothing is re-copied
    assert modal_run._fetch_file(volume, "cpu/x/model_001499.pt", 4, str(local)) is False
    assert local.read_bytes() == b"abcd"


def test_fetch_file_size_mismatch_leaves_no_partial_file(tmp_path: Path) -> None:
    # ### Given: the Volume reports 10 bytes but the content is 3
    local = tmp_path / "model_001499.pt"
    volume = FakeVolume({"cpu/x/model_001499.pt": b"abc"})

    # ### When / Then: the copy fails and no file is left behind to look complete
    with pytest.raises(RuntimeError, match="size mismatch"):
        modal_run._fetch_file(volume, "cpu/x/model_001499.pt", 10, str(local))
    assert not local.exists()
    assert not (tmp_path / "model_001499.pt.part").exists()


def test_fetch_file_refuses_a_folder_where_a_file_belongs(tmp_path: Path) -> None:
    # ### Given: a folder sits where the file should go (the cause of the Errno 21 failure)
    local = tmp_path / "chatsft_checkpoints"
    local.mkdir()

    # ### When / Then: a clear error instead of a confusing OS error
    with pytest.raises(RuntimeError, match="is a folder where a file is expected"):
        modal_run._fetch_file(FakeVolume({}), "cpu/chatsft_checkpoints", 1, str(local))


def test_download_artifacts_falls_back_to_newest_complete_pair_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # ### Given: the newest model (1499) has no meta file on the Volume
    files = _chat_volume_files()
    del files["cpu/chatsft_checkpoints/d6/meta_001499.json"]
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(modal_run.modal.Volume, "from_name", lambda name: FakeVolume(files))
    cache_dir = tmp_path / ".cache" / "nanochat-modal" / "cpu"

    # ### When
    modal_run._download_artifacts("cpu")

    # ### Then: the older complete pair is used, and the operator is told
    assert (cache_dir / "chatsft_checkpoints" / "d6" / "model_000100.pt").exists()
    assert "WARNING: newest step in d6" in capsys.readouterr().out


def test_download_artifacts_fails_when_no_pair_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ### Given: the only checkpoint has a model but no meta
    files = {
        "cpu/tokenizer/tokenizer.pkl": b"tok",
        "cpu/chatsft_checkpoints/d6/model_000100.pt": b"weights",
    }
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(modal_run.modal.Volume, "from_name", lambda name: FakeVolume(files))

    # ### When / Then
    with pytest.raises(RuntimeError, match="no complete model\\+meta pair"):
        modal_run._download_artifacts("cpu")
