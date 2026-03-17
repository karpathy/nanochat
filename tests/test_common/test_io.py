"""Tests for file I/O utilities."""

import os
import urllib.request

import pytest

from nanochat.common.io import download_file_with_lock, print0


# ---------------------------------------------------------------------------
# print0
# ---------------------------------------------------------------------------


def test_print0_rank0_prints(monkeypatch, capsys):
    monkeypatch.setenv("RANK", "0")
    print0("hello")
    assert capsys.readouterr().out.strip() == "hello"


def test_print0_non_master_silent(monkeypatch, capsys):
    monkeypatch.setenv("RANK", "1")
    print0("should not appear")
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# download_file_with_lock
# ---------------------------------------------------------------------------


def test_skips_download_if_file_exists(tmp_path):
    """File already present — urlopen must never be called."""
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True)
    target = data_root / "file.bin"
    target.write_bytes(b"data")

    called = []

    def fake_urlopen(url):
        called.append(url)

    import urllib.request as ur
    original = ur.urlopen
    ur.urlopen = fake_urlopen
    try:
        result = download_file_with_lock(str(tmp_path), "http://example.com/file.bin", "file.bin")
    finally:
        ur.urlopen = original

    assert not called
    assert result == str(target)


def test_double_check_inside_lock(tmp_path, monkeypatch):
    """Second rank arriving at the lock finds the file already downloaded."""
    data_dir = tmp_path / "data" / "climbmix"
    data_dir.mkdir(parents=True)
    target = data_dir / "tok.bin"

    download_count = []

    class FakeResponse:
        def read(self):
            download_count.append(1)
            return b"content"
        def __enter__(self): return self
        def __exit__(self, *a): pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda url: FakeResponse())

    # First call downloads
    download_file_with_lock(str(tmp_path), "http://example.com/tok.bin", "tok.bin")
    assert len(download_count) == 1

    # Second call hits the outer early-exit (file exists)
    download_file_with_lock(str(tmp_path), "http://example.com/tok.bin", "tok.bin")
    assert len(download_count) == 1


def test_lock_file_removed_after_download(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "climbmix"
    data_dir.mkdir(parents=True)

    class FakeResponse:
        def read(self): return b"bytes"
        def __enter__(self): return self
        def __exit__(self, *a): pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda url: FakeResponse())

    download_file_with_lock(str(tmp_path), "http://example.com/f.bin", "f.bin")

    lock_path = str(data_dir / "f.bin") + ".lock"
    assert not os.path.exists(lock_path)


def test_postprocess_fn_called(tmp_path, monkeypatch):
    data_dir = tmp_path / "data" / "climbmix"
    data_dir.mkdir(parents=True)

    class FakeResponse:
        def read(self): return b"raw"
        def __enter__(self): return self
        def __exit__(self, *a): pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda url: FakeResponse())

    processed = []
    download_file_with_lock(
        str(tmp_path), "http://example.com/x.bin", "x.bin",
        postprocess_fn=lambda p: processed.append(p),
    )
    assert len(processed) == 1
    assert processed[0].endswith("x.bin")
