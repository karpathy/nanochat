import os
import runpy
import types

import pytest
import torch

import nanochat.dataset as dataset
import nanochat.dataloader as dataloader


def test_list_parquet_files(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"")
    (tmp_path / "b.tmp").write_bytes(b"")
    (tmp_path / "c.parquet.tmp").write_bytes(b"")
    (tmp_path / "d.parquet").write_bytes(b"")
    got = dataset.list_parquet_files(str(tmp_path))
    assert got == [str(tmp_path / "a.parquet"), str(tmp_path / "d.parquet")]


def test_parquets_iter_batched(monkeypatch):
    files = ["/p0.parquet", "/p1.parquet"]
    monkeypatch.setattr(dataset, "list_parquet_files", lambda: files)

    class RG:
        def __init__(self, vals):
            self._vals = vals

        def column(self, _name):
            return types.SimpleNamespace(to_pylist=lambda: self._vals)

    class PF:
        def __init__(self, _path):
            self.num_row_groups = 3

        def read_row_group(self, idx):
            return RG([f"row-{idx}"])

    monkeypatch.setattr(dataset.pq, "ParquetFile", PF)

    train = list(dataset.parquets_iter_batched("train", start=0, step=2))
    val = list(dataset.parquets_iter_batched("val", start=1, step=2))
    assert train == [["row-0"], ["row-2"]]
    assert val == [["row-1"]]

    with pytest.raises(AssertionError):
        list(dataset.parquets_iter_batched("bad"))


def test_download_single_file(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(dataset, "BASE_URL", "http://example.test")

    # Existing file -> skip.
    pre = tmp_path / dataset.index_to_filename(1)
    pre.write_bytes(b"x")
    assert dataset.download_single_file(1) is True

    # Success path.
    class Resp:
        def __init__(self):
            self.ok = True

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            del chunk_size
            yield b"abc"
            yield b""
            yield b"def"

    monkeypatch.setattr(dataset.requests, "get", lambda *a, **k: Resp())
    assert dataset.download_single_file(2) is True
    out = tmp_path / dataset.index_to_filename(2)
    assert out.read_bytes() == b"abcdef"

    # Failure with retries and eventual False.
    calls = {"n": 0}

    def bad_get(*_a, **_k):
        calls["n"] += 1
        raise dataset.requests.RequestException("boom")

    slept = {"n": 0}
    monkeypatch.setattr(dataset.requests, "get", bad_get)
    monkeypatch.setattr(dataset.time, "sleep", lambda _s: slept.__setitem__("n", slept["n"] + 1))
    assert dataset.download_single_file(3) is False
    assert calls["n"] == 5
    assert slept["n"] == 4

    # Execute final return False line by skipping attempts loop.
    monkeypatch.setattr(dataset, "range", lambda *_a, **_k: [], raising=False)
    assert dataset.download_single_file(4) is False


def test_document_batches(monkeypatch):
    monkeypatch.setattr(dataloader, "list_parquet_files", lambda: ["/a.parquet", "/b.parquet", "/c.parquet"])
    monkeypatch.setattr(dataloader, "get_dist_info", lambda: (True, 1, 0, 2))

    class RG:
        def __init__(self, vals):
            self._vals = vals

        def column(self, _name):
            return types.SimpleNamespace(to_pylist=lambda: self._vals)

    class PF:
        def __init__(self, path):
            self.path = path
            # Make first file tiny to trigger continue in resume branch.
            self.num_row_groups = 1 if "a.parquet" in path else 4

        def read_row_group(self, idx):
            return RG([f"{self.path}-r{idx}-d0", f"{self.path}-r{idx}-d1"])

    monkeypatch.setattr(dataloader.pq, "ParquetFile", PF)

    # Resume in first file should continue to next file due rg_idx>=num_row_groups.
    gen = dataloader._document_batches("train", {"pq_idx": 0, "rg_idx": 3, "epoch": 5}, tokenizer_batch_size=1)
    b1, state1 = next(gen)
    assert len(b1) == 1
    assert state1[0] == 1
    assert state1[2] >= 5

    # Non-resume and val split.
    gen2 = dataloader._document_batches("val", None, tokenizer_batch_size=2)
    b2, state2 = next(gen2)
    assert len(b2) == 2
    assert state2[0] == 0

    monkeypatch.setattr(dataloader, "list_parquet_files", lambda: [])
    with pytest.raises(AssertionError):
        next(dataloader._document_batches("train", None, tokenizer_batch_size=1))


def test_document_batches_resume_cleared_and_epoch_increment(monkeypatch):
    monkeypatch.setattr(dataloader, "list_parquet_files", lambda: ["/only.parquet", "/val.parquet"])
    monkeypatch.setattr(dataloader, "get_dist_info", lambda: (True, 0, 0, 2))

    class RG:
        def __init__(self, vals):
            self._vals = vals

        def column(self, _name):
            return types.SimpleNamespace(to_pylist=lambda: self._vals)

    class PF:
        def __init__(self, _path):
            self.num_row_groups = 3

        def read_row_group(self, idx):
            return RG([f"r{idx}"])

    monkeypatch.setattr(dataloader.pq, "ParquetFile", PF)
    gen = dataloader._document_batches("train", {"pq_idx": 0, "rg_idx": 0, "epoch": 1}, tokenizer_batch_size=1)
    b1, s1 = next(gen)
    b2, s2 = next(gen)
    assert b1 == ["r2"] and s1 == (0, 2, 1)
    # next epoch yield proves first_pass->False and epoch increment happened
    assert b2 == ["r0"] and s2[2] >= 2


def test_tokenizing_loader_and_wrapper(monkeypatch):
    class Tok:
        def get_bos_token_id(self):
            return 7

        def encode(self, doc_batch, prepend=None, num_threads=None):
            del num_threads
            out = []
            for t in doc_batch:
                row = [prepend] if prepend is not None else []
                row.extend([len(t), len(t) + 1, len(t) + 2])
                out.append(row)
            return out

    batches = iter([
        (["a", "bb", "ccc"], (1, 2, 3)),
        (["dddd", "eeeee", "ffffff"], (4, 5, 6)),
    ])
    monkeypatch.setattr(dataloader, "_document_batches", lambda *a, **k: batches)

    gen = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer=Tok(),
        B=2,
        T=4,
        split="train",
        tokenizer_threads=1,
        tokenizer_batch_size=2,
        device="cpu",
        resume_state_dict=None,
        buffer_size=2,
    )
    x, y, st = next(gen)
    assert x.shape == (2, 4)
    assert y.shape == (2, 4)
    assert st == {"pq_idx": 4, "rg_idx": 5, "epoch": 6}

    with pytest.raises(AssertionError):
        next(
            dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
                tokenizer=Tok(),
                B=1,
                T=2,
                split="bad",
                device="cpu",
            )
        )

    batches2 = iter([(["z"], (0, 0, 1))])
    monkeypatch.setattr(dataloader, "_document_batches", lambda *a, **k: batches2)
    gen2 = dataloader.tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer=Tok(),
        B=1,
        T=2,
        split="val",
        device="cpu",
        buffer_size=1,
    )
    x2, y2 = next(gen2)
    assert x2.shape == (1, 2)
    assert y2.shape == (1, 2)


def test_download_single_file_cleanup_except_and_dataset_main(monkeypatch, tmp_path):
    # Cleanup except path (os.remove raises).
    monkeypatch.setattr(dataset, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(dataset, "BASE_URL", "http://example.test")
    monkeypatch.setattr(dataset, "range", lambda *_a, **_k: [5], raising=False)
    monkeypatch.setattr(dataset.requests, "get", lambda *_a, **_k: (_ for _ in ()).throw(dataset.requests.RequestException("x")))

    filename = dataset.index_to_filename(9)
    fp = str(tmp_path / filename)
    tp = str(tmp_path / f"{filename}.tmp")
    with open(tp, "w", encoding="utf-8") as f:
        f.write("y")

    real_exists = dataset.os.path.exists
    calls = {"fp": 0}

    def fake_exists(path):
        if path == fp:
            calls["fp"] += 1
            return calls["fp"] > 1  # first check in function should be False; cleanup checks True
        if path == tp:
            return True
        return real_exists(path)

    monkeypatch.setattr(dataset.os.path, "exists", fake_exists)
    monkeypatch.setattr(dataset.os, "remove", lambda _p: (_ for _ in ()).throw(OSError("deny")))
    assert dataset.download_single_file(9) is False

    # __main__ block coverage.
    class FakeParser:
        def __init__(self, *a, **k):
            pass

        def add_argument(self, *a, **k):
            return None

        def parse_args(self):
            return types.SimpleNamespace(num_files=2, num_workers=3)

    class FakePool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, fn, ids):
            del fn
            return [True for _ in ids]

    monkeypatch.setattr("argparse.ArgumentParser", FakeParser)
    monkeypatch.setattr("multiprocessing.Pool", FakePool)
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "base"))
    runpy.run_module("nanochat.dataset", run_name="__main__")
