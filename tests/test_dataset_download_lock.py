"""Concurrent downloaders must not clobber a shared cached training shard."""
import io
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nanochat import dataset


@pytest.fixture
def shard(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(dataset.time, "sleep", lambda seconds: None)
    buffer = io.BytesIO()
    pq.write_table(pa.table({"text": ["example training document"]}), buffer)
    payload = buffer.getvalue()
    return tmp_path / dataset.index_to_filename(0), payload


class Response:
    def __init__(self, payload, started=None, release=None, fail=False):
        self.payload = payload
        self.started = started
        self.release = release
        self.fail = fail

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.payload[:8]
        if self.started is not None:
            self.started.set()
            assert self.release.wait(timeout=10)
        if self.fail:
            raise dataset.requests.RequestException("stream interrupted")
        yield self.payload[8:]


def test_concurrent_downloads_use_one_writer_and_preserve_the_shard(shard, monkeypatch):
    path, payload = shard
    writing, release, second_started = Event(), Event(), Event()
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            return Response(payload, writing, release)
        if len(calls) == 2:
            return Response(payload)
        raise dataset.requests.RequestException("duplicate download should not be necessary")

    def second_download():
        second_started.set()
        return dataset.download_single_file(0)

    monkeypatch.setattr(dataset.requests, "get", get)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(dataset.download_single_file, 0)
        assert writing.wait(timeout=10)
        second = pool.submit(second_download)
        assert second_started.wait(timeout=10)
        try:
            try:
                second.result(timeout=0.1)
                finished_early = True
            except TimeoutError:
                finished_early = False
        finally:
            release.set()
        first_result, second_result = first.result(timeout=10), second.result(timeout=10)
    assert path.exists(), "A successful writer's final shard must survive its peer's retries"
    assert path.read_bytes() == payload
    assert pq.read_table(path).to_pydict() == {"text": ["example training document"]}
    assert first_result and second_result
    assert not finished_early
    assert len(calls) == 1


def test_logging_failure_after_publication_does_not_delete_the_shard(shard, monkeypatch):
    path, payload = shard
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        return Response(payload)

    def broken_output(message):
        if message.startswith("Successfully downloaded"):
            raise BrokenPipeError("output consumer closed")

    monkeypatch.setattr(dataset.requests, "get", get)
    monkeypatch.setattr(dataset, "print", broken_output, raising=False)
    with pytest.raises(BrokenPipeError, match="consumer closed"):
        dataset.download_single_file(0)
    assert path.read_bytes() == payload
    assert dataset.download_single_file(0)
    assert len(calls) == 1


def test_cached_shard_needs_no_network(shard, monkeypatch):
    path, payload = shard
    path.write_bytes(payload)
    monkeypatch.setattr(dataset.requests, "get", lambda *args, **kwargs: pytest.fail("cached shard redownloaded"))
    assert dataset.download_single_file(0)
    assert path.read_bytes() == payload


def test_stream_failure_cleans_only_the_partial_download(shard, monkeypatch):
    path, payload = shard
    monkeypatch.setattr(dataset.requests, "get", lambda *args, **kwargs: Response(payload, fail=True))
    assert not dataset.download_single_file(0)
    assert not path.exists()
    assert not path.with_name(path.name + ".tmp").exists()
