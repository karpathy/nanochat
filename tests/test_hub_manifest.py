"""Dataset cache publication under concurrent torchrun ranks, without network access."""
import io
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tasks import common


def fake_hub(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "get_base_dir", lambda: str(tmp_path))
    parquet = io.BytesIO()
    pq.write_table(pa.table({"text": ["hello"]}), parquet)

    def urlopen(url):
        if "/api/datasets/" in url:
            return io.BytesIO(json.dumps(["https://example.test/shard.parquet"]).encode())
        return io.BytesIO(parquet.getvalue())

    monkeypatch.setattr(common.urllib.request, "urlopen", urlopen)


def test_manifest_is_not_visible_until_complete(tmp_path, monkeypatch):
    fake_hub(monkeypatch, tmp_path)
    publishing = Event()
    release = Event()
    original_dump = common.json.dump

    def paused_dump(value, stream, *args, **kwargs):
        publishing.set()
        assert release.wait(timeout=10)
        return original_dump(value, stream, *args, **kwargs)

    monkeypatch.setattr(common.json, "dump", paused_dump)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(common.load_hub_dataset, "test/data")
        assert publishing.wait(timeout=10)
        try:
            manifest = tmp_path / "task_data/test--data/default/train/manifest.json"
            assert not manifest.exists(), "Other ranks must not observe an incomplete manifest"
            second = pool.submit(common.load_hub_dataset, "test/data")
            with pytest.raises(TimeoutError):
                second.result(timeout=0.1)
        finally:
            release.set()
        assert first.result(timeout=10)[0] == {"text": "hello"}
        assert second.result(timeout=10)[0] == {"text": "hello"}
    assert json.loads(manifest.read_text()) == ["00000.parquet"]


def test_interrupted_manifest_write_is_retried(tmp_path, monkeypatch):
    fake_hub(monkeypatch, tmp_path)
    original_dump = common.json.dump

    def interrupted_dump(value, stream, *args, **kwargs):
        stream.write("[")
        raise OSError("interrupted manifest write")

    monkeypatch.setattr(common.json, "dump", interrupted_dump)
    with pytest.raises(OSError, match="interrupted manifest"):
        common.load_hub_dataset("test/data")
    monkeypatch.setattr(common.json, "dump", original_dump)
    assert common.load_hub_dataset("test/data")[0] == {"text": "hello"}
