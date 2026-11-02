"""CORE initialization must wait for extraction, not just archive download."""
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Event

import pytest

from nanochat import common
from scripts import base_eval


def archive_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("eval_bundle/core.yaml", "icl_tasks:\n- label: toy\n  icl_task_type: multiple_choice\n  dataset_uri: toy.jsonl\n  num_fewshot: [0]\n")
        archive.writestr("eval_bundle/eval_meta_data.csv", "Eval Task,Random baseline\ntoy,0\n")
        archive.writestr("eval_bundle/eval_data/toy.jsonl", '{}\n')
    return buffer.getvalue()


def setup_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(base_eval, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(common.urllib.request, "urlopen", lambda url: io.BytesIO(archive_bytes()))
    monkeypatch.setattr(base_eval, "evaluate_task", lambda *args: 1.0)


def test_cached_archive_is_extracted_when_bundle_is_missing(monkeypatch, tmp_path):
    setup_bundle(monkeypatch, tmp_path)
    (tmp_path / "eval_bundle.zip").write_bytes(archive_bytes())
    result = base_eval.evaluate_core(None, None, None)
    assert result["core_metric"] == 1.0
    assert (tmp_path / "eval_bundle/core.yaml").is_file()


def test_parallel_evaluation_waits_for_extraction(monkeypatch, tmp_path):
    setup_bundle(monkeypatch, tmp_path)
    extracting = Event()
    release = Event()
    place_bundle = base_eval.place_eval_bundle

    def paused_extraction(path):
        extracting.set()
        assert release.wait(timeout=10)
        place_bundle(path)

    monkeypatch.setattr(base_eval, "place_eval_bundle", paused_extraction)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(base_eval.evaluate_core, None, None, None)
        assert extracting.wait(timeout=10)
        second = pool.submit(base_eval.evaluate_core, None, None, None)
        try:
            with pytest.raises(TimeoutError):
                second.result(timeout=0.1)
        finally:
            release.set()
        assert first.result(timeout=10)["core_metric"] == 1.0
        assert second.result(timeout=10)["core_metric"] == 1.0


def test_interrupted_extraction_is_not_published_and_can_retry(monkeypatch, tmp_path):
    setup_bundle(monkeypatch, tmp_path)
    original_extract = zipfile.ZipFile.extractall

    def interrupted_extract(archive, directory):
        original_extract(archive, directory)
        raise OSError("extraction interrupted")

    monkeypatch.setattr(zipfile.ZipFile, "extractall", interrupted_extract)
    with pytest.raises(OSError, match="extraction interrupted"):
        base_eval.evaluate_core(None, None, None)
    assert not (tmp_path / "eval_bundle").exists()
    assert not [path for path in tmp_path.iterdir() if path.is_dir()]
    assert (tmp_path / "eval_bundle.zip").is_file()
    monkeypatch.setattr(zipfile.ZipFile, "extractall", original_extract)
    assert base_eval.evaluate_core(None, None, None)["core_metric"] == 1.0
