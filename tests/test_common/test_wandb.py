"""Tests for LocalWandb offline logger."""

import json

from nanochat.common.wandb import LocalWandb


def test_creates_output_dir(tmp_path):
    run_dir = tmp_path / "runs" / "my-run"
    assert not run_dir.exists()
    w = LocalWandb("my-run", base_dir=str(tmp_path))
    w.finish()
    assert run_dir.exists()


def test_log_writes_jsonl(tmp_path):
    w = LocalWandb("test-run", base_dir=str(tmp_path))
    w.log({"step": 0, "loss": 1.5})
    w.log({"step": 1, "loss": 1.2})
    w.finish()

    lines = (tmp_path / "runs" / "test-run" / "wandb.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"step": 0, "loss": 1.5}
    assert json.loads(lines[1]) == {"step": 1, "loss": 1.2}


def test_finish_closes_file(tmp_path):
    w = LocalWandb("test-run", base_dir=str(tmp_path))
    w.finish()
    assert w._f.closed
