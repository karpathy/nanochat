"""Tests for LocalWandb offline logger and init_wandb helper."""

import json
import os

from nanochat.common.config import CommonConfig
from nanochat.common.wandb import DummyWandb, LocalWandb, init_wandb


def test_creates_output_dir(tmp_path):
    run_dir = tmp_path / "runs" / "nanochat" / "my-run"
    assert not run_dir.exists()
    w = LocalWandb("my-run", base_dir=str(tmp_path))
    w.finish()
    assert run_dir.exists()


def test_log_writes_jsonl(tmp_path):
    w = LocalWandb("test-run", base_dir=str(tmp_path))
    w.log({"step": 0, "loss": 1.5})
    w.log({"step": 1, "loss": 1.2})
    w.finish()

    lines = (tmp_path / "runs" / "nanochat" / "test-run" / "wandb.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"step": 0, "loss": 1.5}
    assert json.loads(lines[1]) == {"step": 1, "loss": 1.2}


def test_finish_closes_file(tmp_path):
    w = LocalWandb("test-run", base_dir=str(tmp_path))
    w.finish()
    assert w._f.closed


# ---------------------------------------------------------------------------
# init_wandb
# ---------------------------------------------------------------------------


def test_init_wandb_disabled(tmp_path):
    cfg = CommonConfig(run="my-run", wandb="disabled", base_dir=str(tmp_path))
    w = init_wandb(cfg, {}, master_process=True)
    assert isinstance(w, DummyWandb)


def test_init_wandb_local(tmp_path):
    cfg = CommonConfig(run="my-run", wandb="local", base_dir=str(tmp_path))
    w = init_wandb(cfg, {}, master_process=True)
    assert isinstance(w, LocalWandb)
    w.finish()


def test_init_wandb_non_master_always_dummy(tmp_path):
    """Non-master ranks always get DummyWandb regardless of wandb mode."""
    for mode in ("local", "disabled", "online"):
        cfg = CommonConfig(run="my-run", wandb=mode, base_dir=str(tmp_path))
        w = init_wandb(cfg, {}, master_process=False)
        assert isinstance(w, DummyWandb), f"expected DummyWandb for mode={mode}, non-master"


def test_init_wandb_legacy_dummy_run(tmp_path):
    """run='dummy' magic maps to DummyWandb even if wandb='local'."""
    cfg = CommonConfig(run="dummy", wandb="local", base_dir=str(tmp_path))
    w = init_wandb(cfg, {}, master_process=True)
    assert isinstance(w, DummyWandb)


def test_init_wandb_wandb_mode_env(tmp_path, monkeypatch):
    """WANDB_MODE=disabled env var maps to DummyWandb."""
    monkeypatch.setenv("WANDB_MODE", "disabled")
    cfg = CommonConfig(run="my-run", wandb="local", base_dir=str(tmp_path))
    w = init_wandb(cfg, {}, master_process=True)
    assert isinstance(w, DummyWandb)
