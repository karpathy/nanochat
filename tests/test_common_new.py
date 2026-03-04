import io
import logging
import os
import types

import pytest
import torch

import nanochat.common as common


def test_colored_formatter_info_and_non_info():
    fmt = common.ColoredFormatter("%(levelname)s %(message)s")
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "Shard 2: 10 MB 5%", (), None)
    out = fmt.format(rec)
    assert "Shard 2" in out
    assert "10 MB" in out
    assert "5%" in out

    rec2 = logging.LogRecord("x", logging.WARNING, __file__, 1, "warn", (), None)
    out2 = fmt.format(rec2)
    assert "warn" in out2


def test_setup_default_logging(monkeypatch):
    calls = {}

    def fake_basic_config(**kwargs):
        calls["kwargs"] = kwargs

    monkeypatch.setattr(common.logging, "basicConfig", fake_basic_config)
    common.setup_default_logging()
    assert calls["kwargs"]["level"] == logging.INFO
    assert len(calls["kwargs"]["handlers"]) == 1


def test_get_base_dir_env_and_default(monkeypatch, tmp_path):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "from_env"))
    got = common.get_base_dir()
    assert got.endswith("from_env")
    assert os.path.isdir(got)

    monkeypatch.delenv("NANOCHAT_BASE_DIR", raising=False)
    monkeypatch.setattr(common.os.path, "expanduser", lambda _: str(tmp_path))
    got2 = common.get_base_dir()
    assert got2.endswith(".cache/nanochat")
    assert os.path.isdir(got2)


def test_download_file_with_lock_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "get_base_dir", lambda: str(tmp_path))
    out = tmp_path / "f.bin"
    out.write_bytes(b"x")
    assert common.download_file_with_lock("http://x", "f.bin") == str(out)

    out.unlink()
    payload = b"hello"
    marker = {"post": None}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return payload

    monkeypatch.setattr(common.urllib.request, "urlopen", lambda _url: Resp())
    got = common.download_file_with_lock(
        "http://example.test",
        "f.bin",
        postprocess_fn=lambda p: marker.__setitem__("post", p),
    )
    assert got == str(out)
    assert out.read_bytes() == payload
    assert marker["post"] == str(out)


def test_download_file_with_lock_recheck_after_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(common, "get_base_dir", lambda: str(tmp_path))
    file_path = str(tmp_path / "race.bin")

    calls = {"n": 0}
    real_exists = common.os.path.exists

    def fake_exists(path):
        if path == file_path:
            calls["n"] += 1
            # first check (before lock): missing; second check (inside lock): present
            return calls["n"] >= 2
        return real_exists(path)

    class Lock:
        def __init__(self, _p):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(common.os.path, "exists", fake_exists)
    monkeypatch.setattr(common, "FileLock", Lock)
    got = common.download_file_with_lock("http://unused", "race.bin")
    assert got == file_path


def test_print0_and_banner(monkeypatch, capsys):
    monkeypatch.setenv("RANK", "1")
    common.print0("hidden")
    assert capsys.readouterr().out == ""

    monkeypatch.setenv("RANK", "0")
    common.print0("shown")
    assert "shown" in capsys.readouterr().out

    common.print_banner()
    assert "█████" in capsys.readouterr().out


def test_ddp_helpers(monkeypatch):
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(k, raising=False)
    assert common.is_ddp_requested() is False
    assert common.get_dist_info() == (False, 0, 0, 1)

    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "8")
    assert common.is_ddp_requested() is True
    assert common.get_dist_info() == (True, 3, 2, 8)

    monkeypatch.setattr(common.dist, "is_available", lambda: True)
    monkeypatch.setattr(common.dist, "is_initialized", lambda: True)
    assert common.is_ddp_initialized() is True


def test_autodetect_device_type(monkeypatch):
    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: True)
    assert common.autodetect_device_type() == "cuda"

    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(common.torch.backends.mps, "is_available", lambda: True)
    assert common.autodetect_device_type() == "mps"

    monkeypatch.setattr(common.torch.backends.mps, "is_available", lambda: False)
    assert common.autodetect_device_type() == "cpu"


def test_compute_init_cpu_and_cleanup(monkeypatch):
    monkeypatch.setattr(common, "get_dist_info", lambda: (False, 0, 0, 1))
    out = common.compute_init("cpu")
    assert out[0] is False
    assert out[-1].type == "cpu"

    called = {"destroy": 0}
    monkeypatch.setattr(common, "is_ddp_initialized", lambda: True)
    monkeypatch.setattr(common.dist, "destroy_process_group", lambda: called.__setitem__("destroy", 1))
    common.compute_cleanup()
    assert called["destroy"] == 1


def test_compute_init_mps_and_cuda_paths(monkeypatch):
    monkeypatch.setattr(common, "get_dist_info", lambda: (True, 1, 0, 2))
    monkeypatch.setattr(common.torch.backends.mps, "is_available", lambda: True)
    out = common.compute_init("mps")
    assert out[0] is True
    assert out[-1].type == "mps"

    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: True)
    set_device_calls = {}
    init_calls = {}
    barrier_calls = {}
    matmul_calls = {}
    seed_calls = {}
    cuda_seed_calls = {}

    monkeypatch.setattr(common.torch.cuda, "set_device", lambda d: set_device_calls.__setitem__("device", d))
    monkeypatch.setattr(common.dist, "init_process_group", lambda **kw: init_calls.__setitem__("kw", kw))
    monkeypatch.setattr(common.dist, "barrier", lambda: barrier_calls.__setitem__("n", 1))
    monkeypatch.setattr(common.torch, "set_float32_matmul_precision", lambda x: matmul_calls.__setitem__("x", x))
    monkeypatch.setattr(common.torch, "manual_seed", lambda x: seed_calls.__setitem__("x", x))
    monkeypatch.setattr(common.torch.cuda, "manual_seed", lambda x: cuda_seed_calls.__setitem__("x", x))

    out2 = common.compute_init("cuda")
    assert out2[0] is True
    assert out2[-1].type == "cuda"
    assert matmul_calls["x"] == "high"
    assert seed_calls["x"] == 42
    assert cuda_seed_calls["x"] == 42
    assert "device_id" in init_calls["kw"]
    assert barrier_calls["n"] == 1
    assert set_device_calls["device"].type == "cuda"


def test_compute_init_assertions(monkeypatch):
    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: False)
    with pytest.raises(AssertionError):
        common.compute_init("cuda")
    monkeypatch.setattr(common.torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(AssertionError):
        common.compute_init("mps")
    with pytest.raises(AssertionError):
        common.compute_init("bad")


def test_dummy_wandb_and_peak_flops(monkeypatch):
    w = common.DummyWandb()
    assert w.log() is None
    assert w.finish() is None

    assert common.get_peak_flops("NVIDIA H100 NVL") == 835e12
    assert common.get_peak_flops("A100-SXM") == 312e12

    class Props:
        max_compute_units = 8

    class XPU:
        @staticmethod
        def get_device_properties(_name):
            return Props()

    monkeypatch.setattr(common.torch, "xpu", XPU())
    pvc = common.get_peak_flops("Data Center GPU Max 1550")
    assert pvc == 512 * 8 * 1300 * 10**6

    warned = {}
    monkeypatch.setattr(common.logger, "warning", lambda msg: warned.__setitem__("msg", msg))
    unknown = common.get_peak_flops("mystery gpu")
    assert unknown == float("inf")
    assert "Peak flops undefined" in warned["msg"]
