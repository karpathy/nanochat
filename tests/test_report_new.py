import datetime
import runpy
import os
import types

import pytest

import nanochat.report as report


def test_run_command_paths(monkeypatch):
    class R:
        def __init__(self, stdout, returncode):
            self.stdout = stdout
            self.returncode = returncode

    monkeypatch.setattr(report.subprocess, "run", lambda *a, **k: R("x\n", 1))
    assert report.run_command("x") == "x"

    monkeypatch.setattr(report.subprocess, "run", lambda *a, **k: R("  ", 0))
    assert report.run_command("x") == ""

    monkeypatch.setattr(report.subprocess, "run", lambda *a, **k: R("", 2))
    assert report.run_command("x") is None

    def boom(*a, **k):
        raise RuntimeError("fail")

    monkeypatch.setattr(report.subprocess, "run", boom)
    assert report.run_command("x") is None


def test_git_gpu_system_cost_helpers(monkeypatch):
    seq = iter(["abc123", "main", " M x", "subject line\nbody"])
    monkeypatch.setattr(report, "run_command", lambda _cmd: next(seq))
    git = report.get_git_info()
    assert git["commit"] == "abc123"
    assert git["branch"] == "main"
    assert git["dirty"] is True
    assert git["message"] == "subject line"

    monkeypatch.setattr(report.torch.cuda, "is_available", lambda: False)
    assert report.get_gpu_info() == {"available": False}

    class Props:
        def __init__(self, name, total_memory):
            self.name = name
            self.total_memory = total_memory

    monkeypatch.setattr(report.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(report.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(report.torch.cuda, "get_device_properties", lambda i: Props(f"GPU{i}", 8 * 1024**3))
    monkeypatch.setattr(report.torch, "version", types.SimpleNamespace(cuda="12.4"))
    g = report.get_gpu_info()
    assert g["available"] is True
    assert g["count"] == 2
    assert g["cuda_version"] == "12.4"

    monkeypatch.setattr(report.socket, "gethostname", lambda: "host")
    monkeypatch.setattr(report.platform, "system", lambda: "Linux")
    monkeypatch.setattr(report.platform, "python_version", lambda: "3.10.0")
    monkeypatch.setattr(report.torch, "__version__", "2.x")
    monkeypatch.setattr(report.psutil, "cpu_count", lambda logical=False: 8 if not logical else 16)
    monkeypatch.setattr(report.psutil, "virtual_memory", lambda: types.SimpleNamespace(total=32 * 1024**3))
    monkeypatch.setenv("USER", "alice")
    monkeypatch.setenv("NANOCHAT_BASE_DIR", "/n")
    monkeypatch.setattr(report.os, "getcwd", lambda: "/cwd")
    s = report.get_system_info()
    assert s["hostname"] == "host"
    assert s["cpu_count_logical"] == 16
    assert s["nanochat_base_dir"] == "/n"

    assert report.estimate_cost({"available": False}) is None
    c1 = report.estimate_cost({"available": True, "count": 2, "names": ["H100"], "memory_gb": [80]}, runtime_hours=3)
    assert c1["hourly_rate"] == 6.0
    assert c1["estimated_total"] == 18.0
    c2 = report.estimate_cost({"available": True, "count": 3, "names": ["Unknown"], "memory_gb": [1]})
    assert c2["hourly_rate"] == 6.0


def test_generate_header(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "uv.lock").write_text("a\nb\n", encoding="utf-8")
    monkeypatch.setattr(report, "get_git_info", lambda: {"branch": "b", "commit": "c", "dirty": False, "message": "m"})
    monkeypatch.setattr(report, "get_gpu_info", lambda: {"available": True, "count": 1, "names": ["A100"], "memory_gb": [40], "cuda_version": "12"})
    monkeypatch.setattr(report, "get_system_info", lambda: {"platform": "Linux", "cpu_count": 8, "cpu_count_logical": 16, "memory_gb": 31.5, "python_version": "3.10", "torch_version": "2.9"})
    monkeypatch.setattr(report, "estimate_cost", lambda _gpu: {"hourly_rate": 1.23, "gpu_type": "A100", "estimated_total": None})

    def fake_run(cmd):
        if "git ls-files" in cmd and "| xargs" not in cmd:
            return "a.py\nb.md"
        if "| xargs wc -lc" in cmd:
            return "  10  100 total"
        return ""

    monkeypatch.setattr(report, "run_command", fake_run)
    h = report.generate_header()
    assert "# nanochat training report" in h
    assert "Branch: b" in h
    assert "Hourly Rate: $1.23/hour" in h
    assert "Dependencies (uv.lock lines): 2" in h

    # No GPU and no uv.lock branch.
    os.remove(tmp_path / "uv.lock")
    monkeypatch.setattr(report, "get_gpu_info", lambda: {"available": False})
    monkeypatch.setattr(report, "estimate_cost", lambda _gpu: None)
    h2 = report.generate_header()
    assert "GPUs: None available" in h2


def test_slug_extract_timestamp_helpers():
    assert report.slugify("Hello World") == "hello-world"
    section = "a: 1\nb: 2\nc: 3"
    assert report.extract(section, "b") == {"b": "2"}
    assert report.extract(section, ["a", "c"]) == {"a": "1", "c": "3"}

    content = "x\ntimestamp: 2026-01-02 03:04:05\nz"
    dt = report.extract_timestamp(content, "timestamp:")
    assert dt == datetime.datetime(2026, 1, 2, 3, 4, 5)
    assert report.extract_timestamp("timestamp: not-a-time", "timestamp:") is None
    assert report.extract_timestamp("bad", "timestamp:") is None


def test_report_log_generate_reset_and_get_report(monkeypatch, tmp_path):
    r = report.Report(str(tmp_path))
    p = r.log("My Section", [{"i": 12345, "f": 1.23456, "s": "x"}, "", None, "plain\n"])
    text = (tmp_path / "my-section.md").read_text(encoding="utf-8")
    assert p.endswith("my-section.md")
    assert "- i: 12,345" in text
    assert "- f: 1.2346" in text
    assert "plain" in text

    # Prepare files for generate().
    header = tmp_path / "header.md"
    header.write_text(
        "Run started: 2026-01-01 00:00:00\n\n### Bloat\n- Lines: 10\n\n",
        encoding="utf-8",
    )
    (tmp_path / "base-model-evaluation.md").write_text(
        "timestamp: 2026-01-01 01:00:00\nCORE: 0.5\n",
        encoding="utf-8",
    )
    (tmp_path / "chat-evaluation-sft.md").write_text(
        "timestamp: 2026-01-01 02:00:00\nARC-Easy: 0.7\nChatCORE: 0.8\n",
        encoding="utf-8",
    )
    (tmp_path / "chat-evaluation-rl.md").write_text(
        "timestamp: 2026-01-01 03:00:00\nGSM8K: 0.9\n",
        encoding="utf-8",
    )
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    out = r.generate()
    rep = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert out.endswith("report.md")
    assert "## Summary" in rep
    assert "Total wall clock time: 2h0m" in rep
    assert "| CORE" in rep

    # Missing header path + reset.
    (tmp_path / "header.md").unlink()
    out2 = r.generate()
    rep2 = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert out2.endswith("report.md")
    assert "Total wall clock time: unknown" in rep2

    monkeypatch.setattr(report, "generate_header", lambda: "H\n")
    r.reset()
    assert (tmp_path / "header.md").exists()
    assert "Run started:" in (tmp_path / "header.md").read_text(encoding="utf-8")

    d = report.DummyReport()
    assert d.log() is None
    assert d.reset() is None

    # get_report() rank 0 and non-zero branches.
    monkeypatch.setattr("nanochat.common.get_dist_info", lambda: (True, 0, 0, 2))
    monkeypatch.setattr("nanochat.common.get_base_dir", lambda: str(tmp_path))
    got = report.get_report()
    assert isinstance(got, report.Report)

    monkeypatch.setattr("nanochat.common.get_dist_info", lambda: (True, 1, 1, 2))
    got2 = report.get_report()
    assert isinstance(got2, report.DummyReport)


def test_report_main_block_generate_and_reset(monkeypatch, tmp_path):
    class FakeParser:
        def __init__(self, *a, **k):
            pass

        def add_argument(self, *a, **k):
            return None

        def parse_args(self):
            return self.args

    parser = FakeParser()
    parser.args = types.SimpleNamespace(command="generate")
    monkeypatch.setattr("argparse.ArgumentParser", lambda *a, **k: parser)
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "base"))
    runpy.run_module("nanochat.report", run_name="__main__")

    parser.args = types.SimpleNamespace(command="reset")
    runpy.run_module("nanochat.report", run_name="__main__")
