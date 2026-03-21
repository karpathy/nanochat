import io
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.modules.setdefault("rustbpe", types.ModuleType("rustbpe"))

from nanochat.swift_stub_engine import (
    SwiftStubEngine,
    build_swift_request_telemetry,
    choose_swift_backend,
    ensure_stub_is_built,
    parse_timing,
    resolve_preferred_manifest,
    swift_decode_supported,
)


class FakeTokenizer:
    def get_bos_token_id(self):
        return 42

    def encode_special(self, token):
        mapping = {"<|assistant_end|>": 99}
        return mapping[token]


class FakeStdout:
    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        if not self._lines:
            return ""
        return self._lines.pop(0)


class FakeProcess:
    def __init__(self, stdout_lines, *, wait_timeout=False):
        self.stdin = io.StringIO()
        self.stdout = FakeStdout(stdout_lines)
        self.stderr = io.StringIO("")
        self.terminated = False
        self.killed = False
        self.wait_timeout = wait_timeout

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        if self.wait_timeout:
            raise subprocess.TimeoutExpired("fake", 0 if timeout is None else timeout)
        return 0

    def kill(self):
        self.killed = True


def write_manifest(path: Path):
    path.write_text(json.dumps({"config": {}, "export": {}}), encoding="utf-8")


def write_file(path: Path, content: str, *, mtime: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, (mtime, mtime))


def test_parse_timing_extracts_key_value_pairs():
    stdout = "Timing: device=gpu load=10ms avg_decode=28.4ms tokens_decoded=32\n"
    assert parse_timing(stdout) == {
        "device": "gpu",
        "load": "10ms",
        "avg_decode": "28.4ms",
        "tokens_decoded": "32",
    }


def test_swift_decode_supported_is_greedy_only():
    assert swift_decode_supported(temperature=0.0, top_k=0)
    assert swift_decode_supported(temperature=0, top_k=None)
    assert not swift_decode_supported(temperature=0.6, top_k=50)


def test_choose_swift_backend_auto_routes_only_long_greedy_requests(tmp_path, monkeypatch):
    manifest_path = tmp_path / "mlx.json"
    write_manifest(manifest_path)

    monkeypatch.setattr(
        "nanochat.swift_stub_engine.resolve_preferred_manifest",
        lambda *args, **kwargs: manifest_path,
    )

    short = choose_swift_backend(
        tmp_path,
        source="base",
        model_tag=None,
        step=None,
        explicit_manifest_path=None,
        allow_auto=True,
        temperature=0.0,
        top_k=0,
        max_tokens=32,
        min_output_tokens=64,
    )
    long = choose_swift_backend(
        tmp_path,
        source="base",
        model_tag=None,
        step=None,
        explicit_manifest_path=None,
        allow_auto=True,
        temperature=0.0,
        top_k=0,
        max_tokens=128,
        min_output_tokens=64,
    )

    assert not short.use_swift
    assert short.reason_code == "pytorch_swift_short_output"
    assert long.use_swift
    assert long.reason_code == "swift_auto_long_output"
    assert long.manifest_path == str(manifest_path)


def test_choose_swift_backend_explicit_manifest_overrides_threshold_but_not_compatibility(tmp_path):
    manifest_path = tmp_path / "mlx.json"
    write_manifest(manifest_path)

    explicit = choose_swift_backend(
        tmp_path,
        source="base",
        model_tag=None,
        step=None,
        explicit_manifest_path=str(manifest_path),
        allow_auto=True,
        temperature=0.0,
        top_k=0,
        max_tokens=8,
        min_output_tokens=64,
    )
    incompatible = choose_swift_backend(
        tmp_path,
        source="base",
        model_tag=None,
        step=None,
        explicit_manifest_path=str(manifest_path),
        allow_auto=True,
        temperature=0.7,
        top_k=50,
        max_tokens=128,
        min_output_tokens=64,
    )

    assert explicit.use_swift
    assert explicit.reason_code == "swift_explicit_manifest_override"
    assert not incompatible.use_swift
    assert incompatible.reason_code == "pytorch_swift_incompatible_sampling"


def test_build_swift_request_telemetry_normalizes_memory_and_reuse():
    telemetry = build_swift_request_telemetry(
        {
            "device": "gpu",
            "load": "10.5ms",
            "ttft": "41.0ms",
            "avg_decode": "28.4ms",
            "tokens_decoded": "32",
            "active_memory_gb": "12.500",
            "peak_memory_gb": "14.250",
            "cache_memory_gb": "3.125",
        },
        worker_reuse_count=2,
    )

    assert telemetry == {
        "device": "gpu",
        "load_ms": 10.5,
        "ttft_ms": 41.0,
        "decode_ms_per_token": 28.4,
        "tokens_decoded": 32,
        "active_memory_gb": 12.5,
        "peak_memory_gb": 14.25,
        "cache_memory_gb": 3.125,
        "persistent_worker_reuse_count": 2,
    }


def test_resolve_preferred_manifest_uses_largest_model_and_latest_step(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    exports = repo_root / "runs" / "mlx_exports"
    exports.mkdir(parents=True)
    manifest_path = exports / "mlx_base_d8_step10.json"
    write_manifest(manifest_path)

    base_dir = tmp_path / "cache"
    checkpoint_root = base_dir / "base_checkpoints"
    (checkpoint_root / "d4").mkdir(parents=True)
    (checkpoint_root / "d8").mkdir(parents=True)
    (checkpoint_root / "d4" / "model_000020.pt").write_text("", encoding="utf-8")
    (checkpoint_root / "d8" / "model_000010.pt").write_text("", encoding="utf-8")

    monkeypatch.setattr("nanochat.swift_stub_engine.get_base_dir", lambda: str(base_dir))

    resolved = resolve_preferred_manifest(repo_root, source="base", model_tag=None, step=None)
    assert resolved == manifest_path


def test_ensure_stub_is_built_skips_clean_build_when_outputs_are_fresh(tmp_path, monkeypatch):
    write_file(tmp_path / "swift" / "NanochatMLXStub" / "Package.swift", "", mtime=100)
    write_file(tmp_path / "swift" / "NanochatMLXStub" / "Package.resolved", "", mtime=100)
    write_file(tmp_path / "swift" / "NanochatMLXStub" / "Sources" / "NanochatMLXStub" / "main.swift", "", mtime=100)
    write_file(tmp_path / "swift" / "Build" / "Products" / "Debug" / "nanochat-mlx-stub", "", mtime=200)
    write_file(tmp_path / "swift" / "Build" / "Products" / "Debug" / "mlx-swift_Cmlx.bundle", "", mtime=200)

    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr("nanochat.swift_build.subprocess.run", fake_run)

    ensure_stub_is_built(tmp_path, rebuild=False)

    assert calls == []


def test_ensure_stub_is_built_forces_clean_build_when_sources_are_newer(tmp_path, monkeypatch):
    write_file(tmp_path / "swift" / "NanochatMLXStub" / "Package.swift", "", mtime=100)
    write_file(tmp_path / "swift" / "NanochatMLXStub" / "Package.resolved", "", mtime=100)
    write_file(tmp_path / "swift" / "NanochatMLXStub" / "Sources" / "NanochatMLXStub" / "main.swift", "", mtime=300)
    write_file(tmp_path / "swift" / "Build" / "Products" / "Debug" / "nanochat-mlx-stub", "", mtime=200)
    write_file(tmp_path / "swift" / "Build" / "Products" / "Debug" / "mlx-swift_Cmlx.bundle", "", mtime=200)

    calls = []

    def fake_run(command, *, cwd, check):
        calls.append((command, cwd, check))

    monkeypatch.setattr("nanochat.swift_build.subprocess.run", fake_run)

    ensure_stub_is_built(tmp_path, rebuild=False)

    assert calls == [
        (
            [
                "xcodebuild",
                "-scheme",
                "NanochatMLXStub",
                "-destination",
                "platform=macOS",
                "-derivedDataPath",
                ".derived",
                "clean",
                "build",
            ],
            tmp_path / "swift" / "NanochatMLXStub",
            True,
        )
    ]


def test_swift_stub_engine_handles_repeated_requests_and_updates_timing(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path)

    fake_process = FakeProcess(
        [
            '{"status":"ready"}\n',
            '{"ok":true,"generated_token_ids":[7,8],"timing":{"ttft":"40.0ms","avg_decode":"28.4ms","tokens_decoded":"2","active_memory_gb":"12.500","peak_memory_gb":"14.250","cache_memory_gb":"3.125"}}\n',
            '{"ok":true,"generated_token_ids":[9],"timing":{"ttft":"39.5ms","avg_decode":"29.1ms","tokens_decoded":"1","active_memory_gb":"11.750","peak_memory_gb":"13.000","cache_memory_gb":"2.875"}}\n',
        ]
    )

    monkeypatch.setattr("nanochat.swift_stub_engine.ensure_stub_is_built", lambda root, rebuild: None)
    monkeypatch.setattr("nanochat.swift_stub_engine.subprocess.Popen", lambda *args, **kwargs: fake_process)

    engine = SwiftStubEngine(FakeTokenizer(), str(manifest_path))
    first = list(engine.generate([1, 2, 3], max_tokens=2))
    second = list(engine.generate([4, 5], max_tokens=1))

    assert first == [([7], [1]), ([8], [1])]
    assert second == [([9], [1])]
    assert engine.last_timing == {
        "ttft": "39.5ms",
        "avg_decode": "29.1ms",
        "tokens_decoded": "1",
        "active_memory_gb": "11.750",
        "peak_memory_gb": "13.000",
        "cache_memory_gb": "2.875",
    }
    assert engine.last_request_telemetry == {
        "device": None,
        "load_ms": None,
        "ttft_ms": 39.5,
        "decode_ms_per_token": 29.1,
        "tokens_decoded": 1,
        "active_memory_gb": 11.75,
        "peak_memory_gb": 13.0,
        "cache_memory_gb": 2.875,
        "persistent_worker_reuse_count": 1,
    }

    requests = [json.loads(line) for line in fake_process.stdin.getvalue().splitlines()]
    assert requests[0]["prompt_tokens"] == [1, 2, 3]
    assert requests[1]["prompt_tokens"] == [4, 5]

    engine.close()
    assert fake_process.terminated


def test_swift_stub_engine_kills_process_after_wait_timeout(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path)

    fake_process = FakeProcess(['{"status":"ready"}\n'], wait_timeout=True)

    monkeypatch.setattr("nanochat.swift_stub_engine.ensure_stub_is_built", lambda root, rebuild: None)
    monkeypatch.setattr("nanochat.swift_stub_engine.subprocess.Popen", lambda *args, **kwargs: fake_process)

    engine = SwiftStubEngine(FakeTokenizer(), str(manifest_path))
    engine.close()

    assert fake_process.terminated
    assert fake_process.killed