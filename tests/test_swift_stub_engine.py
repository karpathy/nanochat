import io
import json
import subprocess
from pathlib import Path

import pytest

from nanochat.swift_stub_engine import (
    SwiftStubEngine,
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
            raise subprocess.TimeoutExpired("fake", timeout)
        return 0

    def kill(self):
        self.killed = True


def write_manifest(path: Path):
    path.write_text(json.dumps({"config": {}, "export": {}}), encoding="utf-8")


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


def test_swift_stub_engine_handles_repeated_requests_and_updates_timing(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path)

    fake_process = FakeProcess(
        [
            '{"status":"ready"}\n',
            '{"ok":true,"generated_token_ids":[7,8],"timing":{"avg_decode":"28.4ms","tokens_decoded":"2"}}\n',
            '{"ok":true,"generated_token_ids":[9],"timing":{"avg_decode":"29.1ms","tokens_decoded":"1"}}\n',
        ]
    )

    monkeypatch.setattr("nanochat.swift_stub_engine.ensure_stub_is_built", lambda root, rebuild: None)
    monkeypatch.setattr("nanochat.swift_stub_engine.subprocess.Popen", lambda *args, **kwargs: fake_process)

    engine = SwiftStubEngine(FakeTokenizer(), str(manifest_path))
    first = list(engine.generate([1, 2, 3], max_tokens=2))
    second = list(engine.generate([4, 5], max_tokens=1))

    assert first == [([7], [1]), ([8], [1])]
    assert second == [([9], [1])]
    assert engine.last_timing == {"avg_decode": "29.1ms", "tokens_decoded": "1"}

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