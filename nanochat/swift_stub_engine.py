from __future__ import annotations

import json
import os
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from nanochat.checkpoint_manager import find_largest_model, find_last_step
from nanochat.common import get_base_dir
from nanochat.swift_build import (
    build_products_dir,
    bundle_path,
    ensure_stub_is_built,
    package_dir,
    stub_binary_path,
)


SWIFT_AUTO_MIN_OUTPUT_TOKENS = 48


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def exports_dir(root: Path) -> Path:
    return root / "runs" / "mlx_exports"


def resolve_repo_path(root: Path, candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return root / path


def is_valid_manifest_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, dict) and "config" in payload and "export" in payload


def swift_decode_supported(*, temperature: float, top_k: int | None) -> bool:
    return temperature in (0, 0.0) and top_k in (None, 0)


def _parse_duration_ms(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.endswith("ms"):
        stripped = stripped[:-2]
    try:
        return float(stripped)
    except ValueError:
        return None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def build_swift_request_telemetry(
    timing: Mapping[str, str] | None,
    *,
    worker_reuse_count: int,
) -> dict[str, float | int | str | None] | None:
    if timing is None:
        return None
    return {
        "device": timing.get("device"),
        "load_ms": _parse_duration_ms(timing.get("load")),
        "ttft_ms": _parse_duration_ms(timing.get("ttft") or timing.get("prefill")),
        "decode_ms_per_token": _parse_duration_ms(timing.get("avg_decode")),
        "tokens_decoded": _parse_int(timing.get("tokens_decoded")),
        "active_memory_gb": _parse_float(timing.get("active_memory_gb")),
        "peak_memory_gb": _parse_float(timing.get("peak_memory_gb")),
        "cache_memory_gb": _parse_float(timing.get("cache_memory_gb")),
        "persistent_worker_reuse_count": worker_reuse_count,
    }


@dataclass(frozen=True)
class SwiftRoutingDecision:
    use_swift: bool
    manifest_path: str | None
    backend: str
    reason_code: str
    reason_detail: str


def choose_swift_backend(
    root: Path,
    *,
    source: str,
    model_tag: str | None,
    step: int | None,
    explicit_manifest_path: str | None,
    allow_auto: bool,
    temperature: float,
    top_k: int | None,
    max_tokens: int,
    min_output_tokens: int = SWIFT_AUTO_MIN_OUTPUT_TOKENS,
) -> SwiftRoutingDecision:
    decode_supported = swift_decode_supported(temperature=temperature, top_k=top_k)

    if explicit_manifest_path is not None:
        if not decode_supported:
            return SwiftRoutingDecision(
                use_swift=False,
                manifest_path=None,
                backend="pytorch",
                reason_code="pytorch_swift_incompatible_sampling",
                reason_detail=(
                    "Explicit Swift manifest was ignored because the current Swift path only supports "
                    "greedy decoding with temperature=0 and top_k=0."
                ),
            )
        return SwiftRoutingDecision(
            use_swift=True,
            manifest_path=explicit_manifest_path,
            backend="swift",
            reason_code="swift_explicit_manifest_override",
            reason_detail="Using the explicitly requested Swift MLX export manifest.",
        )

    if not allow_auto:
        return SwiftRoutingDecision(
            use_swift=False,
            manifest_path=None,
            backend="pytorch",
            reason_code="pytorch_swift_auto_disabled",
            reason_detail="Automatic Swift MLX routing is disabled for this run.",
        )

    auto_manifest = resolve_preferred_manifest(
        root,
        source=source,
        model_tag=model_tag,
        step=step,
    )
    if auto_manifest is None:
        return SwiftRoutingDecision(
            use_swift=False,
            manifest_path=None,
            backend="pytorch",
            reason_code="pytorch_no_matching_swift_export",
            reason_detail="No matching MLX export manifest was found for automatic Swift routing.",
        )

    if not decode_supported:
        return SwiftRoutingDecision(
            use_swift=False,
            manifest_path=None,
            backend="pytorch",
            reason_code="pytorch_swift_incompatible_sampling",
            reason_detail=(
                f"Keeping the PyTorch engine because Swift auto-routing requires greedy decoding and this request uses "
                f"temperature={temperature} top_k={top_k}."
            ),
        )

    if max_tokens < min_output_tokens:
        return SwiftRoutingDecision(
            use_swift=False,
            manifest_path=None,
            backend="pytorch",
            reason_code="pytorch_swift_short_output",
            reason_detail=(
                f"Keeping the PyTorch engine because expected output length {max_tokens} is below the Swift auto-routing "
                f"threshold of {min_output_tokens} tokens."
            ),
        )

    return SwiftRoutingDecision(
        use_swift=True,
        manifest_path=str(auto_manifest),
        backend="swift",
        reason_code="swift_auto_long_output",
        reason_detail=(
            f"Using Swift MLX because greedy-compatible output length {max_tokens} meets the auto-routing threshold "
            f"of {min_output_tokens} tokens."
        ),
    )


def resolve_preferred_manifest(root: Path, *, source: str, model_tag: str | None, step: int | None) -> Path | None:
    checkpoint_dir_by_source = {
        "base": "base_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }
    source_dir = checkpoint_dir_by_source.get(source)
    if source_dir is None:
        return None

    base_dir = get_base_dir()
    if base_dir is None:
        return None

    checkpoints_dir = Path(base_dir) / source_dir
    if not checkpoints_dir.exists():
        return None

    try:
        resolved_model_tag = model_tag or find_largest_model(str(checkpoints_dir))
        checkpoint_dir = checkpoints_dir / resolved_model_tag
        resolved_step = step if step is not None else find_last_step(str(checkpoint_dir))
    except FileNotFoundError:
        return None

    candidate = exports_dir(root) / f"mlx_{source}_{resolved_model_tag}_step{resolved_step}.json"
    if is_valid_manifest_file(candidate):
        return candidate
    return None


def parse_generated_tokens(stdout: str) -> list[int]:
    prefix = "Generated token ids: "
    for line in stdout.splitlines():
        if not line.startswith(prefix):
            continue
        payload = line[len(prefix):].strip()
        if payload == "":
            return []
        return [int(token) for token in payload.split(",") if token]
    raise RuntimeError("Swift stub output did not include a generated token line")


def parse_timing(stdout: str) -> dict[str, str] | None:
    prefix = "Timing: "
    for line in stdout.splitlines():
        if not line.startswith(prefix):
            continue
        payload = line[len(prefix):].strip()
        result: dict[str, str] = {}
        for pair in payload.split():
            key, _, val = pair.partition("=")
            if key:
                result[key] = val
        return result
    return None


class SwiftStubEngine:
    def __init__(self, tokenizer, manifest_path: str, *, device: str = "gpu", rebuild: bool = False):
        self.tokenizer = tokenizer
        self.root = repo_root()
        self.manifest = resolve_repo_path(self.root, manifest_path)
        self.device = device
        self.rebuild = rebuild
        self.last_timing: dict[str, str] | None = None
        self.last_request_telemetry: dict[str, float | int | str | None] | None = None
        self.request_count = 0

        if not self.manifest.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest}")

        ensure_stub_is_built(self.root, rebuild=rebuild)
        self._lock = threading.Lock()
        self._process = self._start_worker_process()

    def _start_worker_process(self) -> subprocess.Popen[str]:
        env = os.environ.copy()
        env["DYLD_FRAMEWORK_PATH"] = str(build_products_dir(self.root))
        process = subprocess.Popen(
            [
                str(stub_binary_path(self.root)),
                "--manifest",
                str(self.manifest),
                "--device",
                self.device,
                "--serve-stdin",
                "--prompt-tokens",
                "0",
                "--max-new-tokens",
                "1",
            ],
            cwd=self.root,
            env=env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        ready_line = process.stdout.readline() if process.stdout is not None else ""
        if not ready_line:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise RuntimeError(stderr.strip() or "Swift worker failed to start")
        ready = json.loads(ready_line)
        if ready.get("status") != "ready":
            raise RuntimeError(f"Unexpected Swift worker handshake: {ready}")
        return process

    def _default_stop_token_ids(self) -> list[int]:
        return [
            self.tokenizer.get_bos_token_id(),
            self.tokenizer.encode_special("<|assistant_end|>"),
        ]

    def _invoke(self, prompt_tokens: list[int], max_new_tokens: int) -> list[int]:
        request = {
            "prompt_tokens": prompt_tokens,
            "max_new_tokens": max_new_tokens,
            "stop_token_ids": self._default_stop_token_ids(),
        }
        with self._lock:
            process = self._process
            if process is None:
                raise RuntimeError("Swift worker is not running")
            if process.stdin is None or process.stdout is None:
                raise RuntimeError("Swift worker pipes are unavailable")
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            response_line = process.stdout.readline()
            if not response_line:
                stderr = process.stderr.read() if process.stderr is not None else ""
                raise RuntimeError(stderr.strip() or "Swift worker terminated unexpectedly")

        response = json.loads(response_line)
        if not response.get("ok", False):
            raise RuntimeError(response.get("error") or "Swift worker request failed")
        self.last_timing = response.get("timing")
        self.request_count += 1
        self.last_request_telemetry = build_swift_request_telemetry(
            self.last_timing,
            worker_reuse_count=max(self.request_count - 1, 0),
        )
        return response.get("generated_token_ids", [])

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.stdin is not None:
            process.stdin.close()
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        self._process = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=0.0, top_k=None, seed=42):
        del seed
        if num_samples != 1:
            raise ValueError("SwiftStubEngine currently supports num_samples=1 only")
        if temperature not in (0, 0.0) or (top_k not in (None, 0)):
            raise ValueError("SwiftStubEngine currently supports greedy decoding only; use temperature=0 and top_k=0")
        if max_tokens is None or max_tokens < 1:
            raise ValueError("SwiftStubEngine requires max_tokens >= 1")

        generated_tokens = self._invoke(list(tokens), max_tokens)
        for token in generated_tokens:
            yield [token], [1]
