from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def package_dir(root: Path) -> Path:
    return root / "swift" / "NanochatMLXStub"


def build_products_dir(root: Path) -> Path:
    return root / "swift" / "Build" / "Products" / "Debug"


def stub_binary_path(root: Path) -> Path:
    return build_products_dir(root) / "nanochat-mlx-stub"


def bundle_path(root: Path) -> Path:
    return build_products_dir(root) / "mlx-swift_Cmlx.bundle"


def resolve_repo_path(root: Path, candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return root / path


def ensure_stub_is_built(root: Path, *, rebuild: bool) -> None:
    binary = stub_binary_path(root)
    bundle = bundle_path(root)
    if not rebuild and binary.exists() and bundle.exists():
        return

    command = [
        "xcodebuild",
        "-scheme",
        "NanochatMLXStub",
        "-destination",
        "platform=macOS",
        "-derivedDataPath",
        ".derived",
        "build",
    ]
    subprocess.run(command, cwd=package_dir(root), check=True)


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
            if self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError("Swift worker pipes are unavailable")
            self._process.stdin.write(json.dumps(request) + "\n")
            self._process.stdin.flush()
            response_line = self._process.stdout.readline()
            if not response_line:
                stderr = self._process.stderr.read() if self._process.stderr is not None else ""
                raise RuntimeError(stderr.strip() or "Swift worker terminated unexpectedly")

        response = json.loads(response_line)
        if not response.get("ok", False):
            raise RuntimeError(response.get("error") or "Swift worker request failed")
        self.last_timing = response.get("timing")
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
