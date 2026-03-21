from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from nanochat.tokenizer import get_tokenizer
from nanochat.swift_build import (
    build_products_dir,
    bundle_path,
    ensure_stub_is_built,
    package_dir,
    stub_binary_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize a raw prompt and invoke the nanochat MLX Swift stub")
    parser.add_argument(
        "--manifest",
        type=str,
        default="runs/mlx_exports/phase2_d4_l_mps_step20.json",
        help="Path to the exported MLX sidecar manifest, relative to the repo root by default",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Raw text prompt to tokenize and pass to the Swift stub",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["cpu", "gpu"],
        help="Execution device for the Swift stub",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Number of greedy tokens to generate in the Swift stub",
    )
    parser.add_argument(
        "--no-bos",
        action="store_true",
        help="Do not prepend the tokenizer BOS token",
    )
    parser.add_argument(
        "--no-stop-tokens",
        action="store_true",
        help="Do not stop generation on BOS or assistant_end tokens",
    )
    parser.add_argument(
        "--print-token-ids",
        action="store_true",
        help="Print the token ids that will be sent to the Swift stub before invoking it",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the Swift package with xcodebuild before invoking the stub",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(root: Path, candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return root / path


def build_prompt_tokens(prompt: str, *, prepend_bos: bool) -> list[int]:
    tokenizer = get_tokenizer()
    bos_token_id = tokenizer.get_bos_token_id()
    return tokenizer.encode(prompt, prepend=None if not prepend_bos else bos_token_id)


def default_stop_token_ids() -> list[int]:
    tokenizer = get_tokenizer()
    return [
        tokenizer.get_bos_token_id(),
        tokenizer.encode_special("<|assistant_end|>"),
    ]


def decode_tokens(tokens: list[int]) -> str:
    tokenizer = get_tokenizer()
    return tokenizer.decode(tokens)


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
        result = {}
        for pair in payload.split():
            key, _, val = pair.partition("=")
            if key:
                result[key] = val
        return result
    return None


def invoke_stub(root: Path, *, manifest: Path, prompt_tokens: list[int], device: str, max_new_tokens: int, stop_token_ids: list[int]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["DYLD_FRAMEWORK_PATH"] = str(build_products_dir(root))
    token_arg = ",".join(str(token) for token in prompt_tokens)
    command = [
        str(stub_binary_path(root)),
        "--manifest",
        str(manifest),
        "--prompt-tokens",
        token_arg,
        "--max-new-tokens",
        str(max_new_tokens),
        "--device",
        device,
    ]
    if stop_token_ids:
        command.extend(["--stop-token-ids", ",".join(str(token) for token in stop_token_ids)])
    return subprocess.run(command, cwd=root, env=env, text=True, capture_output=True)


def main() -> int:
    args = parse_args()
    root = repo_root()
    manifest = resolve_repo_path(root, args.manifest)
    if not manifest.exists():
        print(f"Manifest not found: {manifest}", file=sys.stderr)
        return 1

    prompt_tokens = build_prompt_tokens(args.prompt, prepend_bos=not args.no_bos)
    if not prompt_tokens:
        print("Prompt tokenization produced no tokens", file=sys.stderr)
        return 1

    stop_token_ids = [] if args.no_stop_tokens else default_stop_token_ids()

    if args.print_token_ids:
        print("Prompt token ids:", ",".join(str(token) for token in prompt_tokens))

    ensure_stub_is_built(root, rebuild=args.rebuild)
    completed = invoke_stub(
        root,
        manifest=manifest,
        prompt_tokens=prompt_tokens,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=stop_token_ids,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        return completed.returncode

    generated_tokens = parse_generated_tokens(completed.stdout)
    print("Generated text:", decode_tokens(generated_tokens))
    timing = parse_timing(completed.stdout)
    if timing:
        print(f"Timing: device={timing.get('device', '?')} "
              f"load={timing.get('load', '?')} "
              f"prefill={timing.get('prefill', '?')} "
              f"avg_decode={timing.get('avg_decode', '?')} "
              f"tokens_decoded={timing.get('tokens_decoded', '?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())