#!/usr/bin/env python3
"""Generate responses for safety prompts using API."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    from nanochat.common import get_base_dir
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Failed to import nanochat.common: {exc}")

try:
    from openai import OpenAI
except ModuleNotFoundError:
    raise SystemExit("Missing dependency: openai. Install with `pip install openai` or `uv sync`.")
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    raise SystemExit("Missing dependency: tqdm. Install with `uv sync` or `pip install tqdm`.")

SYSTEM_MESSAGE = (
    "You are a helpful assistant. If the user's request is safe and allowed, answer it. "
    "If the request is unsafe or harmful, refuse to comply and give a brief explanation, "
    "then offer a safer alternative or positive guidance."
)


def _default_output_dir() -> str:
    if os.environ.get("SAFETY_CURATION_DIR"):
        return os.environ["SAFETY_CURATION_DIR"]
    return os.path.join(get_base_dir(), "safety_response_curation")


def _load_existing(output_path: Path) -> set[Tuple[str, int]]:
    existing: set[Tuple[str, int]] = set()
    if not output_path.exists():
        return existing
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = str(obj.get("id"))
            gen_id = obj.get("generation_id")
            if record_id and isinstance(gen_id, int):
                existing.add((record_id, gen_id))
    return existing


def _iter_requests(input_path: Path, n_responses: int, skip: set[Tuple[str, int]]):
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            record_id = str(row.get("id"))
            for gen_id in range(n_responses):
                if (record_id, gen_id) in skip:
                    continue
                yield row, gen_id


def _make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def _call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    retries: int,
    retry_backoff: float,
) -> str:
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=False,
            )
            content = completion.choices[0].message.content
            return content or ""
        except Exception as exc:  # pragma: no cover - network errors
            last_err = exc
            if attempt < retries:
                time.sleep(retry_backoff * (2 ** attempt))
            else:
                break
    raise RuntimeError(f"Model call failed after {retries + 1} attempts: {last_err}")


def _build_record(row: Dict, gen_id: int, response: str, model: str, temperature: float, top_p: float, max_tokens: int):
    return {
        "id": row.get("id"),
        "prompt": row.get("prompt"),
        "category": row.get("category"),
        "subcategory": row.get("subcategory"),
        "source_split": row.get("source_split"),
        "generation_id": gen_id,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "response": response,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate responses for safety prompts")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL from step1")
    parser.add_argument("--output-dir", type=str, default=_default_output_dir(), help="Output directory")
    parser.add_argument(
        "--output-name",
        type=str,
        default="step2_generations.jsonl",
        help="Output JSONL filename",
    )
    parser.add_argument("--model", type=str, default="openai/gpt-oss-120b", help="Model name")
    parser.add_argument("--n-responses", type=int, default=2, help="Responses per prompt")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p")
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens")
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("BASE_URL", "https://integrate.api.nvidia.com/v1"),
        help="OpenAI-compatible base URL",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="API_KEY",
        help="Env var holding API key",
    )
    parser.add_argument("--retries", type=int, default=2, help="Retry count")
    parser.add_argument("--retry-backoff", type=float, default=1.5, help="Retry backoff seconds")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    skip = _load_existing(output_path)
    requests_iter = list(_iter_requests(Path(args.input), args.n_responses, skip))
    if not requests_iter:
        print("No new requests to process.")
        return

    client = _make_client(args.base_url, api_key)

    def worker(item):
        row, gen_id = item
        response = _call_model(
            client,
            args.model,
            row.get("prompt", ""),
            args.temperature,
            args.top_p,
            args.max_tokens,
            args.retries,
            args.retry_backoff,
        )
        return _build_record(row, gen_id, response, args.model, args.temperature, args.top_p, args.max_tokens)

    with output_path.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(worker, item): item for item in requests_iter}
            with tqdm(total=len(futures), desc="generate") as progress:
                for future in as_completed(futures):
                    row, gen_id = futures[future]
                    try:
                        record = future.result()
                        f.write(json.dumps(record, ensure_ascii=True) + "\n")
                    except Exception as exc:
                        error_record = {
                            "id": row.get("id"),
                            "prompt": row.get("prompt"),
                            "category": row.get("category"),
                            "subcategory": row.get("subcategory"),
                            "source_split": row.get("source_split"),
                            "generation_id": gen_id,
                            "model": args.model,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "max_tokens": args.max_tokens,
                            "response": None,
                            "error": str(exc),
                            "created_at": datetime.utcnow().isoformat() + "Z",
                        }
                        f.write(json.dumps(error_record, ensure_ascii=True) + "\n")
                    f.flush()
                    progress.update(1)

    print(f"Wrote generations to {output_path}")


if __name__ == "__main__":
    main()
