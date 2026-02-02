#!/usr/bin/env python3
"""Prepare safety eval prompts from CoCoNot and Arena, with training dedup."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    from nanochat.common import get_base_dir
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Failed to import nanochat.common: {exc}")

try:
    from datasets import load_dataset
except ModuleNotFoundError:
    raise SystemExit(
        "Missing dependency: datasets. Install with `uv sync` or `pip install datasets`."
    )
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    raise SystemExit("Missing dependency: tqdm. Install with `uv sync` or `pip install tqdm`.")

DEFAULT_CATEGORY = "Requests with safety concerns"


def _default_output_dir() -> str:
    return str(Path(__file__).resolve().parents[3] / "safety_eval_data")


def _default_train_paths() -> list[str]:
    base_dir = get_base_dir()
    candidates = [
        os.path.join(base_dir, "safety_response_curation", "step4_sft.jsonl"),
        os.path.join(base_dir, "safety_response_curation", "step4_selected_metadata.jsonl"),
    ]
    return [path for path in candidates if os.path.exists(path)]


def _normalize_prompt(text: str) -> str:
    return " ".join(text.strip().split())


def _iter_splits(dataset_dict, include_splits: list[str]) -> Iterable[tuple[str, object]]:
    if include_splits:
        for split in include_splits:
            if split not in dataset_dict:
                raise SystemExit(
                    f"Split '{split}' not found in dataset. Available: {list(dataset_dict.keys())}"
                )
            yield split, dataset_dict[split]
    else:
        for split, ds in dataset_dict.items():
            yield split, ds


def _load_training_prompts(paths: list[str]) -> set[str]:
    prompts: set[str] = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"Training file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = _extract_prompt(obj)
                if prompt:
                    prompts.add(_normalize_prompt(prompt))
    return prompts


def _extract_prompt(obj) -> str | None:
    if isinstance(obj, dict):
        if "prompt" in obj and isinstance(obj["prompt"], str):
            return obj["prompt"]
        if "messages" in obj and isinstance(obj["messages"], list):
            for message in obj["messages"]:
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
        if "conversation" in obj and isinstance(obj["conversation"], list):
            for message in obj["conversation"]:
                if isinstance(message, dict) and message.get("role") == "user":
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
    if isinstance(obj, list):
        for message in obj:
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content
    return None


def _sample_records(records: list[dict], sample_size: int, seed: int) -> list[dict]:
    if not records:
        raise SystemExit("No eligible records found after filtering.")
    rng = random.Random(seed)
    if sample_size >= len(records):
        print(f"Warning: only {len(records)} records available, sampling all.")
        return records
    return rng.sample(records, k=sample_size)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare safety eval prompts")
    parser.add_argument("--coconot-subset", type=str, default="original")
    parser.add_argument("--coconot-category", type=str, default=DEFAULT_CATEGORY)
    parser.add_argument("--coconot-splits", type=str, default="")
    parser.add_argument("--coconot-sample-size", type=int, default=50)
    parser.add_argument("--arena-sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output-dir", type=str, default=_default_output_dir())
    parser.add_argument("--coconot-output", type=str, default="coconot_eval_50.jsonl")
    parser.add_argument("--arena-output", type=str, default="arena_eval_50.jsonl")
    parser.add_argument(
        "--train-jsonl",
        action="append",
        default=[],
        help="Training JSONL(s) to deduplicate against",
    )
    parser.add_argument(
        "--allow-missing-train",
        action="store_true",
        help="Allow running without training JSONL present",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_paths = list(args.train_jsonl)
    if not train_paths:
        train_paths = _default_train_paths()

    if not train_paths:
        if args.allow_missing_train:
            print("Warning: no training JSONL provided; skipping train dedup.")
        else:
            raise SystemExit(
                "No training JSONL provided. Use --train-jsonl or --allow-missing-train to proceed."
            )

    train_prompts = _load_training_prompts(train_paths) if train_paths else set()

    coconot = load_dataset("allenai/coconot", args.coconot_subset)
    category_key = " ".join(args.coconot_category.strip().lower().split())
    coconot_records: list[dict] = []
    coconot_splits = [s.strip() for s in args.coconot_splits.split(",") if s.strip()]
    for split_name, split_ds in _iter_splits(coconot, coconot_splits):
        for row in tqdm(split_ds, total=len(split_ds), desc=f"scan coconot {split_name}"):
            category = " ".join(str(row.get("category", "")).strip().lower().split())
            if category != category_key:
                continue
            prompt = row.get("prompt") or row.get("query") or row.get("instruction")
            if not prompt:
                continue
            normalized = _normalize_prompt(prompt)
            if normalized in train_prompts:
                continue
            coconot_records.append(
                {
                    "id": row.get("id"),
                    "prompt": prompt,
                    "category": row.get("category"),
                    "subcategory": row.get("subcategory"),
                    "source_split": split_name,
                    "dataset": "coconot",
                }
            )

    coconot_sample = _sample_records(coconot_records, args.coconot_sample_size, args.seed)
    _write_jsonl(output_dir / args.coconot_output, coconot_sample)
    print(f"Wrote {len(coconot_sample)} CoCoNot prompts to {output_dir / args.coconot_output}")

    arena = load_dataset("lmarena-ai/arena-human-preference-55k")
    arena_records: list[dict] = []
    arena_seen: set[str] = set()
    for split_name, split_ds in _iter_splits(arena, []):
        for idx, row in enumerate(tqdm(split_ds, total=len(split_ds), desc=f"scan arena {split_name}")):
            prompt = row.get("prompt")
            if not prompt:
                continue
            normalized = _normalize_prompt(prompt)
            if normalized in train_prompts:
                continue
            if normalized in arena_seen:
                continue
            arena_seen.add(normalized)
            arena_records.append(
                {
                    "id": row.get("id") or f"arena_{split_name}_{idx}",
                    "prompt": prompt,
                    "source_split": split_name,
                    "dataset": "arena_human_preference",
                }
            )

    arena_sample = _sample_records(arena_records, args.arena_sample_size, args.seed + 1)
    _write_jsonl(output_dir / args.arena_output, arena_sample)
    print(f"Wrote {len(arena_sample)} Arena prompts to {output_dir / args.arena_output}")


if __name__ == "__main__":
    main()
