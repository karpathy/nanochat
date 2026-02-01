#!/usr/bin/env python3
"""Download CoCoNot (original) and sample safety-concern prompts."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

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
    if os.environ.get("SAFETY_CURATION_DIR"):
        return os.environ["SAFETY_CURATION_DIR"]
    return os.path.join(get_base_dir(), "safety_response_curation")


def _normalize_category(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _iter_splits(dataset_dict, include_splits):
    if include_splits:
        for split in include_splits:
            if split not in dataset_dict:
                raise SystemExit(f"Split '{split}' not found in dataset. Available: {list(dataset_dict.keys())}")
            yield split, dataset_dict[split]
    else:
        for split, ds in dataset_dict.items():
            yield split, ds


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample safety prompts from CoCoNot (original)")
    parser.add_argument("--subset", type=str, default="original", help="HF subset (default: original)")
    parser.add_argument("--category", type=str, default=DEFAULT_CATEGORY, help="Category to filter")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of prompts to sample")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument("--splits", type=str, default="", help="Comma-separated splits (default: all)")
    parser.add_argument("--output-dir", type=str, default=_default_output_dir(), help="Output directory")
    parser.add_argument(
        "--output-name",
        type=str,
        default="step1_coconot_safety_sample.jsonl",
        help="Output JSONL filename",
    )
    args = parser.parse_args()

    include_splits = [s.strip() for s in args.splits.split(",") if s.strip()] if args.splits else []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / args.output_name

    dataset = load_dataset("allenai/coconot", args.subset)
    target_category = _normalize_category(args.category)

    records = []
    for split_name, split_ds in _iter_splits(dataset, include_splits):
        for row in tqdm(split_ds, total=len(split_ds), desc=f"scan {split_name}"):
            category = _normalize_category(str(row.get("category", "")))
            if category != target_category:
                continue
            prompt = row.get("prompt") or row.get("query") or row.get("instruction")
            if not prompt:
                continue
            records.append(
                {
                    "id": row.get("id"),
                    "prompt": prompt,
                    "category": row.get("category"),
                    "subcategory": row.get("subcategory"),
                    "source_split": split_name,
                }
            )

    if not records:
        raise SystemExit("No records matched the requested category.")

    rng = random.Random(args.seed)
    sample_size = min(args.sample_size, len(records))
    if sample_size < args.sample_size:
        print(f"Warning: only {len(records)} records available, sampling all of them.")

    sample = rng.sample(records, k=sample_size)
    with output_path.open("w", encoding="utf-8") as f:
        for record in sample:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Wrote {len(sample)} samples to {output_path}")


if __name__ == "__main__":
    main()
