#!/usr/bin/env python3
"""Build Nanochat SFT JSONL from verified responses."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    from nanochat.common import get_base_dir
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Failed to import nanochat.common: {exc}")
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    raise SystemExit("Missing dependency: tqdm. Install with `uv sync` or `pip install tqdm`.")


def _default_output_dir() -> str:
    if os.environ.get("SAFETY_CURATION_DIR"):
        return os.environ["SAFETY_CURATION_DIR"]
    return os.path.join(get_base_dir(), "safety_response_curation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample verified responses and format for SFT")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL from step3")
    parser.add_argument("--output-dir", type=str, default=_default_output_dir(), help="Output directory")
    parser.add_argument(
        "--output-name",
        type=str,
        default="step4_sft.jsonl",
        help="Output JSONL filename for SFT",
    )
    parser.add_argument(
        "--meta-name",
        type=str,
        default="step4_selected_metadata.jsonl",
        help="Output JSONL filename for selected metadata",
    )
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of prompts to sample")
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sft_path = output_dir / args.output_name
    meta_path = output_dir / args.meta_name

    verified_by_id = defaultdict(list)

    with Path(args.input).open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="load verifications"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("verified"):
                continue
            record_id = str(row.get("id"))
            verified_by_id[record_id].append(row)

    eligible_ids = list(verified_by_id.keys())
    if not eligible_ids:
        raise SystemExit("No verified responses found.")

    rng = random.Random(args.seed)
    sample_size = min(args.sample_size, len(eligible_ids))
    if sample_size < args.sample_size:
        print(f"Warning: only {len(eligible_ids)} prompts have verified responses. Sampling all.")

    selected_ids = rng.sample(eligible_ids, k=sample_size)

    with sft_path.open("w", encoding="utf-8") as sft_file, meta_path.open("w", encoding="utf-8") as meta_file:
        for record_id in tqdm(selected_ids, desc="write sft"):
            options = verified_by_id[record_id]
            chosen = rng.choice(options)
            prompt = chosen.get("prompt", "")
            response = chosen.get("response", "")
            conversation = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            sft_file.write(json.dumps(conversation, ensure_ascii=True) + "\n")
            meta_file.write(json.dumps(chosen, ensure_ascii=True) + "\n")

    print(f"Wrote {len(selected_ids)} SFT conversations to {sft_path}")
    print(f"Wrote metadata to {meta_path}")


if __name__ == "__main__":
    main()
