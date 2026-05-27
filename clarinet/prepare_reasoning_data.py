"""
One-time download + re-shard of HuggingFaceTB/finemath into the same parquet
layout that nanochat/dataset.py expects for climbmix:
  <base_dir>/reasoning_data/finemath/shard_NNNNN.parquet
  - single 'text' column
  - ~DOCS_PER_ROW_GROUP docs per row group (matches climbmix layout for the
    upstream best-fit dataloader's row-group-at-a-time read pattern)

History: clarinet's original plan called for EleutherAI/proof-pile-2 as the
reasoning instrument. As of 2026-05, proof-pile-2's data hosting on HF Hub
has rotted (the dataset card and loader script resolve, but the actual data
files return 404). FineMath (HuggingFaceTB/finemath, finemath-4plus subset)
is the practical replacement: parquet-native (no zstandard or trust_remote_code
gymnastics), actively maintained, math-focused. Narrower than proof-pile-2's
algebraic-stack + arxiv + open-web-math mix, but still a clear "reasoning-rich"
instrument distinct from general web text.

Run once before clarinet pretraining:

  python -m clarinet.prepare_reasoning_data --num-shards 50
"""

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from clarinet.dataset import reasoning_data_dir

DEFAULT_REPO = "HuggingFaceTB/finemath"
DEFAULT_CONFIG = "finemath-4plus"
DOCS_PER_SHARD = 65536  # ~64 row groups of 1024 docs each, in line with climbmix shard sizes
DOCS_PER_ROW_GROUP = 1024


def streaming_corpus(repo, config):
    return load_dataset(repo, config, split="train", streaming=True)


def write_shard(out_dir, shard_idx, texts):
    path = os.path.join(out_dir, f"shard_{shard_idx:05d}.parquet")
    table = pa.table({"text": texts})
    # Match the climbmix layout (compressed, multiple row groups per shard) so
    # the upstream loader's read_row_group path behaves identically.
    pq.write_table(
        table,
        path,
        compression="zstd",
        row_group_size=DOCS_PER_ROW_GROUP,
    )
    return path


def main():
    parser = argparse.ArgumentParser(description="Download and re-shard the reasoning corpus")
    parser.add_argument("-n", "--num-shards", type=int, default=50,
                        help="Number of train shards to write (validation shard is always written last). "
                             "At DOCS_PER_SHARD=65536 docs/shard this gives ~3M docs per 50 shards.")
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help="HF dataset repo ID. Default: HuggingFaceTB/finemath")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="HF dataset config / subset. Default: finemath-4plus")
    args = parser.parse_args()

    out_dir = reasoning_data_dir()
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing {args.repo}/{args.config} shards to: {out_dir}")

    stream = streaming_corpus(args.repo, args.config)

    buffer = []
    shard_idx = 0
    total_shards = args.num_shards + 1  # +1 for validation shard, mirrors climbmix layout

    for record in stream:
        text = record.get("text")
        if not text:
            continue
        buffer.append(text)
        if len(buffer) >= DOCS_PER_SHARD:
            path = write_shard(out_dir, shard_idx, buffer)
            print(f"  wrote {path} ({len(buffer)} docs)")
            buffer = []
            shard_idx += 1
            if shard_idx >= total_shards:
                break

    if buffer and shard_idx < total_shards:
        path = write_shard(out_dir, shard_idx, buffer)
        print(f"  wrote {path} ({len(buffer)} docs, final partial shard)")
        shard_idx += 1

    print(f"Done. Wrote {shard_idx} shards.")


if __name__ == "__main__":
    main()
