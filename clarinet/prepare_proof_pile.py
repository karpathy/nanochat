"""
One-time download + re-shard of EleutherAI/proof-pile-2 into the same parquet
layout that nanochat/dataset.py expects for climbmix:
  <base_dir>/reasoning_data/proof_pile_2/shard_NNNNN.parquet
  - single 'text' column
  - ~docs_per_row_group docs per row group (matches climbmix layout for the
    upstream best-fit dataloader's row-group-at-a-time read pattern)

Run once before clarinet pretraining:

  python -m clarinet.prepare_proof_pile --num-shards 50

proof-pile-2 has three subsets: algebraic-stack, arxiv, open-web-math. We
stream-interleave them so each shard mixes domains rather than ordering by
subset (which would create epoch-level distribution shift).
"""

import argparse
import os

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import interleave_datasets, load_dataset

from nanochat.common import get_base_dir

REASONING_DIR_NAME = "reasoning_data"
PROOF_PILE_DIR_NAME = "proof_pile_2"
DOCS_PER_SHARD = 65536  # ~64 row groups of 1024 docs each, in line with climbmix shard sizes
DOCS_PER_ROW_GROUP = 1024


def proof_pile_dir():
    return os.path.join(get_base_dir(), REASONING_DIR_NAME, PROOF_PILE_DIR_NAME)


def streaming_proof_pile(subsets):
    streams = [
        load_dataset("EleutherAI/proof-pile-2", subset, split="train", streaming=True)
        for subset in subsets
    ]
    return interleave_datasets(streams, stopping_strategy="all_exhausted")


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
    parser = argparse.ArgumentParser(description="Download and re-shard proof-pile-2")
    parser.add_argument("-n", "--num-shards", type=int, default=50,
                        help="Number of train shards to write (validation shard is always written last). "
                             "At DOCS_PER_SHARD=65536 docs/shard this gives ~3M docs per 50 shards.")
    parser.add_argument("--subsets", nargs="+",
                        default=["algebraic-stack", "arxiv", "open-web-math"],
                        help="Which proof-pile-2 subsets to interleave.")
    args = parser.parse_args()

    out_dir = proof_pile_dir()
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing proof-pile-2 shards to: {out_dir}")
    print(f"Subsets: {args.subsets}")

    stream = streaming_proof_pile(args.subsets)

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
