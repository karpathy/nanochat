"""
Source-labelled parquet listing for clarinet.

Combines the existing climbmix shards (from nanochat/dataset.py) with the
proof-pile-2 shards produced by clarinet/prepare_proof_pile.py, attaching an
is_reasoning flag to each path so the clarinet dataloader can pick a
source-marker token per document.

Train/val convention matches upstream nanochat: last shard in each source dir
is the validation shard, all earlier shards are train.
"""

import os

from nanochat.common import get_base_dir
from nanochat.dataset import list_parquet_files as list_climbmix_parquet_files

REASONING_DIR_NAME = "reasoning_data"
PROOF_PILE_DIR_NAME = "proof_pile_2"


def proof_pile_dir():
    """Where prepare_proof_pile.py writes shards, and where the dataloader reads them from."""
    return os.path.join(get_base_dir(), REASONING_DIR_NAME, PROOF_PILE_DIR_NAME)


def list_proof_pile_parquet_files():
    """Returns paths to proof-pile-2 shards written by prepare_proof_pile.py."""
    pdir = proof_pile_dir()
    if not os.path.isdir(pdir):
        return []
    files = sorted(
        f for f in os.listdir(pdir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )
    return [os.path.join(pdir, f) for f in files]


def list_parquet_files_with_source(split):
    """
    Returns [(path, is_reasoning), ...] for the requested split across both
    corpora. Last shard in each source dir is val; everything else is train.
    Caller (the dataloader) chooses how to interleave the two sources.
    """
    assert split in ("train", "val"), "split must be 'train' or 'val'"

    climbmix = list_climbmix_parquet_files(warn_on_legacy=(split == "train"))
    reasoning = list_proof_pile_parquet_files()
    assert reasoning, (
        f"No proof-pile-2 shards found under {proof_pile_dir()}. "
        "Run `python -m clarinet.prepare_proof_pile` first."
    )

    if split == "train":
        climbmix = climbmix[:-1]
        reasoning = reasoning[:-1]
    else:
        climbmix = climbmix[-1:]
        reasoning = reasoning[-1:]

    return [(p, False) for p in climbmix] + [(p, True) for p in reasoning]
