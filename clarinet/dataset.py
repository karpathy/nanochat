"""
Source-labelled parquet listing for clarinet.

Combines the existing climbmix shards (from nanochat/dataset.py) with the
reasoning-corpus shards produced by clarinet/prepare_reasoning_data.py,
attaching an is_reasoning flag to each path so the clarinet dataloader can
pick a source-marker token per document.

Train/val convention matches upstream nanochat: last shard in each source dir
is the validation shard, all earlier shards are train.
"""

import os

from nanochat.common import get_base_dir
from nanochat.dataset import list_parquet_files as list_climbmix_parquet_files

REASONING_DIR_NAME = "reasoning_data"
FINEMATH_DIR_NAME = "finemath"


def reasoning_data_dir():
    """
    Where prepare_reasoning_data.py writes the reasoning-corpus shards, and where
    the dataloader reads them from. Currently FineMath (HuggingFaceTB/finemath,
    finemath-4plus subset) — the original plan called for EleutherAI/proof-pile-2
    but their data hosting rotted out (data files return 404 on HF Hub as of
    2026-05). FineMath is the closest practical replacement: parquet-native,
    actively maintained, math-focused.
    """
    return os.path.join(get_base_dir(), REASONING_DIR_NAME, FINEMATH_DIR_NAME)


def list_reasoning_parquet_files():
    """Returns paths to reasoning-corpus shards written by prepare_reasoning_data.py."""
    rdir = reasoning_data_dir()
    if not os.path.isdir(rdir):
        return []
    files = sorted(
        f for f in os.listdir(rdir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    )
    return [os.path.join(rdir, f) for f in files]


def list_parquet_files_with_source(split):
    """
    Returns [(path, is_reasoning), ...] for the requested split across both
    corpora. Last shard in each source dir is val; everything else is train.
    Caller (the dataloader) chooses how to interleave the two sources.
    """
    assert split in ("train", "val"), "split must be 'train' or 'val'"

    climbmix = list_climbmix_parquet_files(warn_on_legacy=(split == "train"))
    reasoning = list_reasoning_parquet_files()
    assert reasoning, (
        f"No reasoning-corpus shards found under {reasoning_data_dir()}. "
        "Run `python -m clarinet.prepare_reasoning_data` first."
    )

    if split == "train":
        climbmix = climbmix[:-1]
        reasoning = reasoning[:-1]
    else:
        climbmix = climbmix[-1:]
        reasoning = reasoning[-1:]

    return [(p, False) for p in climbmix] + [(p, True) for p in reasoning]
