"""Utilities for the ClimbMix-400B pretraining dataset (parquet shards).

Functions:
    list_parquet_files: Return sorted paths to all parquet files in a data directory.
    parquets_iter_batched: Iterate over row-group batches for train or val split.
    main: Download dataset shards to the configured base directory.
"""


import os


import pyarrow.parquet as pq


from nanochat.common import data_dir as _data_dir




# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(base_dir: str, warn_on_legacy: bool = False) -> list[str]:
    """Looks into a data dir and returns full paths to all parquet files."""
    data_dir = _data_dir(base_dir=base_dir)
    parquet_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet") and not f.endswith(".tmp")])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def parquets_iter_batched(base_dir: str, split: str, start: int = 0, step: int = 1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(base_dir)
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            yield texts


