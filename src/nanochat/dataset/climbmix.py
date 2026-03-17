"""Utilities for the ClimbMix-400B pretraining dataset (parquet shards).

Functions:
    list_parquet_files: Return sorted paths to all parquet files in a data directory.
    parquets_iter_batched: Iterate over row-group batches for train or val split.
    main: Download dataset shards to the configured base directory.
"""

from multiprocessing import Pool

from nanochat.common import data_dir as _data_dir
from nanochat.common import download_single_file
from nanochat.config import Config

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542  # the last datashard is shard_06542.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet"  # format of the filenames


def climbmix_download(cfg: Config, num_files: int = -1, num_workers: int = 4) -> None:
    """Download dataset shards to the configured base directory.

    num_files download ~170 shards, enough for GPT-2, adjust as desired

    Args:
        cfg: Resolved nanochat config (only ``cfg.common.base_dir`` is used).
        num_files: Number of train shards to download; -1 downloads all shards.
        num_workers: Number of parallel download workers.
    """
    # Prepare the output directory
    data_dir = _data_dir(base_dir=cfg.common.base_dir)

    # The way this works is that the user specifies the number of train shards to download via the -n flag.
    # In addition to that, the validation shard is *always* downloaded and is pinned to be the last shard.
    num_train_shards = MAX_SHARD if num_files == -1 else min(num_files, MAX_SHARD)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD)  # always download the validation shard

    # Download the shards
    print(f"Downloading {len(ids_to_download)} shards using {num_workers} workers...")
    print(f"Target directory: {data_dir}")
    print()

    def download_wrapper(index: int):
        filename = index_to_filename(index)
        return download_single_file(data_dir, f"{BASE_URL}/{filename}", filename)

    with Pool(processes=num_workers) as pool:
        results = pool.map(download_wrapper, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {data_dir}")
