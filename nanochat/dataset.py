"""Base/pretraining dataset utilities.

This module contains utilities for:
- Iterating over the parquet files and yielding documents from them
- Downloading the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Iterator
from multiprocessing import Pool

import pyarrow.parquet as pq
import requests

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822  # the last datashard is shard_01822.parquet


def index_to_filename(index: int) -> str:
    """Converts a shard index to its corresponding filename.

    Args:
        index: The shard index number.

    Returns:
        The filename in the format "shard_XXXXX.parquet".
    """
    return f"shard_{index:05d}.parquet"


base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported


def list_parquet_files(data_dir: str | None = None) -> list[str]:
    """Lists all parquet files in a data directory.

    Args:
        data_dir: The directory to search for parquet files.
            Defaults to DATA_DIR if None.

    Returns:
        A sorted list of full paths to all parquet files in the directory.
    """
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp")
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def parquets_iter_batched(
    split: str,
    start: int = 0,
    step: int = 1,
) -> Iterator[list[str]]:
    """Iterates through the dataset in batches of underlying row_groups.

    Args:
        split: The data split to iterate over. Must be "train" or "val".
            The last parquet file is used for validation.
        start: The starting row group index. Useful for skipping rows in DDP
            (e.g., start=rank).
        step: The step size between row groups. Useful for DDP
            (e.g., step=world_size).

    Yields:
        Lists of text documents from each row group.

    Raises:
        AssertionError: If split is not "train" or "val".
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            yield texts


# -----------------------------------------------------------------------------


def download_single_file(index: int) -> bool:
    """Downloads a single file by index with exponential backoff retry.

    Args:
        index: The shard index to download.

    Returns:
        True if the download was successful or the file already exists,
        False if the download failed after all retry attempts.
    """
    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument(
        "-n", "--num-files",
        type=int,
        default=-1,
        help="Number of shards to download (default: -1), -1 = disable",
    )
    parser.add_argument(
        "-w", "--num-workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )
    args = parser.parse_args()

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
