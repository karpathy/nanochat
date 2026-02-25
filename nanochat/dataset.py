"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames

# Always use a fixed shard for val so that metrics don't depend on how many shards are downloaded
# Keeping pinned to shard_01822.
VAL_SHARD_INDEX = 1822
assert 0 <= VAL_SHARD_INDEX <= MAX_SHARD, "VAL_SHARD_INDEX must be within [0, MAX_SHARD]"
VAL_SHARD_FILENAME = index_to_filename(VAL_SHARD_INDEX)

base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None, exclude_filenames=()):
    """Looks into a data dir and returns full paths to parquet files."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    exclude = set(exclude_filenames)
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".parquet") and not f.endswith(".tmp") and f not in exclude
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def get_parquet_paths(split, data_dir=None):
    """
    Returns the parquet paths for a split.

    Validation is always a fixed shard so that metrics are stable across partial downloads.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    data_dir = DATA_DIR if data_dir is None else data_dir
    val_path = os.path.join(data_dir, VAL_SHARD_FILENAME)
    if split == "val":
        if not os.path.exists(val_path):
            raise FileNotFoundError(
                f"Validation shard {VAL_SHARD_FILENAME} not found in {data_dir}. "
                f"Run: python -m nanochat.dataset -n <N> (downloads the val shard too)."
            )
        return [val_path]
    else:
        # train split: list files while excluding val
        return list_parquet_files(data_dir, exclude_filenames=(VAL_SHARD_FILENAME,))

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". validation is always a fixed shard.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    parquet_paths = get_parquet_paths(split)
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(index):
    """ Downloads a single file index, with some backoff """

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
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
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
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
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
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of training shards to download (default: -1 = all).")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    if VAL_SHARD_INDEX not in ids_to_download:
        ids_to_download.append(VAL_SHARD_INDEX)
    ids_to_download = sorted(set(ids_to_download))
    
    if args.num_files != -1 and args.num_files <= MAX_SHARD:
        print(f"Downloading {len(ids_to_download)} shards ({num} train + 1 val) using {args.num_workers} workers...")
    else:
        print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")

    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
