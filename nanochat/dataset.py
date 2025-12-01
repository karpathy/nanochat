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
from dataclasses import dataclass
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# Language-specific data configuration

@dataclass
class DataConfig:
    """Configuration for language-specific data sources."""
    base_url: str
    max_shard: int
    text_column: str  # column name in parquet file

# Data configurations for each supported language
DATA_CONFIGS = {
    "en": DataConfig(
        base_url="https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main",
        max_shard=1822,
        text_column="text",
    ),
    "ja": DataConfig(
        base_url="https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese/resolve/main/data",
        max_shard=1238,  # 1239 files (train-00000-of-01239 to train-01238-of-01239)
        text_column="text",
    ),
}

def get_data_config(lang: str = None) -> DataConfig:
    """
    Get data configuration for the specified language.
    Falls back to NANOCHAT_LANG environment variable, then defaults to "en".
    """
    if lang is None:
        lang = os.environ.get("NANOCHAT_LANG", "en")
    if lang not in DATA_CONFIGS:
        raise ValueError(f"Unsupported language '{lang}'. Supported: {list(DATA_CONFIGS.keys())}")
    return DATA_CONFIGS[lang]

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset (legacy compatibility)

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def get_data_dir(lang: str = None):
    """Get the data directory for the specified language."""
    if lang is None:
        lang = os.environ.get("NANOCHAT_LANG", "en")
    base_dir = get_base_dir()
    if lang == "en":
        return os.path.join(base_dir, "base_data")
    else:
        return os.path.join(base_dir, f"base_data_{lang}")

def list_parquet_files(data_dir=None, lang=None):
    """
    Looks into a data dir and returns full paths to all parquet files.
    If lang is specified, uses the appropriate language-specific data directory.
    """
    if data_dir is None:
        data_dir = get_data_dir(lang)
    os.makedirs(data_dir, exist_ok=True)
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1, lang=None):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    - lang: language code (e.g., "en", "ja"). Defaults to NANOCHAT_LANG env var or "en".
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    config = get_data_config(lang)
    parquet_paths = list_parquet_files(lang=lang)
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column(config.text_column).to_pylist()
            yield texts

# -----------------------------------------------------------------------------
# Language-specific filename formats
def get_filename_formatter(lang: str):
    """Get the filename formatter function for the specified language."""
    if lang == "ja":
        # Japanese dataset uses train-XXXXX-of-YYYYY.parquet format
        def formatter(index, total):
            return f"train-{index:05d}-of-{total:05d}.parquet"
        return formatter
    else:
        # English dataset uses shard_XXXXX.parquet format
        return lambda index, total: f"shard_{index:05d}.parquet"

def download_single_file(index, lang=None, config=None, data_dir=None):
    """Downloads a single file index, with some backoff"""
    if config is None:
        config = get_data_config(lang)
    if data_dir is None:
        data_dir = get_data_dir(lang)
    os.makedirs(data_dir, exist_ok=True)

    # Get the filename formatter for this language
    if lang is None:
        lang = os.environ.get("NANOCHAT_LANG", "en")
    formatter = get_filename_formatter(lang)
    total = config.max_shard + 1
    filename = formatter(index, total)

    # Construct the local filepath for this file and skip if it already exists
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{config.base_url}/{filename}"
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

# Legacy wrapper for backward compatibility with multiprocessing Pool
def _download_single_file_en(index):
    """Download a single English file (for multiprocessing compatibility)."""
    return download_single_file(index, lang="en")

def _download_single_file_ja(index):
    """Download a single Japanese file (for multiprocessing compatibility)."""
    return download_single_file(index, lang="ja")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1 = all)")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    parser.add_argument("-l", "--lang", type=str, default=None, help="Language code (en/ja). Defaults to NANOCHAT_LANG or 'en'")
    args = parser.parse_args()

    lang = args.lang if args.lang else os.environ.get("NANOCHAT_LANG", "en")
    config = get_data_config(lang)
    data_dir = get_data_dir(lang)

    num = config.max_shard + 1 if args.num_files == -1 else min(args.num_files, config.max_shard + 1)
    ids_to_download = list(range(num))
    print(f"Language: {lang}")
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {data_dir}")
    print()

    # Use the appropriate download function based on language
    download_fn = _download_single_file_ja if lang == "ja" else _download_single_file_en
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_fn, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {data_dir}")
