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

# Support for loading openwebtext from local parquet files
USE_OPENWEBTEXT = True

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    
    # If using openwebtext, load from local parquet files directly (no download, no API calls)
    if USE_OPENWEBTEXT:
        # Load openwebtext dataset directly from local parquet files (hardcoded path)
        # Note: openwebtext only has 'train' split, so we use it for both train and val
        # Try multiple possible paths
        parquet_dir = None
        for possible_dir in [
            "/thullms/dpq23/.cache/huggingface/datasets/openwebtext/plain_text",
            "/thullms/public/openwebtext_new/openwebtext/plain_text"
        ]:
            if os.path.exists(possible_dir) and os.path.exists(os.path.join(possible_dir, "train-00000-of-00080.parquet")):
                parquet_dir = possible_dir
                break
        
        if parquet_dir is None:
            raise RuntimeError("Could not find openwebtext parquet files in expected locations")
        
        # Load all parquet files directly
        parquet_files = sorted([f for f in os.listdir(parquet_dir) if f.endswith('.parquet')])
        parquet_paths = [os.path.join(parquet_dir, f) for f in parquet_files]
        
        # Calculate total rows to determine train/val split
        total_rows = 0
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            total_rows += pf.metadata.num_rows
        
        # For validation, use the last 1% of the dataset
        if split == "val":
            val_start_row = int(total_rows * 0.99)
            # Find which file contains the validation start
            current_row = 0
            val_file_start_idx = 0
            for i, filepath in enumerate(parquet_paths):
                pf = pq.ParquetFile(filepath)
                if current_row + pf.metadata.num_rows > val_start_row:
                    val_file_start_idx = i
                    break
                current_row += pf.metadata.num_rows
            parquet_paths = parquet_paths[val_file_start_idx:]
        else:
            # For training, use 99% of the dataset - limit to first 99% of files
            train_end_row = int(total_rows * 0.99)
            current_row = 0
            train_file_end_idx = len(parquet_paths)
            for i, filepath in enumerate(parquet_paths):
                pf = pq.ParquetFile(filepath)
                if current_row + pf.metadata.num_rows >= train_end_row:
                    train_file_end_idx = i + 1
                    break
                current_row += pf.metadata.num_rows
            parquet_paths = parquet_paths[:train_file_end_idx]
        
        # Now iterate through parquet files similar to original code
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(start, pf.num_row_groups, step):
                rg = pf.read_row_group(rg_idx)
                texts = rg.column('text').to_pylist()
                yield texts
        return
    
    # Original parquet file iteration
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
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


def download_openwebtext():
    """Check if openwebtext dataset exists locally (no download, no API calls)."""
    if not USE_OPENWEBTEXT:
        print("USE_OPENWEBTEXT is False. Skipping openwebtext check.")
        return False
    
    print("Checking for openwebtext dataset in local directories...")
    
    # Try multiple possible paths
    parquet_dir = None
    for possible_dir in [
        "/thullms/dpq23/.cache/huggingface/datasets/openwebtext/plain_text",
        "/thullms/public/openwebtext_new/openwebtext/plain_text"
    ]:
        if os.path.exists(possible_dir) and os.path.exists(os.path.join(possible_dir, "train-00000-of-00080.parquet")):
            parquet_dir = possible_dir
            break
    
    if parquet_dir is None:
        print("ERROR: Could not find openwebtext parquet files in expected locations:")
        print("  - /thullms/dpq23/.cache/huggingface/datasets/openwebtext/plain_text")
        print("  - /thullms/public/openwebtext_new/openwebtext/plain_text")
        return False
    
    # Count total rows from parquet files
    parquet_files = sorted([f for f in os.listdir(parquet_dir) if f.endswith('.parquet')])
    total_rows = 0
    for filename in parquet_files:
        filepath = os.path.join(parquet_dir, filename)
        pf = pq.ParquetFile(filepath)
        total_rows += pf.metadata.num_rows
    
    print(f"Found openwebtext dataset at: {parquet_dir}")
    print(f"Total parquet files: {len(parquet_files)}")
    print(f"Total rows: {total_rows:,}")
    print("OpenWebText dataset is ready to use (loaded directly from local parquet files).")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    # If using openwebtext, download it
    if USE_OPENWEBTEXT:
        print("Using openwebtext dataset from HuggingFace datasets.")
        success = download_openwebtext()
        if success:
            print("Done! OpenWebText dataset is ready to use.")
        else:
            print("Failed to download openwebtext dataset.")
            exit(1)
        exit(0)

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
