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
import logging 
import requests
from tqdm import tqdm 
import pyarrow.parquet as pq
from functools import partial
from multiprocessing import Pool

from nanochat.common import get_base_dir, setup_file_logger

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
# Minimal logger setup for DEBUG level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_path = setup_file_logger(
    logger_name=__name__,
    filename="dataset_download.log",
    level=logging.DEBUG,
    formatter=logging.Formatter(
        "%(asctime)s - %(processName)s - %(levelname)s - %(message)s"
    ),
)

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
        logger.debug(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    logger.debug(f"Downloading {filename}...")

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
            logger.debug(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            logger.warning(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
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
                logger.debug(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.debug(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    parser.add_argument(
        "-f",
        "--work-share-factor",
        type=int,
        default=8,
        help=(
            """Controls how each worker's share of shards is subdivided. CHUNK_SIZE is computed as len(ids_to_download) // (num_workers * work_share_factor), so it is the number of tasks a worker pulls per request from the main process. for example, for 240 shards and 4 workers the default value (8) produces 7 shards per request. setting it 1 gives a worker its entire share (~60 shards) in one go with minimal coordination but slow progress updates. larger work-share-factor values make the main process hand out smaller batches more often for faster feedback at a small scheduling cost."""
        ),
    )
    args = parser.parse_args()

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    logger.info(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    logger.info(f"Dataset target directory: {DATA_DIR}")
    logger.info(f"Dataset downloader debug logs will be written to: {log_path}")

    # pool.imap_unordered pulls `chunksize` tasks from the main process before asking for more
    work_share_factor = max(1, args.work_share_factor)
    CHUNK_SIZE = max(1, len(ids_to_download) // (args.num_workers * work_share_factor))
    ok_count = 0
    with Pool(processes=args.num_workers) as pool:
        for ok in tqdm(
            pool.imap_unordered(
                partial(download_single_file), ids_to_download, chunksize=CHUNK_SIZE
            ),
            total=len(ids_to_download),
            desc="all shards",
            smoothing=0.1,
        ):
            ok_count += int(ok)

    # Report results
    logger.info(f"Done! Downloaded: {ok_count}/{len(ids_to_download)} shards to {DATA_DIR}")
