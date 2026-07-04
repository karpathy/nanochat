"""
The base/pretraining dataset is a set of parquet files.

Datasets are *named*: they live in $NANOCHAT_BASE_DIR/datasets/<name>/ and an experiment
references one by name (default: the canonical "climbmix"). The contract for a dataset
is deliberately narrow:
- a directory of .parquet files, each with a 'text' column
- filenames sort into the intended read order
- the LAST file is the validation split, all others are train

nanochat knows how to download ("materialize") the canonical dataset from its hosting.
Any other name is user-provided: drop parquet shards satisfying the contract into the
directory and select it, e.g. NANOCHAT_DATASET=my_remix. This opens the data to
experimentation while everything downstream stays fixed.

This file contains utilities for:
- resolving a dataset name to its directory
- materializing and verifying the canonical dataset (quiet when already cached)
- iterating over the parquet files and yielding documents

For details of how the canonical dataset was prepared, see dev/repackage_data_reference.py.
"""

import os
import argparse
import time
from functools import partial
from multiprocessing import Pool

import requests
import pyarrow.parquet as pq

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The canonical dataset: hosted, downloaded on demand

CANONICAL_DATASET = "climbmix"
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542 # the last datashard is shard_06542.parquet, and it is the val split
MIN_SHARD_BYTES = 1024 * 1024 # a healthy shard is ~100MB, anything under 1MB is corrupt
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def get_dataset_name():
    """The active dataset name: $NANOCHAT_DATASET, defaulting to the canonical dataset."""
    return os.environ.get("NANOCHAT_DATASET", CANONICAL_DATASET)

def get_dataset_dir(name=None):
    """Resolve a dataset name to its directory in the shared store."""
    name = get_dataset_name() if name is None else name
    return os.path.join(get_base_dir(), "datasets", name)

def maybe_notice_legacy_migration(data_dir):
    """
    Migration notice for the July 2026 experiment refactor (see experiment_refactor.md).
    Everyone who does `git pull` across it with an existing dataset download is expected
    to see this once. Returns True if the old layout was detected; the caller should
    abort so the user can `mv` instead of re-downloading hundreds of GB.
    """
    legacy_dir = os.path.join(get_base_dir(), "base_data_climbmix")
    is_canonical = data_dir == get_dataset_dir(CANONICAL_DATASET)
    if not (is_canonical and os.path.isdir(legacy_dir) and not os.path.isdir(data_dir)):
        return False
    print("=" * 80)
    print("  MIGRATION REQUIRED: nanochat's on-disk layout changed (the experiment refactor)")
    print("=" * 80)
    print()
    print("  Your pretraining dataset was found at the old location:")
    print(f"    {legacy_dir}")
    print("  Datasets are now named and live in a shared store. Move yours (instant, no re-download):")
    print()
    print(f"    mkdir -p {os.path.dirname(data_dir)}")
    print(f"    mv {legacy_dir} {data_dir}")
    print()
    print("  Also note: tokenizers and checkpoints now live per-experiment in")
    print(f"    {os.path.join(get_base_dir(), 'experiments')}/<name>/")
    print("  The old tokenizer/ and *_checkpoints/ dirs are orphaned (old checkpoints do not")
    print("  load with the current code anyway) and can be deleted to reclaim disk.")
    print("=" * 80)
    return True

def list_parquet_files(data_dir=None):
    """Looks into a data dir and returns full paths to all parquet files."""
    data_dir = get_dataset_dir() if data_dir is None else data_dir
    if not os.path.isdir(data_dir):
        if maybe_notice_legacy_migration(data_dir):
            raise FileNotFoundError(f"Dataset found at the pre-refactor location, see the migration notice above")
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            f"For the canonical dataset, download shards first, e.g.: python -m nanochat.dataset -n 240\n"
            f"For a custom dataset, place parquet shards (with a 'text' column) in that directory.\n"
            f"(the last shard, in sorted filename order, is used as the validation split)"
        )
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
# Materializing and verifying

def is_valid_shard(filepath):
    """Downloads are atomic (tmp file + rename), so existence implies a completed
    download; the size check additionally guards against corrupt/empty files."""
    return os.path.exists(filepath) and os.path.getsize(filepath) >= MIN_SHARD_BYTES

def print_summary(name, data_dir):
    """One quiet line describing the state of a dataset on disk."""
    parquet_paths = list_parquet_files(data_dir)
    total_bytes = sum(os.path.getsize(p) for p in parquet_paths)
    num_train = len(parquet_paths) - 1
    print(f"dataset {name}: {num_train} train shards + 1 val shard, {total_bytes / 2**30:.1f} GiB, at {data_dir}")

def download_single_file(index, data_dir):
    """Downloads a single canonical shard by index, with some backoff."""
    filename = index_to_filename(index)
    filepath = os.path.join(data_dir, filename)
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first, then move into place atomically
            temp_path = filepath + ".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024): # 1MB chunks
                    if chunk:
                        f.write(chunk)
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
    parser = argparse.ArgumentParser(description="Materialize/verify a pretraining dataset")
    parser.add_argument("-d", "--dataset", type=str, default=None, help="Dataset name (default: $NANOCHAT_DATASET or the canonical dataset)")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of canonical train shards to ensure on disk (-1 = all)")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    name = args.dataset if args.dataset is not None else get_dataset_name()
    data_dir = get_dataset_dir(name)

    if name != CANONICAL_DATASET:
        # user-provided dataset: nothing to download, just verify it satisfies the contract
        parquet_paths = list_parquet_files(data_dir) # raises with instructions if missing
        assert len(parquet_paths) >= 2, f"Dataset {name} needs at least 2 parquet files (train + val), found {len(parquet_paths)}"
        print_summary(name, data_dir)
        raise SystemExit(0)

    # The canonical dataset: ensure the requested train shards + the val shard are on disk.
    # The user asks for the first n train shards; the val shard is pinned to be the last shard.
    if maybe_notice_legacy_migration(data_dir):
        raise SystemExit(1) # abort so the user can mv their download instead of re-downloading
    os.makedirs(data_dir, exist_ok=True)
    num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
    ids_needed = list(range(num_train_shards))
    ids_needed.append(MAX_SHARD) # the val shard
    ids_missing = [i for i in ids_needed if not is_valid_shard(os.path.join(data_dir, index_to_filename(i)))]

    if ids_missing:
        print(f"Downloading {len(ids_missing)} shards using {args.num_workers} workers...")
        with Pool(processes=args.num_workers) as pool:
            results = pool.map(partial(download_single_file, data_dir=data_dir), ids_missing)
        successful = sum(1 for success in results if success)
        print(f"Downloaded {successful}/{len(ids_missing)} shards")
        if successful < len(ids_missing):
            raise SystemExit(1) # fail loudly so a driving script (e.g. runs/run.sh) stops

    print_summary(name, data_dir)
