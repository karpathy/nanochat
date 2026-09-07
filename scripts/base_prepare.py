"""
Prepare the pretraining data: everything that touches text happens here, once.

    1. download the raw parquet shards (the canonical dataset; or bring your own
       directory of shards, see harness/dataset.py for the contract)
    2. train the BPE tokenizer on a sample of train text; save it + token_bytes
    3. tokenize every shard and pack its documents into rows (BOS-aligned best-fit:
       every row starts at a document start), writing one bin shard per raw shard:
       shard_00042.parquet -> train_00042.bin, and the val shard -> val_00000.bin

Every artifact is skipped if it already exists, so re-running is cheap and a crash
resumes. Everything lands in $NANOCHAT_BASE_DIR/datasets/<name>/ next to the raw shards
(NANOCHAT_DATASET selects <name>). The shard format is in nanochat/dataloader.py. The
chat data (SFT rows, same format) is scripts/chat_prepare.py, run after this.

python -m scripts.base_prepare -n 300     # the first 300 train shards (+ the val shard)
"""

import os
import time
import argparse
from multiprocessing import Pool

import numpy as np
import pyarrow.parquet as pq
import torch

from harness.experiment import format_record, format_invocation
from nanochat.common import get_dataset_name, get_dataset_dir
from harness.dataset import materialize, list_parquet_files, parquets_iter_batched
from nanochat.tokenizer import RustBPETokenizer, get_tokenizer_dir, tokenizer_id
from nanochat.dataloader import write_shard, read_header, split_info, pack_rows

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Prepare the pretraining data: download, tokenizer, packed token shards")
parser.add_argument("-n", "--num-shards", type=int, default=300, help="train shards to prepare (~48M tokens each); the val shard always is")
parser.add_argument("-T", "--row-len", type=int, default=2048, help="tokens per packed row = the largest max_seq_len models can train at")
parser.add_argument("--vocab-size", type=int, default=32768, help="tokenizer vocabulary size (2^15)")
parser.add_argument("--max-chars", type=int, default=2_000_000_000, help="characters of train text to train the tokenizer on")
parser.add_argument("--doc-cap", type=int, default=10_000, help="max characters per document for tokenizer training")
parser.add_argument("-w", "--num-workers", type=int, default=32, help="parallel workers for packing shards")
args = parser.parse_args()
print(format_invocation(args))

TOKENIZER_THREADS = 4 # per packing worker

# -----------------------------------------------------------------------------
# 1. The raw shards
dataset_name = get_dataset_name()
dataset_dir = get_dataset_dir()
materialize(dataset_name, args.num_shards)
parquet_paths = list_parquet_files(dataset_dir)
train_parquets = parquet_paths[:-1][:args.num_shards]
val_parquet = parquet_paths[-1]

# -----------------------------------------------------------------------------
# 2. The tokenizer (trained on the first --max-chars characters of the train split)
tokenizer_dir = get_tokenizer_dir()
tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
if os.path.exists(tokenizer_path):
    print(f"tokenizer: exists at {tokenizer_dir}, skipping")
    tokenizer_train_time = 0.0
else:
    def text_iterator():
        """Train documents, each cropped to --doc-cap characters, until --max-chars characters."""
        nchars = 0
        for batch in parquets_iter_batched(split="train"):
            for doc in batch:
                doc_text = doc[:args.doc_cap]
                nchars += len(doc_text)
                yield doc_text
                if nchars > args.max_chars:
                    return
    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(text_iterator(), args.vocab_size)
    tokenizer_train_time = time.time() - t0
    tokenizer.save(tokenizer_dir)
    # quick sanity check: encode/decode round-trips
    test_text = "Hello world! Numbers: 123, 4567. Contractions: I'm, you're. Unicode: 你好世界 🌍"
    assert tokenizer.decode(tokenizer.encode(test_text)) == test_text
    # cache the byte length of every token, for the bits-per-byte metric (see evals/bpb.py).
    # special tokens count 0 bytes. Use the raw bytes of the token: decoding to a string first
    # corrupts tokens that are not valid standalone UTF-8 (e.g. the raw bytes >= 0x80)
    special_ids = set(tokenizer.encode_special(s) for s in tokenizer.get_special_tokens())
    token_bytes = []
    for token_id in range(tokenizer.get_vocab_size()):
        num_bytes = 0 if token_id in special_ids else len(tokenizer.decode_single_token_bytes(token_id))
        token_bytes.append(num_bytes)
    token_bytes = torch.tensor(token_bytes, dtype=torch.int32)
    torch.save(token_bytes, os.path.join(tokenizer_dir, "token_bytes.pt"))
    print(f"tokenizer: trained in {tokenizer_train_time:.0f}s, saved to {tokenizer_dir}")
tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
vocab_size = tokenizer.get_vocab_size()
active_tokenizer_id = tokenizer_id(tokenizer_dir) # recorded in every shard, checked by the loader

# -----------------------------------------------------------------------------
# 3. Pack every shard into rows, one worker per shard

def init_worker(tokenizer_dir):
    global worker_tokenizer
    worker_tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

def pack_shard(job):
    """Tokenize one raw shard, pack it, write the bin shard. Returns (name, num_rows, raw_tokens)."""
    parquet_path, row_capacity, vocab_size, active_tokenizer_id, out_path = job
    bos = worker_tokenizer.get_bos_token_id()
    raw_tokens = 0
    def docs():
        """The shard's documents in order, tokenized, BOS-prepended, and pre-cropped to one row
        (a longer document only ever trains its prefix anyway, and would never fit otherwise)."""
        nonlocal raw_tokens
        pf = pq.ParquetFile(parquet_path)
        for rg_idx in range(pf.num_row_groups):
            texts = pf.read_row_group(rg_idx).column("text").to_pylist()
            token_lists = worker_tokenizer.encode(texts, prepend=bos, num_threads=TOKENIZER_THREADS)
            for tokens in token_lists:
                raw_tokens += len(tokens)
                yield tokens[:row_capacity]
    rows = [np.array(row, dtype=np.uint16) for row in pack_rows(docs(), row_capacity)]
    rows = np.stack(rows) # (num_rows, row_capacity)
    write_shard(out_path, rows[:, :-1], rows[:, 1:], vocab_size, active_tokenizer_id)
    return os.path.basename(out_path), rows.shape[0], raw_tokens

def remaining_jobs(jobs, what):
    """The jobs whose shard is not on disk yet. Shards that exist must match this run's row length and tokenizer."""
    todo = []
    for job in jobs:
        out_path = job[-1]
        if os.path.exists(out_path):
            shard_row_len, _, shard_vocab_size, shard_tokenizer_id = read_header(out_path)
            assert shard_row_len == args.row_len, f"{out_path} has row_len={shard_row_len}, not --row-len={args.row_len}: delete the .bin shards to re-pack"
            assert shard_vocab_size == vocab_size, f"{out_path} was packed with vocab_size={shard_vocab_size}, the tokenizer has {vocab_size}"
            assert shard_tokenizer_id == active_tokenizer_id, f"{out_path} was packed with a different tokenizer: delete the .bin shards to re-pack"
        else:
            todo.append(job)
    print(f"{what}: {len(jobs) - len(todo)} of {len(jobs)} shards exist, {len(todo)} to pack with {args.num_workers} workers")
    return todo

row_capacity = args.row_len + 1 # a row's targets are its inputs shifted by one
jobs = [(parquet_path, row_capacity, vocab_size, active_tokenizer_id, os.path.join(dataset_dir, f"train_{i:05d}.bin")) for i, parquet_path in enumerate(train_parquets)]
jobs.append((val_parquet, row_capacity, vocab_size, active_tokenizer_id, os.path.join(dataset_dir, "val_00000.bin")))
todo = remaining_jobs(jobs, "pretraining shards")
t0 = time.time()
if todo:
    with Pool(args.num_workers, initializer=init_worker, initargs=(tokenizer_dir,)) as pool:
        for name, num_rows, raw_tokens in pool.imap_unordered(pack_shard, todo):
            packed_tokens = num_rows * row_capacity
            crop_frac = 1 - packed_tokens / raw_tokens
            print(f"{name}: {num_rows:,} rows, {packed_tokens:,} of {raw_tokens:,} tokens ({100 * crop_frac:.1f}% cropped)")
pack_time = time.time() - t0

# -----------------------------------------------------------------------------
# The stage record (see harness/experiment.py): what is on disk now
train_row_len, train_rows, _, _ = split_info("train")
_, val_rows, _, _ = split_info("val")
print(format_record("summary",
    dataset=dataset_name,
    vocab_size=vocab_size,
    row_len=train_row_len,
    train_shards=len(train_parquets),
    train_rows=train_rows,
    train_tokens=train_rows * train_row_len,
    val_rows=val_rows,
    tokenizer_train_time_sec=round(tokenizer_train_time, 1),
    pack_time_sec=round(pack_time, 1),
))
