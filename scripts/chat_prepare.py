"""
Prepare the chat data: the SFT rows, once per dataset, after scripts/base_prepare.py
(it needs the tokenizer that base_prepare trained there).

The SFT data is general chat (SmolTalk) plus a multiple-choice format primer (MMLU's
auxiliary_train: the train splits of ARC, OBQA, RACE and MC-TEST). Models this small
don't do multiple choice out of the box; the primer teaches the "answer with a letter"
format that ChatCORE's categorical tasks score. Conversations are rendered to tokens
with the loss mask folded into the targets (IGNORE everywhere but the assistant's
tokens) and packed into rows of the same length and format as the pretraining shards
(nanochat/dataloader.py). Whole conversations only (a reply needs its prompt), so rows
are padded rather than cropped. Val is SmolTalk's test split.

    sft_train_NNNNN.bin (fixed-size chunks of conversations), sft_val_00000.bin

Shards that exist are skipped. Changing the mix (e.g. --primer-epochs) needs the
sft_*.bin shards deleted.

python -m scripts.chat_prepare
"""

import os
import time
import random
import argparse
from multiprocessing import Pool

import numpy as np

from harness.experiment import format_record, format_invocation
from nanochat.common import get_dataset_name, get_dataset_dir
from nanochat.tokenizer import RustBPETokenizer, get_tokenizer_dir
from nanochat.dataloader import write_shard, read_header, split_info, pack_rows, IGNORE
from harness.tasks import SmolTalk, MMLU

parser = argparse.ArgumentParser(description="Prepare the chat data: the packed SFT rows")
parser.add_argument("--primer-epochs", type=int, default=3, help="epochs of the multiple-choice format primer (MMLU auxiliary_train) in the SFT mix")
parser.add_argument("-w", "--num-workers", type=int, default=32, help="parallel workers for packing shards")
args = parser.parse_args()
print(format_invocation(args))

SHARD_CONVERSATIONS = 20_000 # conversations per SFT train shard (fixed, so shard names don't depend on -w)

# the tokenizer and the row geometry come from the pretraining shards, so the two agree by construction
dataset_name = get_dataset_name()
dataset_dir = get_dataset_dir()
tokenizer_dir = get_tokenizer_dir()
assert os.path.exists(os.path.join(tokenizer_dir, "tokenizer.pkl")), f"No tokenizer in {dataset_dir}: run python -m scripts.base_prepare first"
row_len, _, vocab_size, active_tokenizer_id = split_info("val")

def sft_conversations(split):
    """The conversations of an SFT split as (source, row) pairs in a fixed shuffled order,
    so the sources are mixed throughout training. The primer is oversampled by repetition."""
    if split == "sft_train":
        smoltalk = SmolTalk(split="train")
        primer = MMLU(split="auxiliary_train")
        pairs = [(smoltalk, row) for row in range(len(smoltalk))]
        pairs += [(primer, row) for _ in range(args.primer_epochs) for row in range(len(primer))]
        random.Random(42).shuffle(pairs)
    else:
        smoltalk = SmolTalk(split="test")
        pairs = [(smoltalk, row) for row in range(len(smoltalk))]
    return pairs

def init_worker(tokenizer_dir):
    global worker_tokenizer
    worker_tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

def pack_shard(job):
    """Render conversations [start, stop) of a split into padded rows and write the shard.
    Returns (name, num_rows, content_tokens)."""
    split, start, stop, out_path = job
    pairs = sft_conversations(split)
    bos = worker_tokenizer.get_bos_token_id()
    def docs():
        """Each conversation as (input, target) pairs: the target is the next token where
        the assistant speaks and IGNORE elsewhere, so the packer can treat it as one sequence."""
        for i in range(start, stop):
            source, row = pairs[i]
            ids, mask = worker_tokenizer.render_conversation(source[row])
            doc = [(ids[t], ids[t + 1] if mask[t + 1] else IGNORE) for t in range(len(ids) - 1)]
            yield doc[:row_len]
    rows = []
    content_tokens = 0
    for row in pack_rows(docs(), row_len, crop=False):
        content_tokens += len(row)
        row = row + [(bos, IGNORE)] * (row_len - len(row)) # pad: never trained on
        rows.append(np.array(row, dtype=np.uint16))
    rows = np.stack(rows) # (num_rows, row_len, 2)
    write_shard(out_path, rows[:, :, 0], rows[:, :, 1], vocab_size, active_tokenizer_id)
    return os.path.basename(out_path), rows.shape[0], content_tokens

num_train_conversations = len(sft_conversations("sft_train")) # constructing the datasets here also downloads them once, before the workers start
num_val_conversations = len(sft_conversations("sft_val"))
jobs = []
for i, start in enumerate(range(0, num_train_conversations, SHARD_CONVERSATIONS)):
    stop = min(start + SHARD_CONVERSATIONS, num_train_conversations)
    jobs.append(("sft_train", start, stop, os.path.join(dataset_dir, f"sft_train_{i:05d}.bin")))
jobs.append(("sft_val", 0, num_val_conversations, os.path.join(dataset_dir, "sft_val_00000.bin")))
todo = []
for job in jobs:
    out_path = job[-1]
    if os.path.exists(out_path):
        _, _, _, shard_tokenizer_id = read_header(out_path)
        assert shard_tokenizer_id == active_tokenizer_id, f"{out_path} was packed with a different tokenizer: delete the sft_*.bin shards to re-pack"
    else:
        todo.append(job)
print(f"sft shards: {len(jobs) - len(todo)} of {len(jobs)} exist, {len(todo)} to pack with {args.num_workers} workers")
t0 = time.time()
if todo:
    with Pool(args.num_workers, initializer=init_worker, initargs=(tokenizer_dir,)) as pool:
        for name, num_rows, content_tokens in pool.imap_unordered(pack_shard, todo):
            pad_frac = 1 - content_tokens / (num_rows * row_len)
            print(f"{name}: {num_rows:,} rows, {content_tokens:,} tokens ({100 * pad_frac:.1f}% padding)")
pack_time = time.time() - t0

# The stage record (see harness/experiment.py): what is on disk now
_, sft_train_rows, _, _ = split_info("sft_train")
_, sft_val_rows, _, _ = split_info("sft_val")
print(format_record("summary",
    dataset=dataset_name,
    row_len=row_len,
    sft_train_rows=sft_train_rows,
    sft_val_rows=sft_val_rows,
    primer_epochs=args.primer_epochs,
    pack_time_sec=round(pack_time, 1),
))
