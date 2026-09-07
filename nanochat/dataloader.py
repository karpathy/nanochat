"""
The pretraining data format and its loader.

scripts/base_prepare.py (and chat_prepare.py for the SFT rows) writes a dataset once, as
shards of packed rows; this module owns the
shard format (write_shard / read_shard) and the loader that trains on those shards.

Shard format (v1), language-agnostic:
    256-byte header of int32: magic, version, row_len, num_rows, vocab_size, tokenizer_id, then zeros
    inputs   uint16 (num_rows, row_len)   what the model sees
    targets  uint16 (num_rows, row_len)   what it must predict; IGNORE = do not train here
Targets are stored explicitly (not derived by shifting inputs) so the loader has no
logic and "do not train on this position" is just a target value (SFT masking, padding).
The version fixes which sections follow; later versions append sections. Read it from
anywhere, e.g. numpy:
    header = np.fromfile(path, dtype=np.int32, count=64)
    inputs = np.memmap(path, dtype=np.uint16, mode="r", offset=256, shape=(num_rows, row_len))

Rows are BOS-aligned: every row starts at a document start (see the packing in
base_prepare.py), so no row ever begins mid-document. The row length is fixed at prepare
time; training at any max_seq_len <= row_len uses the first max_seq_len columns of every
row. The shards of a split are $NANOCHAT_BASE_DIR/datasets/<name>/<split>_NNNNN.bin.
"""

import os
import glob
from itertools import islice

import numpy as np
import torch

from nanochat.common import get_dist_info, print0, get_dataset_dir
from nanochat.tokenizer import get_tokenizer_dir, tokenizer_id

MAGIC = 20260902
VERSION = 1
HEADER_BYTES = 256
HEADER_INTS = HEADER_BYTES // 4
IGNORE = 65535 # a target that is never trained on; vocab ids must stay below it

# -----------------------------------------------------------------------------
# The shard format

def write_shard(path, inputs, targets, vocab_size, tokenizer_id):
    """Write one shard atomically (tmp file, then rename). inputs/targets: (num_rows, row_len) uint16."""
    assert inputs.shape == targets.shape, "inputs and targets must have the same shape"
    assert inputs.dtype == np.uint16 and targets.dtype == np.uint16, "sections are uint16"
    assert vocab_size < IGNORE, f"vocab_size {vocab_size} must leave {IGNORE} free as the ignore target"
    num_rows, row_len = inputs.shape
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[:6] = [MAGIC, VERSION, row_len, num_rows, vocab_size, tokenizer_id]
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(header.tobytes())
        f.write(np.ascontiguousarray(inputs).tobytes())
        f.write(np.ascontiguousarray(targets).tobytes())
    os.replace(tmp_path, path)

def read_header(path):
    """Returns (row_len, num_rows, vocab_size, tokenizer_id) of a shard, checking magic and version."""
    header = np.fromfile(path, dtype=np.int32, count=HEADER_INTS)
    magic, version, row_len, num_rows, vocab_size, tokenizer_id = (int(v) for v in header[:6])
    assert magic == MAGIC, f"{path} is not a nanochat shard (magic {magic} != {MAGIC})"
    assert version == VERSION, f"{path} is shard format v{version}, this code reads v{VERSION}"
    return row_len, num_rows, vocab_size, tokenizer_id

def read_shard(path):
    """Returns (inputs, targets): read-only memmaps of shape (num_rows, row_len). Nothing is loaded."""
    row_len, num_rows, vocab_size, tokenizer_id = read_header(path)
    section_bytes = num_rows * row_len * np.dtype(np.uint16).itemsize
    inputs = np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_BYTES, shape=(num_rows, row_len))
    targets = np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_BYTES + section_bytes, shape=(num_rows, row_len))
    return inputs, targets

# -----------------------------------------------------------------------------
# Packing documents into rows (used by scripts/base_prepare.py and chat_prepare.py)

def pack_rows(docs, row_capacity, crop=True, buffer_size=1000):
    """
    BOS-aligned best-fit packing. For each row: repeatedly take the largest buffered
    document that fits entirely. When none fits, crop=True crops the shortest one to fill
    the row exactly (pretraining: no padding, ~15% of climbmix tokens lost this way at
    row length 2048); crop=False ends the row short and the caller pads it (SFT: a reply
    is never separated from its prompt). docs: iterator of sequences, each starting at a
    document start and at most row_capacity long. Yields rows as lists. With crop=True
    every row is exactly row_capacity long and a final partial row is dropped.
    """
    buffer = [] # the documents best-fit chooses from
    lens = np.zeros(0, dtype=np.int64) # their lengths, kept in sync, so the search is vectorized
    while True:
        row = []
        while len(row) < row_capacity:
            if len(buffer) < buffer_size:
                new_docs = list(islice(docs, buffer_size - len(buffer)))
                new_lens = np.array([len(doc) for doc in new_docs], dtype=np.int64)
                buffer.extend(new_docs)
                lens = np.concatenate([lens, new_lens])
            if len(buffer) == 0:
                break # the source is exhausted
            remaining = row_capacity - len(row)
            fits = lens <= remaining
            if fits.any():
                idx = int(np.argmax(np.where(fits, lens, 0))) # the largest doc that fits entirely
                row.extend(buffer.pop(idx))
            elif crop:
                idx = int(np.argmin(lens)) # crop the shortest to fill the row exactly
                row.extend(buffer.pop(idx)[:remaining])
            else:
                break # nothing fits and nothing may be cropped: the row ends short
            lens = np.delete(lens, idx)
        if len(row) == 0:
            return # the source is exhausted
        if crop and len(row) < row_capacity:
            return # the source ran out mid-row: drop the partial row
        yield row

# -----------------------------------------------------------------------------
# The shards of a split

def list_shards(split, num_shards=None):
    """The shards of a split in order, e.g. train_00000.bin, train_00001.bin, ...
    num_shards limits to the first n, e.g. to multi-epoch over a subset of the data."""
    dataset_dir = get_dataset_dir()
    paths = sorted(glob.glob(os.path.join(dataset_dir, f"{split}_*.bin")))
    assert len(paths) > 0, f"No {split} shards in {dataset_dir}. Run: python -m scripts.base_prepare (chat_prepare for the SFT splits)"
    if num_shards is not None:
        paths = paths[:num_shards]
    return paths

def split_info(split, num_shards=None):
    """(row_len, total_rows, vocab_size, tokenizer_id) of a split, from its shard headers. Nothing is loaded."""
    paths = list_shards(split, num_shards)
    headers = [read_header(path) for path in paths]
    row_len, _, vocab_size, shards_tokenizer_id = headers[0]
    for path, header in zip(paths, headers):
        assert header[0] == row_len, f"{path} has row_len={header[0]}, the split's first shard has {row_len}"
        assert header[2] == vocab_size, f"{path} has vocab_size={header[2]}, the split's first shard has {vocab_size}"
        assert header[3] == shards_tokenizer_id, f"{path} was packed with a different tokenizer than the split's first shard"
    total_rows = sum(header[1] for header in headers)
    return row_len, total_rows, vocab_size, shards_tokenizer_id

# -----------------------------------------------------------------------------
# The loader

def data_loader(B, T, split, device="cuda", resume_row=0, num_shards=None):
    """
    Infinite iterator over (inputs, targets, row) batches of B rows, striding across DDP
    ranks: at each step rank r takes B consecutive rows starting at row + r*B, and row
    advances by B * world_size. row is the global index of this batch's first row across
    all ranks and is the resume state (one integer). inputs/targets are (B, T) int64 on
    device: the first T columns of each row, with IGNORE targets mapped to -1 (the loss's
    ignore_index). Wrapping around the split (an epoch) is silent by design but loud in
    the log.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    row_len, total_rows, vocab_size, shards_tokenizer_id = split_info(split, num_shards)
    tokenizer_dir = get_tokenizer_dir()
    assert shards_tokenizer_id == tokenizer_id(tokenizer_dir), f"the {split} shards were packed with a different tokenizer than the one in {tokenizer_dir}: delete the .bin shards and re-run prepare"
    assert T <= row_len, f"max_seq_len {T} exceeds the prepared row length {row_len}; re-run prepare with -T {T} or larger"
    paths = list_shards(split, num_shards)
    shards = [read_shard(path) for path in paths]
    rows_per_shard = [inputs.shape[0] for inputs, targets in shards]
    shard_starts = np.cumsum([0] + rows_per_shard) # global row index where each shard starts
    rows_per_step = B * ddp_world_size

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)] so that each
    # batch is a single async HtoD copy. The numpy views write straight into pinned memory.
    device = torch.device(device)
    use_cuda = device.type == "cuda"
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    copy_done = torch.cuda.Event() if use_cuda else None # marks when the async HtoD copy has finished reading cpu_buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T).numpy()
    cpu_targets = cpu_buffer[B * T:].view(B, T).numpy()
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    row = resume_row
    while True:
        epoch = row // total_rows + 1
        # wait for the previous async copy to finish reading cpu_buffer before overwriting it
        if use_cuda:
            copy_done.synchronize()
        for i in range(B):
            global_row = (row + ddp_rank * B + i) % total_rows
            shard_idx = int(np.searchsorted(shard_starts, global_row, side="right") - 1)
            local_row = global_row - int(shard_starts[shard_idx])
            shard_inputs, shard_targets = shards[shard_idx]
            cpu_inputs[i] = shard_inputs[local_row, :T]
            cpu_targets[i] = shard_targets[local_row, :T]
        cpu_targets[cpu_targets == IGNORE] = -1
        # Single async HtoD copy into the persistent GPU buffer and yield.
        # Reusing gpu_buffer is safe: the copy is stream-ordered after the consumer's kernels.
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        if use_cuda:
            copy_done.record()
        yield inputs, targets, row
        row += rows_per_step
        if row // total_rows + 1 != epoch:
            # a val wrap means metrics get computed on duplicated data; a train wrap is
            # usually an under-provisioned NUM_SHARDS (cropping and the data:param ratio raise demand)
            print0(f"WARNING: {split} split exhausted after {total_rows:,} rows, wrapping to epoch {epoch + 1}. If unintentional, prepare more shards (or reduce --eval-tokens for val).")

def data_loader_batches(*args, **kwargs):
    """Helper that omits the row from yields, for consumers that only want (inputs, targets)."""
    for inputs, targets, row in data_loader(*args, **kwargs):
        yield inputs, targets
