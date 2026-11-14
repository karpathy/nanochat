"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files

def _start_position(resume_state_dict):
    """Resume position for this rank. Only rank 0 writes the checkpoint, so the saved
    rg_idx is rank 0's; re-derive this rank's row group in the same stride block.
    doc_offset is an offset into rank 0's row group, so non-zero ranks start at 0."""
    _, ddp_rank, _, ddp_world_size = get_dist_info()
    state = resume_state_dict or {}
    saved_rg = state.get("rg_idx", ddp_rank)
    rg_idx = (saved_rg // ddp_world_size) * ddp_world_size + ddp_rank
    doc_offset = state.get("doc_offset", 0) if ddp_rank == 0 else 0
    return (state.get("pq_idx", 0), rg_idx, doc_offset, state.get("epoch", 1))


def _resume_doc_buffer(resume_state_dict):
    """Rank 0's saved buffer holds rank 0's documents; other ranks start with an empty buffer."""
    _, ddp_rank, _, _ = get_dist_info()
    if not resume_state_dict or ddp_rank != 0:
        return []
    return list(resume_state_dict.get("doc_buffer", []))


def _document_batches(split, start_position, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    start_position is (pq_idx, rg_idx, doc_offset, epoch) of the next unread slice.
    Each yield is (text_batch, next_position) in the same shape, so the stream can resume exactly.
    epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    _, ddp_rank, _, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    pq_idx, rg_idx, doc_offset, epoch = start_position

    while True:  # iterate infinitely (multi-epoch)
        while pq_idx < len(parquet_paths):
            pf = pq.ParquetFile(parquet_paths[pq_idx])
            while rg_idx < pf.num_row_groups:
                batch = pf.read_row_group(rg_idx).column('text').to_pylist()
                for i in range(doc_offset, len(batch), tokenizer_batch_size):
                    next_offset = i + tokenizer_batch_size
                    if next_offset < len(batch):
                        next_position = (pq_idx, rg_idx, next_offset, epoch)
                    else:
                        next_position = (pq_idx, rg_idx + ddp_world_size, 0, epoch)
                    yield batch[i:i+tokenizer_batch_size], next_position
                doc_offset = 0
                rg_idx += ddp_world_size
            pq_idx += 1
            rg_idx = ddp_rank
        pq_idx = 0
        rg_idx = ddp_rank
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Reduces token waste compared to simple greedy cropping by searching a buffer
    for documents that fit well, while maintaining 100% utilization (no padding).

    Algorithm for each row:
    1. From buffered docs, pick the LARGEST doc that fits entirely
    2. Repeat until no doc fits
    3. When nothing fits, crop a doc to fill remaining space exactly

    Key properties:
    - Every row starts with BOS
    - 100% utilization (no padding, every token is trained on)
    - Approximately 35% of all tokens are discarded due to cropping
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    start_position = _start_position(resume_state_dict)
    batches = _document_batches(split, start_position, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = _resume_doc_buffer(resume_state_dict)
    reader_position = start_position

    def refill_buffer():
        nonlocal reader_position
        doc_batch, reader_position = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        pq_idx, rg_idx, doc_offset, epoch = reader_position
        # State from before this batch: the training loop holds a prefetched batch, so resuming here re-yields it
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "doc_offset": doc_offset, "epoch": epoch, "doc_buffer": list(doc_buffer)}
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        # Single HtoD copy into persistent GPU buffer and yield
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets
