"""
Distributed dataloaders for pretraining.

Two implementations are provided:

1. Original (tokenizing_distributed_data_loader):
   - Streams tokens into a flat buffer, reshapes to (B, T)
   - Rows may start mid-document (no guaranteed BOS at position 0)
   - 100% token utilization, simple and efficient

2. BOS-aligned bestfit (tokenizing_distributed_data_loader_bos_bestfit):
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

The tradeoff: BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.
(2) is the new default if you have enough data.
Fallback to (1) if you have very limited data AND long documents.
"""

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files

def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    parquet_paths = list_parquet_files()
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state(tokenizer, B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This is the original dataloader that streams tokens into a flat buffer and reshapes.
    Rows may start mid-document (no guaranteed BOS at position 0).

    Supports approximate resume via state_dict.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    needed_tokens = B * T + 1  # +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()
    token_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    while True:

        # Accumulate enough tokens
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        tokens = token_buffer[:needed_tokens] # Read B*T+1 tokens (+1 is only for the target for the last token)
        token_buffer = token_buffer[B*T:] # Advance by B*T tokens, so we move exactly one window of B*T tokens over

        # Package tokens into inputs and targets, yield
        use_cuda = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda)
        inputs = scratch[:-1].view(B, T).to(device=device, non_blocking=use_cuda)
        targets = scratch[1:].view(B, T).to(device=device, non_blocking=use_cuda)
        yield inputs, targets, {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}


def tokenizing_distributed_data_loader(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets


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
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    while True:
        rows = []
        for _ in range(B):
            row = []
            while len(row) < row_capacity:
                # Ensure buffer has documents
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row)

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
                    row.extend(doc)
                else:
                    # No doc fits - crop first doc to fill remaining
                    doc = doc_buffer.pop(0)
                    row.extend(doc[:remaining])

            rows.append(row[:row_capacity])

        use_cuda = device == "cuda"
        batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_tensor[:, :-1].to(device=device, non_blocking=use_cuda)
        targets = batch_tensor[:, 1:].to(device=device, non_blocking=use_cuda)

        yield inputs, targets, {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}


def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets
