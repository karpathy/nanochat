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

import threading
from queue import Queue

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

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
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


class _BatchBuilder:
    """
    Stateful batch builder for the dataloader.

    Owns the document buffer, tokenizer and dataset iterator, and packs one
    batch of rows into buffer slots. Splitting this out from the generator
    lets us run the builder in a background thread (see _PrefetchDataloader)
    so parquet reads, tokenization and packing overlap with GPU compute
    instead of stalling the training loop.
    """

    def __init__(self, tokenizer, B, T, split, tokenizer_threads, tokenizer_batch_size, resume_state_dict, buffer_size):
        self.tokenizer = tokenizer
        self.B = B
        self.T = T
        self.row_capacity = T + 1
        self.tokenizer_threads = tokenizer_threads
        self.buffer_size = buffer_size
        self.batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
        self.bos_token = tokenizer.get_bos_token_id()
        self.doc_buffer = []
        self.pq_idx, self.rg_idx, self.epoch = 0, 0, 1

    def _refill_buffer(self):
        doc_batch, (self.pq_idx, self.rg_idx, self.epoch) = next(self.batches)
        token_lists = self.tokenizer.encode(doc_batch, prepend=self.bos_token, num_threads=self.tokenizer_threads)
        for tokens in token_lists:
            self.doc_buffer.append(tokens)

    def _pack_rows(self, row_buffer):
        B, row_capacity, buffer_size = self.B, self.row_capacity, self.buffer_size
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                while len(self.doc_buffer) < buffer_size:
                    self._refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(self.doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = self.doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    shortest_idx = min(range(len(self.doc_buffer)), key=lambda i: len(self.doc_buffer[i]))
                    doc = self.doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

    def build(self, row_buffer, cpu_buffer, gpu_buffer, stream=None):
        """
        Pack one batch into the given buffers and copy it to the device.
        - row_buffer: (B, T+1) CPU tensor used to pack rows into
        - cpu_buffer: (2*B*T,) pinned CPU staging buffer
        - gpu_buffer: (2*B*T,) device buffer receiving [inputs|targets]
        Returns (inputs, targets, state_dict). When stream is not None the HtoD
        copy is enqueued on that stream (used to share the consumer's CUDA
        stream from the producer thread so slot reuse is safely ordered).
        """
        B, T = self.B, self.T
        row_buffer = row_buffer.view(B, T + 1)
        self._pack_rows(row_buffer)
        cpu_inputs = cpu_buffer[:B * T].view(B, T)
        cpu_targets = cpu_buffer[B * T:].view(B, T)
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        state_dict = {"pq_idx": self.pq_idx, "rg_idx": self.rg_idx, "epoch": self.epoch}
        if stream is not None:
            with torch.cuda.stream(stream):
                gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        else:
            gpu_buffer.copy_(cpu_buffer)
        inputs = gpu_buffer[:B * T].view(B, T)
        targets = gpu_buffer[B * T:].view(B, T)
        return inputs, targets, state_dict


class _PrefetchDataloader:
    """
    Background-thread dataloader for CUDA.

    Runs the _BatchBuilder in a daemon thread and hands batches out through a
    queue, so parquet reads, tokenization and row packing overlap with the
    GPU's forward/backward passes. A small pool of preallocated buffer slots is reused;
    a slot is only handed back to the producer once the consumer has finished with it.

    The producer shares the consumer's CUDA stream, so reusing a slot is always safe.
    """

    def __init__(self, builder, n_slots, device, B, T):
        self._ready = Queue(maxsize=n_slots)
        self._free = Queue()
        self._slots = []  # each slot: (row_buffer, cpu_buffer, gpu_buffer)
        pin_memory = device == "cuda"
        for _ in range(n_slots):
            row_buffer = torch.empty(B * (T + 1), dtype=torch.long, pin_memory=pin_memory)
            cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=pin_memory)
            gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
            self._slots.append((row_buffer, cpu_buffer, gpu_buffer))
            self._free.put(len(self._slots) - 1)
        self._error = None
        self._stream = torch.cuda.current_stream(device) if device == "cuda" else None
        self._thread = threading.Thread(target=self._produce, args=(builder,), daemon=True)
        self._thread.start()

    def _produce(self, builder):
        try:
            while True:
                slot_idx = self._free.get()
                row_buffer, cpu_buffer, gpu_buffer = self._slots[slot_idx]
                inputs, targets, state_dict = builder.build(row_buffer, cpu_buffer, gpu_buffer, stream=self._stream)
                self._ready.put((inputs, targets, state_dict, slot_idx))
        except Exception as e:
            self._error = e

    def get(self):
        # block until a batch is ready. Returns (inputs, targets, state_dict, slot_idx).
        if self._error is not None:
            raise self._error
        return self._ready.get()

    def release(self, slot_idx):
        self._free.put(slot_idx)


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000, prefetch_batches=2,
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

    On CUDA the batch builder runs in a background thread with prefetch_batches
    buffer slots, so data prep overlaps the GPU's forward/backward passes. On
    CPU/MPS (or prefetch_batches=0) batches are built synchronously.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    builder = _BatchBuilder(tokenizer, B, T, split, tokenizer_threads, tokenizer_batch_size, resume_state_dict, buffer_size)

    if device == "cuda" and prefetch_batches > 0:
        # CUDA: background producer thread so data prep overlaps forward/backward
        loader = _PrefetchDataloader(builder, prefetch_batches, device, B, T)
        while True:
            inputs, targets, state_dict, slot_idx = loader.get()
            yield inputs, targets, state_dict
            loader.release(slot_idx)
    else:
        # CPU / MPS (or prefetch disabled): build batches synchronously
        row_buffer = torch.empty(B * (T + 1), dtype=torch.long)
        cpu_buffer = torch.empty(2 * B * T, dtype=torch.long)
        gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
        while True:
            inputs, targets, state_dict = builder.build(row_buffer, cpu_buffer, gpu_buffer, stream=None)
            yield inputs, targets, state_dict


def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets
