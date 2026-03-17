from __future__ import annotations

from collections import deque

import mlx.core as mx

from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import get_tokenizer


def build_repeated_reference_batch(batch_size: int, seq_len: int, vocab_size: int, bos_token_id: int | None):
    seed_ids = [
        bos_token_id if bos_token_id is not None else 1,
        17,
        29,
        113,
        509,
        997,
        4093,
        8191,
    ]
    seed_ids = [token_id % vocab_size for token_id in seed_ids]
    repeated = (seed_ids * ((seq_len // len(seed_ids)) + 1))[: seq_len + 1]
    batch = mx.array([repeated for _ in range(batch_size)], dtype=mx.int32)
    return batch[:, :-1], batch[:, 1:], {"mode": "repeated", "documents_used": 0}


def _fill_row_from_docs(token_docs: deque[list[int]], row_capacity: int) -> tuple[list[int], int, int]:
    row: list[int] = []
    documents_touched = 0
    documents_completed = 0
    while len(row) < row_capacity:
        if not token_docs:
            raise RuntimeError("Token buffer exhausted while building dataset-backed row")
        doc = token_docs[0]
        take = min(len(doc), row_capacity - len(row))
        row.extend(doc[:take])
        documents_touched += 1
        if take == len(doc):
            token_docs.popleft()
            documents_completed += 1
        else:
            token_docs[0] = doc[take:]
    return row, documents_touched, documents_completed


class RepeatedBatchProvider:
    def __init__(self, batch_size: int, seq_len: int, vocab_size: int, bos_token_id: int | None):
        self.inputs, self.targets, self.metadata = build_repeated_reference_batch(batch_size, seq_len, vocab_size, bos_token_id)

    def next_batch(self):
        metadata = dict(self.metadata)
        metadata["fresh_batch_each_call"] = False
        return self.inputs, self.targets, metadata


class DatasetBatchProvider:
    def __init__(self, batch_size: int, seq_len: int, split: str = "train", tokenizer_threads: int = 4):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.split = split
        self.tokenizer_threads = tokenizer_threads
        self.tokenizer = get_tokenizer()
        self.bos_token = self.tokenizer.get_bos_token_id()
        self.row_capacity = seq_len + 1
        self.token_docs: deque[list[int]] = deque()
        self.doc_iter = parquets_iter_batched(split)
        self.parquet_batches_loaded = 0
        self.documents_loaded_total = 0
        self.documents_completed_total = 0
        self.rows_emitted_total = 0

    def _load_more_docs(self) -> int:
        texts = next(self.doc_iter)
        encoded = self.tokenizer.encode(texts, prepend=self.bos_token, num_threads=self.tokenizer_threads)
        self.token_docs.extend(encoded)
        self.parquet_batches_loaded += 1
        self.documents_loaded_total += len(encoded)
        return len(encoded)

    def next_batch(self):
        rows: list[list[int]] = []
        documents_touched = 0
        documents_loaded_before = self.documents_loaded_total
        documents_completed_before = self.documents_completed_total
        parquet_batches_before = self.parquet_batches_loaded

        while len(rows) < self.batch_size:
            while not self.token_docs:
                self._load_more_docs()
            row, row_documents_touched, row_documents_completed = _fill_row_from_docs(self.token_docs, self.row_capacity)
            rows.append(row)
            documents_touched += row_documents_touched
            self.documents_completed_total += row_documents_completed
            self.rows_emitted_total += 1

        batch = mx.array(rows, dtype=mx.int32)
        metadata = {
            "mode": "dataset",
            "split": self.split,
            "fresh_batch_each_call": True,
            "documents_touched": documents_touched,
            "documents_loaded": self.documents_loaded_total - documents_loaded_before,
            "documents_completed": self.documents_completed_total - documents_completed_before,
            "documents_loaded_total": self.documents_loaded_total,
            "documents_completed_total": self.documents_completed_total,
            "parquet_batches_loaded": self.parquet_batches_loaded - parquet_batches_before,
            "parquet_batches_loaded_total": self.parquet_batches_loaded,
            "buffered_documents": len(self.token_docs),
            "rows_emitted_total": self.rows_emitted_total,
        }
        return batch[:, :-1], batch[:, 1:], metadata


def build_dataset_backed_batch(batch_size: int, seq_len: int, split: str = "train", tokenizer_threads: int = 4):
    provider = DatasetBatchProvider(batch_size, seq_len, split=split, tokenizer_threads=tokenizer_threads)
    return provider.next_batch()


def make_input_batch_provider(input_mode: str, batch_size: int, seq_len: int, vocab_size: int, bos_token_id: int | None, dataset_split: str = "train", tokenizer_threads: int = 4):
    if input_mode == "repeated":
        return RepeatedBatchProvider(batch_size, seq_len, vocab_size, bos_token_id)
    if input_mode == "dataset":
        return DatasetBatchProvider(batch_size, seq_len, split=dataset_split, tokenizer_threads=tokenizer_threads)
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def build_input_batch(input_mode: str, batch_size: int, seq_len: int, vocab_size: int, bos_token_id: int | None, dataset_split: str = "train", tokenizer_threads: int = 4):
    provider = make_input_batch_provider(
        input_mode,
        batch_size,
        seq_len,
        vocab_size,
        bos_token_id,
        dataset_split=dataset_split,
        tokenizer_threads=tokenizer_threads,
    )
    return provider.next_batch()