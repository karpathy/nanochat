from __future__ import annotations

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


def _fill_row_from_docs(token_docs: list[list[int]], row_capacity: int) -> list[int]:
    row: list[int] = []
    while len(row) < row_capacity:
        if not token_docs:
            raise RuntimeError("Token buffer exhausted while building dataset-backed row")
        doc = token_docs.pop(0)
        take = min(len(doc), row_capacity - len(row))
        row.extend(doc[:take])
    return row


def build_dataset_backed_batch(batch_size: int, seq_len: int, split: str = "train", tokenizer_threads: int = 4):
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    row_capacity = seq_len + 1
    token_docs: list[list[int]] = []
    docs_used = 0
    doc_iter = parquets_iter_batched(split)
    rows: list[list[int]] = []

    while len(rows) < batch_size:
        while not token_docs:
            texts = next(doc_iter)
            encoded = tokenizer.encode(texts, prepend=bos_token, num_threads=tokenizer_threads)
            token_docs.extend(encoded)
        rows.append(_fill_row_from_docs(token_docs, row_capacity))
        docs_used += 1

    batch = mx.array(rows, dtype=mx.int32)
    return batch[:, :-1], batch[:, 1:], {"mode": "dataset", "split": split, "documents_used": docs_used}


def build_input_batch(input_mode: str, batch_size: int, seq_len: int, vocab_size: int, bos_token_id: int | None, dataset_split: str = "train", tokenizer_threads: int = 4):
    if input_mode == "repeated":
        return build_repeated_reference_batch(batch_size, seq_len, vocab_size, bos_token_id)
    if input_mode == "dataset":
        return build_dataset_backed_batch(batch_size, seq_len, split=dataset_split, tokenizer_threads=tokenizer_threads)
    raise ValueError(f"Unsupported input_mode: {input_mode}")