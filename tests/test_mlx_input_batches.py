from collections import deque

import pytest

pytest.importorskip("mlx.core")

from dev.mlx_input_batches import DatasetBatchProvider, _fill_row_from_docs


class FakeTokenizer:
    def __init__(self):
        self.docs = {
            "doc1": [1, 2],
            "doc2": [3, 4],
            "doc3": [5, 6],
        }

    def get_bos_token_id(self):
        return 99

    def encode(self, texts, prepend=None, num_threads=None):
        encoded = []
        for text in texts:
            row = list(self.docs[text])
            if prepend is not None:
                row.insert(0, prepend)
            encoded.append(row)
        return encoded


def test_fill_row_from_docs_preserves_partial_document_tail():
    token_docs = deque([[10, 11, 12, 13]])

    row, documents_touched, documents_completed = _fill_row_from_docs(token_docs, row_capacity=3)

    assert row == [10, 11, 12]
    assert documents_touched == 1
    assert documents_completed == 0
    assert list(token_docs) == [[13]]

    row, documents_touched, documents_completed = _fill_row_from_docs(token_docs, row_capacity=1)

    assert row == [13]
    assert documents_touched == 1
    assert documents_completed == 1
    assert list(token_docs) == []


def test_dataset_batch_provider_streams_forward_across_calls(monkeypatch):
    fake_tokenizer = FakeTokenizer()

    def fake_parquets_iter_batched(split):
        assert split == "train"
        yield ["doc1", "doc2"]
        yield ["doc3"]

    monkeypatch.setattr("dev.mlx_input_batches.get_tokenizer", lambda: fake_tokenizer)
    monkeypatch.setattr("dev.mlx_input_batches.parquets_iter_batched", fake_parquets_iter_batched)

    provider = DatasetBatchProvider(batch_size=1, seq_len=2, split="train")

    inputs_1, targets_1, metadata_1 = provider.next_batch()
    inputs_2, targets_2, metadata_2 = provider.next_batch()
    inputs_3, targets_3, metadata_3 = provider.next_batch()

    assert inputs_1.tolist() == [[99, 1]]
    assert targets_1.tolist() == [[1, 2]]
    assert metadata_1["documents_loaded"] == 2
    assert metadata_1["documents_completed"] == 1

    assert inputs_2.tolist() == [[99, 3]]
    assert targets_2.tolist() == [[3, 4]]
    assert metadata_2["documents_loaded"] == 0
    assert metadata_2["documents_completed"] == 1

    assert inputs_3.tolist() == [[99, 5]]
    assert targets_3.tolist() == [[5, 6]]
    assert metadata_3["documents_loaded"] == 1
    assert metadata_3["documents_completed"] == 1
    assert metadata_3["rows_emitted_total"] == 3