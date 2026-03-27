import torch

import nanochat.dataloader as dataloader


def test_length_bucket_buffer_best_fit_and_shortest():
    buffer = dataloader._LengthBucketBuffer()

    for length in [5, 2, 7, 5, 3]:
        buffer.append(torch.arange(length, dtype=torch.long))

    assert len(buffer) == 5
    assert buffer.pop_largest_fitting(6).size(0) == 5
    assert buffer.pop_largest_fitting(5).size(0) == 5
    assert buffer.pop_largest_fitting(4).size(0) == 3
    assert buffer.pop_largest_fitting(1) is None
    assert buffer.pop_shortest().size(0) == 2
    assert buffer.pop_shortest().size(0) == 7
    assert len(buffer) == 0


class _FakeTokenizer:
    def __init__(self):
        self.docs = {
            "doc_a": [1, 2, 3],
            "doc_b": [4, 5, 6],
        }

    def get_bos_token_id(self):
        return 99

    def encode(self, texts, prepend=None, num_threads=4):
        assert prepend == 99
        return [[prepend] + self.docs[text] for text in texts]


def test_bos_bestfit_loader_keeps_bos_alignment_when_cropping():
    original_document_batches = dataloader._document_batches

    def fake_document_batches(split, resume_state_dict, tokenizer_batch_size):
        while True:
            yield ["doc_a", "doc_b"], (7, 8, 9)

    dataloader._document_batches = fake_document_batches
    try:
        loader = dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit(
            _FakeTokenizer(),
            B=1,
            T=5,
            split="train",
            device="cpu",
            buffer_size=2,
        )
        inputs, targets, state_dict = next(loader)
    finally:
        dataloader._document_batches = original_document_batches

    assert inputs.shape == (1, 5)
    assert targets.shape == (1, 5)
    assert inputs[0].tolist() == [99, 1, 2, 3, 99]
    assert targets[0].tolist() == [1, 2, 3, 99, 4]
    assert state_dict == {"pq_idx": 7, "rg_idx": 8, "epoch": 9}
