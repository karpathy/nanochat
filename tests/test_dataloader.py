"""
Unit tests for nanochat.dataloader (dtypes, contiguity, CUDA device detection, and best-fit parity).

python -m pytest tests/test_dataloader.py -v
"""

import torch

import nanochat.dataloader as dl
from nanochat.gpt import GPT, GPTConfig
from nanochat.loss_eval import evaluate_bpb


class _MockTokenizer:
    def get_bos_token_id(self):
        return 0

    def encode(self, docs, prepend=None, num_threads=1):
        out = []
        for doc in docs:
            toks = [int(x) for x in doc]
            if prepend is not None:
                toks = [prepend] + toks
            out.append(toks)
        return out


def _reference_bestfit_batches(docs_stream, bos_token, B, T, buffer_size, num_batches):
    row_capacity = T + 1
    doc_buffer = []
    stream_iter = iter(docs_stream)

    def refill():
        batch, _ = next(stream_iter)
        for doc in batch:
            doc_buffer.append([bos_token] + [int(x) for x in doc])

    expected = []
    for _ in range(num_batches):
        row = torch.empty((B, row_capacity), dtype=torch.long)
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill()
                remaining = row_capacity - pos
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    if len(doc) <= remaining and len(doc) > best_len:
                        best_idx = i
                        best_len = len(doc)
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining
        expected.append((row[:, :-1].clone(), row[:, 1:].clone()))
    return expected


def test_bos_bestfit_dtypes_contiguity_and_parity(monkeypatch):
    # Deterministic synthetic document lengths exercising both full fits and cropping
    synthetic_batches = [
        ([[10 + (k + i) % 50 for k in range(3 + (i * 5) % 19)] for i in range(16)], (0, rg, 1))
        for rg in range(20)
    ]

    def fake_doc_batches(split, resume_state_dict, tokenizer_batch_size):
        assert split == "train"
        for item in synthetic_batches:
            yield item

    monkeypatch.setattr(dl, "_document_batches", fake_doc_batches)

    B, T, buffer_size, num_batches = 4, 16, 12, 6
    expected = _reference_bestfit_batches(synthetic_batches, 0, B, T, buffer_size, num_batches)
    loader = dl.tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer=_MockTokenizer(),
        B=B,
        T=T,
        split="train",
        device="cpu",
        buffer_size=buffer_size,
    )

    for step in range(num_batches):
        x, y, state = next(loader)
        assert x.dtype == torch.int32
        assert y.dtype == torch.int64
        assert x.is_contiguous() and y.is_contiguous()
        assert torch.all(x[:, 0] == 0)
        assert torch.equal(y[:, :-1], x[:, 1:].long())

        exp_x, exp_y = expected[step]
        assert torch.equal(x.long(), exp_x)
        assert torch.equal(y, exp_y)
        assert "pq_idx" in state and "rg_idx" in state and "epoch" in state


def test_gpt_and_evaluate_bpb_int32_parity():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        window_pattern="L",
    )

    torch.manual_seed(7)
    m32 = GPT(cfg).to(device)
    m32.init_weights()
    torch.manual_seed(7)
    m64 = GPT(cfg).to(device)
    m64.init_weights()

    row = torch.randint(1, 64, (2, 17), dtype=torch.int32, device=device)
    row[:, 0] = 0
    x32 = row[:, :-1].contiguous()
    x64 = x32.long().contiguous()
    y64 = row[:, 1:].long().contiguous()

    loss32 = m32(x32, targets=y64)
    loss32.backward()
    loss64 = m64(x64, targets=y64)
    loss64.backward()

    assert abs(loss32.item() - loss64.item()) == 0.0
    for (_, p1), (_, p2) in zip(m32.named_parameters(), m64.named_parameters()):
        if p1.grad is not None:
            assert (p1.grad - p2.grad).abs().max().item() == 0.0

    token_bytes = torch.full((64,), 2, dtype=torch.int32, device=device)
    token_bytes[0] = 0
    bpb32 = evaluate_bpb(m32, iter([(x32, y64)]), steps=1, token_bytes=token_bytes)
    bpb64 = evaluate_bpb(m64, iter([(x64, y64)]), steps=1, token_bytes=token_bytes)
    assert abs(bpb32 - bpb64) == 0.0


def test_device_cuda_detection_handles_torch_device_object():
    # compute_init() returns torch.device("cuda", local_rank), where `device == "cuda"` is False.
    dev_obj = torch.device("cuda", 0)
    assert (dev_obj == "cuda") is False
    assert torch.device(dev_obj).type == "cuda"
    assert torch.device("cuda").type == "cuda"
    assert torch.device("cpu").type == "cpu"
