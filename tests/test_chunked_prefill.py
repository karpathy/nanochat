"""Cached model logits must not depend on how the prompt is chunked."""
import pytest
import torch

from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig


@pytest.mark.parametrize("chunks", [(2, 4), (1, 3, 2), (3, 2, 1)])
@torch.no_grad()
def test_chunked_prefill_preserves_token_smearing(chunks, monkeypatch):
    # Exercise CPU/float32 consistently even on CUDA-capable test hosts.
    monkeypatch.setattr("nanochat.gpt.COMPUTE_DTYPE", torch.float32)
    monkeypatch.setattr("nanochat.flash_attention.USE_FA3", False)
    torch.manual_seed(42)
    config = GPTConfig(sequence_len=16, vocab_size=64, n_layer=2,
                       n_head=2, n_kv_head=2, n_embd=32, window_pattern="L")
    model = GPT(config)
    model.init_weights()
    model.eval()
    # The initialization disables smearing, which would hide this regression.
    model.smear_lambda.fill_(0.7)
    for block in model.transformer.h:
        block.attn.c_proj.weight.normal_(std=0.02)
        block.mlp.c_proj.weight.normal_(std=0.02)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
    expected = model(tokens)

    def cached_logits(sizes):
        cache = KVCache(batch_size=2, num_heads=2, seq_len=16, head_dim=16,
                        num_layers=2, device="cpu", dtype=torch.float32)
        outputs = []
        position = 0
        for size in sizes:
            outputs.append(model(tokens[:, position:position + size], kv_cache=cache))
            position += size
        assert cache.get_pos() == tokens.size(1)
        return torch.cat(outputs, dim=1)

    torch.testing.assert_close(cached_logits((6,)), expected, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(cached_logits((1, 1, 1, 1, 1, 1)), expected, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(cached_logits(chunks), expected, rtol=1e-4, atol=1e-6)
