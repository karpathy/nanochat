
import torch
import pytest
from nanochat.gpt import GPT, GPTConfig

@pytest.fixture
def gpt_config_small():
    return GPTConfig(
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        vocab_size=100,
        sequence_len=32
    )

def test_gpt_config_defaults():
    config = GPTConfig()
    assert config.n_layer == 12
    assert config.n_head == 6
    assert config.n_embd == 768

def test_gpt_init(gpt_config_small):
    model = GPT(gpt_config_small)
    assert model.transformer.wte.weight.shape == (100, 32)
    assert len(model.transformer.h) == 2
    # Check rotary buffers
    assert hasattr(model, 'cos')
    assert hasattr(model, 'sin')

def test_gpt_forward_inference(gpt_config_small):
    model = GPT(gpt_config_small)
    idx = torch.randint(0, 100, (2, 8)) # B=2, T=8
    logits = model(idx)
    assert logits.shape == (2, 8, 100)

def test_gpt_forward_training(gpt_config_small):
    model = GPT(gpt_config_small)
    idx = torch.randint(0, 100, (2, 8))
    targets = torch.randint(0, 100, (2, 8))
    loss = model(idx, targets=targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0

def test_gpt_forward_return_embeddings(gpt_config_small):
    model = GPT(gpt_config_small)
    idx = torch.randint(0, 100, (2, 8))
    targets = torch.randint(0, 100, (2, 8))
    loss, embeddings = model(idx, targets=targets, return_embeddings=True)
    assert embeddings.shape == (2, 8, 32) # n_embd=32

def test_gpt_generate(gpt_config_small):
    model = GPT(gpt_config_small)
    model.eval()
    start_tokens = [1, 2, 3]
    # We ask for 5 new tokens
    gen = model.generate(start_tokens, max_tokens=5, temperature=0.0)
    generated = list(gen)
    assert len(generated) == 5
    assert all(isinstance(x, int) for x in generated)

def test_gpt_kv_cache_inference(gpt_config_small):
    # This requires mocking KVCache or using engine.KVCache if available
    # But GPT.forward accepts kv_cache object.
    # Let's import KVCache from nanochat.engine if possible to test integration
    from nanochat.engine import KVCache

    # Setup
    batch_size = 2
    model = GPT(gpt_config_small)
    kv_cache = KVCache(
        batch_size=batch_size,
        num_heads=gpt_config_small.n_head,
        seq_len=gpt_config_small.sequence_len,
        head_dim=gpt_config_small.n_embd // gpt_config_small.n_head,
        num_layers=gpt_config_small.n_layer
    )

    # First token
    idx = torch.randint(0, 100, (batch_size, 1))
    logits = model(idx, kv_cache=kv_cache)
    assert logits.shape == (batch_size, 1, 100)
    assert kv_cache.get_pos() == 1

    # Second token
    idx = torch.randint(0, 100, (batch_size, 1))
    logits = model(idx, kv_cache=kv_cache)
    assert logits.shape == (batch_size, 1, 100)
    assert kv_cache.get_pos() == 2

def test_rotary_embedding_logic(gpt_config_small):
    # Directly test that rotary embeddings are applied
    # We can't easily access intermediate values without hooks or breaking open the class
    # But we can check if `cos` and `sin` are correctly initialized
    model = GPT(gpt_config_small)
    # Check shape of cos/sin
    # In GPT.__init__:
    # self.rotary_seq_len = config.sequence_len * 10
    # cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
    # cos/sin shape: (1, seq_len, 1, head_dim/2)

    expected_seq_len = gpt_config_small.sequence_len * 10
    head_dim = gpt_config_small.n_embd // gpt_config_small.n_head # 32 // 2 = 16
    expected_shape = (1, expected_seq_len, 1, head_dim // 2)

    assert model.cos.shape == expected_shape
    assert model.sin.shape == expected_shape
    assert model.cos.dtype == torch.bfloat16
