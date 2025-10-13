import torch
from dataclasses import replace

from nanochat.gpt import GPTConfig, MLP


def _mlp_config(**kwargs):
    cfg = GPTConfig(
        n_embd=32,
        mlp_type="relu2",
        mlp_width_mult=4.0,
        mlp_glu_width_mult=None,
        n_layer=1,
        n_head=4,
        n_kv_head=4,
        sequence_len=8,
        vocab_size=128,
    )
    return replace(cfg, **kwargs)


def test_relu2_mlp_shape():
    cfg = _mlp_config(mlp_type="relu2")
    mlp = MLP(cfg)
    assert mlp.c_fc.weight.shape[0] == int(cfg.mlp_width_mult * cfg.n_embd)
    x = torch.randn(2, 5, cfg.n_embd)
    out = mlp(x)
    assert out.shape == x.shape


def test_swiglu_width_scaling_defaults_to_two_thirds():
    cfg = _mlp_config(mlp_type="swiglu")
    mlp = MLP(cfg)
    expected_hidden = int(cfg.n_embd * cfg.mlp_width_mult * (2.0 / 3.0))
    assert mlp.c_fc.weight.shape[0] == 2 * expected_hidden
    x = torch.randn(4, 3, cfg.n_embd)
    out = mlp(x)
    assert out.shape == x.shape


def test_geglu_respects_custom_width():
    cfg = _mlp_config(mlp_type="geglu", mlp_glu_width_mult=0.5)
    mlp = MLP(cfg)
    expected_hidden = int(cfg.n_embd * 0.5)
    assert mlp.c_fc.weight.shape[0] == 2 * expected_hidden
    x = torch.randn(3, 2, cfg.n_embd)
    out = mlp(x)
    assert out.shape == x.shape
