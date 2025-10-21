import torch
from dataclasses import replace

from nanochat.gpt import GPT, GPTConfig


def _tiny_config(**kwargs):
    base = GPTConfig(
        sequence_len=16,
        vocab_size=128,
        n_layer=1,
        n_head=4,
        n_kv_head=2,
        n_embd=64,
        use_fused_qkv=False,
        mlp_type="relu2",
    )
    return replace(base, **kwargs)


def test_fused_qkv_matches_legacy_split_projection():
    torch.manual_seed(0)
    split_cfg = _tiny_config(use_fused_qkv=False)
    fused_cfg = replace(split_cfg, use_fused_qkv=True)

    split_model = GPT(split_cfg)
    split_model.init_weights()
    state = split_model.state_dict()

    fused_model = GPT(fused_cfg)
    fused_model.init_weights()
    fused_model.load_state_dict(state, strict=True)

    tokens = torch.randint(0, split_cfg.vocab_size, (2, 5))
    with torch.no_grad():
        logits_split = split_model(tokens)
        logits_fused = fused_model(tokens)

    assert torch.allclose(logits_split, logits_fused, atol=1e-5)


def test_split_loads_from_fused_state_dict():
    torch.manual_seed(1)
    fused_cfg = _tiny_config(use_fused_qkv=True)
    fused_model = GPT(fused_cfg)
    fused_model.init_weights()
    state = fused_model.state_dict()

    split_cfg = replace(fused_cfg, use_fused_qkv=False)
    split_model = GPT(split_cfg)
    split_model.init_weights()
    split_model.load_state_dict(state, strict=True)

    tokens = torch.randint(0, split_cfg.vocab_size, (1, 7))
    with torch.no_grad():
        logits_split = split_model(tokens)
        logits_fused = fused_model(tokens)

    assert torch.allclose(logits_split, logits_fused, atol=1e-5)
