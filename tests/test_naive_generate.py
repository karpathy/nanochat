"""
Test the naive GPT.generate path. Example run:

python -m pytest tests/test_naive_generate.py -v
"""

import torch
from nanochat.gpt import GPT, GPTConfig


def test_naive_generate_single_token_prompt():
    # Naive generate should accept a single-token prompt (e.g. just BOS for
    # unconditional sampling). The no-KV-cache smear path must treat T == 1 as a
    # no-op rather than asserting T > 1.
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=64, vocab_size=262, n_layer=2, n_head=4, n_kv_head=4, n_embd=64)
    model = GPT(cfg)
    model.eval()
    tokens = list(model.generate([261], max_tokens=4, temperature=0.0))
    assert len(tokens) == 4
