"""Prompt preparation uses the real local tokenizer without running GPU timing."""

import pytest
from nanochat.tokenizer import RustBPETokenizer
from scripts.infer_bench import build_prompt


@pytest.fixture(scope="module")
def tokenizer():
    return RustBPETokenizer.train_from_iterator(
        ["The history of science is a short local tokenizer training corpus. " * 20],
        270,
    )


@pytest.mark.parametrize("length", [*range(1, 12), 128, 2048])
def test_prompt_has_exact_positive_token_length(tokenizer, length):
    tokens = build_prompt(tokenizer, length)
    assert len(tokens) == length
    assert tokens[0] == tokenizer.get_bos_token_id()


def test_valid_context_clamp_can_leave_a_short_prompt(tokenizer):
    context_length, decode_tokens, requested_prompt_tokens = 128, 120, 2048
    prompt_length = min(requested_prompt_tokens, context_length - decode_tokens)
    tokens = build_prompt(tokenizer, prompt_length)
    assert len(tokens) + decode_tokens == context_length


@pytest.mark.parametrize("length", [0, -1, -10])
def test_nonpositive_input_behavior_is_unchanged(tokenizer, length):
    assert build_prompt(tokenizer, length) == []
