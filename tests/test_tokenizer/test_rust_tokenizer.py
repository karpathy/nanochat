"""Tests for RustBPETokenizer: encode/decode, special tokens, save/load, and chat rendering."""

import os
import pytest

from nanochat.tokenizer.rust_tokenizer import RustBPETokenizer


@pytest.fixture(scope="module")
def tok():
    """GPT-2 pretrained tokenizer — no training required."""
    return RustBPETokenizer.from_pretrained("gpt2")


# ---------------------------------------------------------------------------
# constants.py
# ---------------------------------------------------------------------------

def test_special_tokens_count():
    from nanochat.tokenizer.constants import SPECIAL_TOKENS
    assert len(SPECIAL_TOKENS) == 9


def test_special_tokens_bos_first():
    from nanochat.tokenizer.constants import SPECIAL_TOKENS
    assert SPECIAL_TOKENS[0] == "<|bos|>"


def test_split_pattern_nonempty():
    from nanochat.tokenizer.constants import SPLIT_PATTERN
    assert isinstance(SPLIT_PATTERN, str) and SPLIT_PATTERN


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def test_vocab_size_positive(tok):
    assert tok.get_vocab_size() > 0


def test_bos_token_id_is_int(tok):
    assert isinstance(tok.get_bos_token_id(), int)


def test_special_tokens_set_nonempty(tok):
    assert len(tok.get_special_tokens()) > 0


def test_encode_special_returns_int(tok):
    bos_id = tok.encode_special("<|endoftext|>")
    assert isinstance(bos_id, int)


def test_id_to_token_roundtrip(tok):
    bos_id = tok.encode_special("<|endoftext|>")
    assert tok.id_to_token(bos_id) == "<|endoftext|>"


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------

def test_encode_str_returns_list_of_ints(tok):
    ids = tok.encode("hello world")
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)


def test_encode_decode_roundtrip(tok):
    text = "Hello, world! 42 tokens."
    assert tok.decode(tok.encode(text)) == text


def test_encode_batch_returns_list_of_lists(tok):
    result = tok.encode(["hello", "world"])
    assert isinstance(result, list) and len(result) == 2
    assert all(isinstance(row, list) for row in result)


def test_encode_batch_roundtrip(tok):
    texts = ["Hello!", "Goodbye."]
    for text, ids in zip(texts, tok.encode(texts)):
        assert tok.decode(ids) == text


def test_encode_prepend_str(tok):
    bos_id = tok.encode_special("<|endoftext|>")
    ids = tok.encode("hi", prepend="<|endoftext|>")
    assert ids[0] == bos_id


def test_encode_prepend_int(tok):
    bos_id = tok.encode_special("<|endoftext|>")
    ids = tok.encode("hi", prepend=bos_id)
    assert ids[0] == bos_id


def test_encode_append_str(tok):
    eos_id = tok.encode_special("<|endoftext|>")
    ids = tok.encode("hi", append="<|endoftext|>")
    assert ids[-1] == eos_id


def test_callable_alias(tok):
    assert tok("hello") == tok.encode("hello")


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip(chat_tok, tmp_path):
    chat_tok.save(str(tmp_path))
    assert os.path.exists(tmp_path / "tokenizer.pkl")
    loaded = RustBPETokenizer.from_directory(str(tmp_path))
    text = "roundtrip test 123"
    assert loaded.decode(loaded.encode(text)) == text


def test_save_and_load_vocab_size(chat_tok, tmp_path):
    chat_tok.save(str(tmp_path))
    loaded = RustBPETokenizer.from_directory(str(tmp_path))
    assert loaded.get_vocab_size() == chat_tok.get_vocab_size()


# ---------------------------------------------------------------------------
# render_conversation
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chat_tok():
    """Minimal tiktoken encoding with nanochat special tokens — no training, no disk."""
    import tiktoken
    from nanochat.tokenizer.constants import SPECIAL_TOKENS, SPLIT_PATTERN
    mergeable_ranks = {bytes([i]): i for i in range(256)}
    special_tokens = {name: 256 + i for i, name in enumerate(SPECIAL_TOKENS)}
    enc = tiktoken.Encoding(
        name="test",
        pat_str=SPLIT_PATTERN,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    return RustBPETokenizer(enc, "<|bos|>")


def _simple_conv(user_text="Hello", assistant_text="Hi there"):
    return {"messages": [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]}


def test_render_conversation_lengths_match(chat_tok):
    ids, mask = chat_tok.render_conversation(_simple_conv())
    assert len(ids) == len(mask)


def test_render_conversation_mask_values(chat_tok):
    ids, mask = chat_tok.render_conversation(_simple_conv())
    assert set(mask).issubset({0, 1})


def test_render_conversation_has_assistant_tokens(chat_tok):
    ids, mask = chat_tok.render_conversation(_simple_conv())
    assert 1 in mask


def test_render_conversation_bos_is_masked_zero(chat_tok):
    ids, mask = chat_tok.render_conversation(_simple_conv())
    assert mask[0] == 0


def test_render_conversation_max_tokens_truncates(chat_tok):
    ids, mask = chat_tok.render_conversation(_simple_conv(), max_tokens=5)
    assert len(ids) == 5
    assert len(mask) == 5


def test_render_conversation_system_merged_into_user(chat_tok):
    conv = {"messages": [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]}
    ids, mask = chat_tok.render_conversation(conv)
    assert len(ids) == len(mask)


def test_render_conversation_bad_role_order_raises(chat_tok):
    conv = {"messages": [
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Hello"},
    ]}
    with pytest.raises(AssertionError):
        chat_tok.render_conversation(conv)


def test_render_for_completion_ends_with_assistant_start(chat_tok):
    conv = {"messages": [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]}
    ids = chat_tok.render_for_completion(conv)
    assert ids[-1] == chat_tok.encode_special("<|assistant_start|>")


def test_render_for_completion_last_message_must_be_assistant(chat_tok):
    conv = {"messages": [
        {"role": "user", "content": "Hello"},
    ]}
    with pytest.raises(AssertionError):
        chat_tok.render_for_completion(conv)


# ---------------------------------------------------------------------------
# _encode_text (eval.py helper)
# ---------------------------------------------------------------------------

def test_encode_text_keys(tok):
    from nanochat.tokenizer.eval import _encode_text
    result = _encode_text(tok, "test", "hello world")
    assert {"bytes", "tokens", "ratio"} == result.keys()


def test_encode_text_ratio(tok):
    from nanochat.tokenizer.eval import _encode_text
    text = "hello"
    result = _encode_text(tok, "test", text)
    assert result["bytes"] == len(text.encode("utf-8"))
    assert result["tokens"] > 0
    assert result["ratio"] == result["bytes"] / result["tokens"]


def test_encode_text_roundtrip_failure_raises(tok):
    from nanochat.tokenizer.eval import _encode_text
    from unittest.mock import MagicMock
    bad_tok = MagicMock()
    bad_tok.encode.return_value = [1, 2, 3]
    bad_tok.decode.return_value = "wrong"
    with pytest.raises(ValueError, match="roundtrip"):
        _encode_text(bad_tok, "label", "original text")
