
import os
import shutil
import pytest
import tempfile
from unittest.mock import patch, MagicMock
from nanochat.tokenizer import HuggingFaceTokenizer, RustBPETokenizer, SPECIAL_TOKENS

@pytest.fixture
def sample_text():
    return "Hello world! This is a test."

@pytest.fixture
def vocab_size():
    return 300 # minimum is 256 for bytes

def test_huggingface_tokenizer(sample_text, vocab_size):
    # Train
    tok = HuggingFaceTokenizer.train_from_iterator([sample_text], vocab_size)
    assert tok.get_vocab_size() <= vocab_size # It might be smaller if not enough merges
    assert tok.get_vocab_size() >= 256

    # Encode
    ids = tok.encode(sample_text)
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(x, int) for x in ids)

    # Decode
    decoded = tok.decode(ids)
    assert decoded == sample_text

    # Special tokens
    assert tok.get_bos_token_id() is not None
    assert isinstance(tok.get_special_tokens(), list)

def test_rustbpe_tokenizer(sample_text, vocab_size):
    # Train
    # RustBPE requires at least 256 vocab size for bytes
    # Plus special tokens.
    # Logic in RustBPETokenizer.train_from_iterator:
    # vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
    # assert vocab_size_no_special >= 256

    # SPECIAL_TOKENS has 9 tokens.
    # So vocab_size needs to be at least 256 + 9 = 265.

    tok = RustBPETokenizer.train_from_iterator([sample_text], vocab_size)

    # Encode
    ids = tok.encode(sample_text)
    assert isinstance(ids, list)
    assert len(ids) > 0

    # Decode
    decoded = tok.decode(ids)
    assert decoded == sample_text

    # Special tokens
    # RustBPETokenizer uses tiktoken which handles special tokens differently
    # But it should have them registered
    specials = tok.get_special_tokens()
    assert "<|bos|>" in specials
    assert "<|user_start|>" in specials

    # BOS token ID
    bos_id = tok.get_bos_token_id()
    assert isinstance(bos_id, int)

    # Encode with prepend/append
    ids_padded = tok.encode(sample_text, prepend="<|bos|>", append="<|user_end|>")
    assert ids_padded[0] == bos_id
    assert ids_padded[-1] == tok.encode_special("<|user_end|>")
    assert ids_padded[1:-1] == ids

def test_rustbpe_render_conversation():
    # We need a trained tokenizer for this
    # Train a minimal one
    vocab_size = 300
    tok = RustBPETokenizer.train_from_iterator(["hello user assistant python output"], vocab_size)

    conversation = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"}
        ]
    }

    ids, mask = tok.render_conversation(conversation)

    assert len(ids) == len(mask)
    # Validate structure based on logic:
    # <|bos|> (mask 0)
    # <|user_start|> (0) "hello" (0) <|user_end|> (0)
    # <|assistant_start|> (0) "hi there" (1) <|assistant_end|> (1)

    # We can check specific tokens if we knew their IDs, but we can verify the mask pattern
    # It should have some 0s then 0s then 0s then 0s then 1s then 1s.

    # First token is BOS, mask 0
    assert mask[0] == 0

    # Last tokens should be mask 1 (assistant reply + end token)
    assert mask[-1] == 1

def test_rustbpe_render_conversation_system():
    vocab_size = 300
    tok = RustBPETokenizer.train_from_iterator(["hello system user assistant"], vocab_size)

    conversation = {
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ok"}
        ]
    }

    ids, mask = tok.render_conversation(conversation)
    # The system message is merged into the user message
    # "be helpful\n\nhello"

    decoded = tok.decode(ids)
    assert "be helpful" in decoded
    assert "hello" in decoded

def test_tokenizer_save_load():
    vocab_size = 300
    tok = RustBPETokenizer.train_from_iterator(["save load test"], vocab_size)

    with tempfile.TemporaryDirectory() as tmpdir:
        tok.save(tmpdir)

        # Check file exists
        assert os.path.exists(os.path.join(tmpdir, "tokenizer.pkl"))

        # Load
        loaded_tok = RustBPETokenizer.from_directory(tmpdir)
        assert loaded_tok.get_vocab_size() == tok.get_vocab_size()

        # Check encoding match
        text = "save load test"
        assert tok.encode(text) == loaded_tok.encode(text)
