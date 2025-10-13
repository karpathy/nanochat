"""
Tests for the tokenizer wrapper (high-level API).

Run with:
python -m pytest tests/test_tokenizer.py -v -s --timeout=60
"""

import tempfile
import pytest
from nanochat.tokenizer import RustBPETokenizer


@pytest.fixture
def sample_text():
    """Sample text for training tokenizers."""
    return """
    Hello world! This is a test.
    Machine learning is fascinating.
    Python is a great programming language.
    Tokenization is the first step in NLP.
    """ * 10  # Repeat to have enough data


@pytest.fixture
def trained_tokenizer(sample_text):
    """A small trained tokenizer for testing."""
    vocab_size = 300
    tokenizer = RustBPETokenizer.train_from_iterator([sample_text], vocab_size)
    return tokenizer


def test_tokenizer_train_from_iterator(sample_text):
    """Test training a tokenizer from text."""
    vocab_size = 300
    tokenizer = RustBPETokenizer.train_from_iterator([sample_text], vocab_size)
    
    # Check vocab size
    assert tokenizer.get_vocab_size() == vocab_size


def test_tokenizer_encode_decode(trained_tokenizer):
    """Test encode/decode round trip."""
    text = "Hello world!"
    
    # Encode
    ids = trained_tokenizer.encode(text)
    
    # Check it returns list of ints
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    
    # Decode
    decoded = trained_tokenizer.decode(ids)
    
    # Should match original
    assert decoded == text


def test_tokenizer_encode_empty_string(trained_tokenizer):
    """Test encoding empty string."""
    ids = trained_tokenizer.encode("")
    assert ids == []


def test_tokenizer_decode_empty_list(trained_tokenizer):
    """Test decoding empty list."""
    text = trained_tokenizer.decode([])
    assert text == ""


def test_tokenizer_encode_batch(trained_tokenizer):
    """Test batch encoding."""
    texts = ["Hello", "World", "Test"]
    
    batch_ids = trained_tokenizer.encode(texts)
    
    # Should return list of lists
    assert isinstance(batch_ids, list)
    assert len(batch_ids) == len(texts)
    
    # Each should be a list of ints
    for ids in batch_ids:
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
    
    # Should match individual encoding
    for text, ids in zip(texts, batch_ids):
        assert ids == trained_tokenizer.encode(text)


def test_tokenizer_special_tokens(trained_tokenizer):
    """Test special token encoding."""
    special_token = "<|endoftext|>"
    
    # Should be able to encode special token
    token_id = trained_tokenizer.encode_special(special_token)
    
    assert isinstance(token_id, int)
    assert token_id >= 0


def test_tokenizer_prepend_append(trained_tokenizer):
    """Test prepend and append functionality."""
    text = "Hello world"
    bos_id = trained_tokenizer.encode_special("<|bos|>")
    eos_id = trained_tokenizer.encode_special("<|eos|>")
    
    # Encode with prepend/append
    ids_with_special = trained_tokenizer.encode(
        text, 
        prepend="<|bos|>",
        append="<|eos|>"
    )
    
    # Should have BOS at start and EOS at end
    assert ids_with_special[0] == bos_id
    assert ids_with_special[-1] == eos_id
    
    # Middle should be the text
    ids_without_special = trained_tokenizer.encode(text)
    assert ids_with_special[1:-1] == ids_without_special


def test_tokenizer_save_load(trained_tokenizer):
    """Test saving and loading tokenizer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        trained_tokenizer.save(tmpdir)
        
        # Load
        loaded_tokenizer = RustBPETokenizer.from_directory(tmpdir)
        
        # Should produce same results
        text = "Test tokenization"
        original_ids = trained_tokenizer.encode(text)
        loaded_ids = loaded_tokenizer.encode(text)
        
        assert original_ids == loaded_ids


def test_tokenizer_vocab_size(trained_tokenizer):
    """Test vocab size is correct."""
    vocab_size = trained_tokenizer.get_vocab_size()
    
    assert vocab_size > 0
    assert isinstance(vocab_size, int)


def test_tokenizer_handles_unicode(trained_tokenizer):
    """Test encoding/decoding unicode characters."""
    text = "Hello 世界 🌍"
    
    ids = trained_tokenizer.encode(text)
    decoded = trained_tokenizer.decode(ids)
    
    assert decoded == text


def test_tokenizer_handles_newlines(trained_tokenizer):
    """Test encoding/decoding newlines."""
    text = "Line 1\nLine 2\nLine 3"
    
    ids = trained_tokenizer.encode(text)
    decoded = trained_tokenizer.decode(ids)
    
    assert decoded == text


def test_tokenizer_handles_special_chars(trained_tokenizer):
    """Test encoding/decoding special characters."""
    text = "Special: !@#$%^&*()_+-={}[]|:;<>?,."
    
    ids = trained_tokenizer.encode(text)
    decoded = trained_tokenizer.decode(ids)
    
    assert decoded == text


def test_tokenizer_consistency(trained_tokenizer):
    """Test that encoding same text multiple times gives same result."""
    text = "Consistency test"
    
    ids1 = trained_tokenizer.encode(text)
    ids2 = trained_tokenizer.encode(text)
    ids3 = trained_tokenizer.encode(text)
    
    assert ids1 == ids2 == ids3


def test_tokenizer_different_texts_different_ids(trained_tokenizer):
    """Test that different texts give different token IDs."""
    text1 = "Hello"
    text2 = "World"
    
    ids1 = trained_tokenizer.encode(text1)
    ids2 = trained_tokenizer.encode(text2)
    
    assert ids1 != ids2


def test_tokenizer_bos_token(trained_tokenizer):
    """Test getting BOS token ID."""
    bos_id = trained_tokenizer.get_bos_token_id()
    
    assert isinstance(bos_id, int)
    assert bos_id >= 0


def test_tokenizer_longer_text(trained_tokenizer):
    """Test with longer text."""
    text = "This is a longer piece of text that should be tokenized properly. " * 20
    
    ids = trained_tokenizer.encode(text)
    decoded = trained_tokenizer.decode(ids)
    
    assert decoded == text
    assert len(ids) > 0


def test_tokenizer_encode_decode_various_lengths(trained_tokenizer):
    """Test encode/decode with various text lengths."""
    texts = [
        "a",
        "ab",
        "abc",
        "short",
        "This is a medium length text.",
        "This is a much longer text that contains many words and should test the tokenizer's ability to handle longer sequences without any issues." * 5
    ]
    
    for text in texts:
        ids = trained_tokenizer.encode(text)
        decoded = trained_tokenizer.decode(ids)
        assert decoded == text, f"Failed for text length {len(text)}"

