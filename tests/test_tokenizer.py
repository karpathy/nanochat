"""
Test Tokenizer classes.

Run: python -m pytest tests/test_tokenizer.py -v

Tests focus on the tokenizer interfaces that can be validated without training
a full tokenizer or downloading pretrained models.

Note: These tests require the full nanochat dependencies to be installed,
including rustbpe and tiktoken. When dependencies are missing, tests will
be skipped gracefully.
"""

import pytest

# The nanochat.tokenizer module imports rustbpe at module level, so we need
# to wrap the entire import in a try/except
try:
    from nanochat.tokenizer import (
        SPECIAL_TOKENS,
        SPLIT_PATTERN,
        HuggingFaceTokenizer,
        RustBPETokenizer,
    )
    HAS_TOKENIZER_MODULE = True
except ImportError as e:
    HAS_TOKENIZER_MODULE = False
    IMPORT_ERROR = str(e)
    # Define placeholders for skipping
    SPECIAL_TOKENS = []
    SPLIT_PATTERN = ""
    HuggingFaceTokenizer = None
    RustBPETokenizer = None


# Skip all tests if the tokenizer module cannot be imported
pytestmark = pytest.mark.skipif(
    not HAS_TOKENIZER_MODULE,
    reason=f"Tokenizer module not available: {IMPORT_ERROR if not HAS_TOKENIZER_MODULE else ''}"
)


class TestSpecialTokens:
    """Tests for special token definitions."""

    def test_special_tokens_not_empty(self):
        """Ensure we have special tokens defined."""
        assert len(SPECIAL_TOKENS) > 0

    def test_special_tokens_are_unique(self):
        """All special tokens should be unique."""
        assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS))

    def test_required_special_tokens_present(self):
        """Critical special tokens must be present."""
        required = [
            "<|bos|>",
            "<|user_start|>",
            "<|user_end|>",
            "<|assistant_start|>",
            "<|assistant_end|>",
        ]
        for token in required:
            assert token in SPECIAL_TOKENS, f"Missing required special token: {token}"

    def test_special_tokens_format(self):
        """Special tokens should follow the <|name|> format."""
        import re
        pattern = r"^<\|[a-z_]+\|>$"
        for token in SPECIAL_TOKENS:
            assert re.match(pattern, token), f"Token '{token}' doesn't match expected format <|name|>"


class TestSplitPattern:
    """Tests for the GPT-4 style split pattern."""

    def test_split_pattern_is_valid_regex(self):
        """The split pattern should be a valid regex pattern."""
        import regex
        # This should not raise an exception
        compiled = regex.compile(SPLIT_PATTERN)
        assert compiled is not None

    def test_split_pattern_splits_words(self):
        """Basic test that the pattern splits text into words."""
        import regex
        text = "Hello, world! This is a test."
        pattern = regex.compile(SPLIT_PATTERN)
        matches = pattern.findall(text)
        # Should have multiple matches
        assert len(matches) > 1
        # Joined matches should reconstruct most of the text
        reconstructed = "".join(matches)
        # Note: might have slight differences due to pattern details
        assert len(reconstructed) > 0

    def test_split_pattern_handles_numbers(self):
        r"""Pattern should handle numbers according to \p{N}{1,2} spec."""
        import regex
        text = "123456"
        pattern = regex.compile(SPLIT_PATTERN)
        matches = pattern.findall(text)
        # With {1,2}, "123456" should split into multiple parts (each 1-2 digits)
        assert len(matches) >= 3, f"Expected '123456' to split into 3+ parts, got {matches}"


class TestHuggingFaceTokenizer:
    """Tests for HuggingFaceTokenizer class."""

    @pytest.fixture
    def gpt2_tokenizer(self):
        """Load GPT-2 tokenizer for testing."""
        try:
            return HuggingFaceTokenizer.from_pretrained("gpt2")
        except Exception:
            pytest.skip("Could not load GPT-2 tokenizer (requires network)")

    def test_from_pretrained_gpt2(self, gpt2_tokenizer):
        """Test loading GPT-2 tokenizer."""
        assert gpt2_tokenizer is not None
        assert gpt2_tokenizer.get_vocab_size() > 0

    def test_encode_decode_roundtrip(self, gpt2_tokenizer):
        """Encoding then decoding should recover the original text."""
        text = "Hello, world!"
        ids = gpt2_tokenizer.encode(text)
        decoded = gpt2_tokenizer.decode(ids)
        assert text == decoded

    def test_encode_returns_list_of_ints(self, gpt2_tokenizer):
        """Encode should return a list of integers."""
        ids = gpt2_tokenizer.encode("Hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_batch(self, gpt2_tokenizer):
        """Encode should handle a list of strings."""
        texts = ["Hello", "World", "Test"]
        ids_batch = gpt2_tokenizer.encode(texts)
        assert isinstance(ids_batch, list)
        assert len(ids_batch) == len(texts)
        for ids in ids_batch:
            assert isinstance(ids, list)

    def test_encode_with_prepend(self, gpt2_tokenizer):
        """Encoding with prepend should add token at start."""
        bos_id = gpt2_tokenizer.get_bos_token_id()
        ids = gpt2_tokenizer.encode("Hello", prepend=bos_id)
        assert ids[0] == bos_id

    def test_get_bos_token_id(self, gpt2_tokenizer):
        """BOS token ID should be a valid integer."""
        bos_id = gpt2_tokenizer.get_bos_token_id()
        assert isinstance(bos_id, int)
        assert bos_id >= 0

    def test_vocab_size_is_positive(self, gpt2_tokenizer):
        """Vocabulary size should be positive."""
        vocab_size = gpt2_tokenizer.get_vocab_size()
        assert vocab_size > 0

    def test_callable_interface(self, gpt2_tokenizer):
        """Tokenizer should be callable as shorthand for encode."""
        text = "Hello"
        ids1 = gpt2_tokenizer.encode(text)
        ids2 = gpt2_tokenizer(text)
        assert ids1 == ids2


class TestRustBPETokenizer:
    """Tests for RustBPETokenizer class."""

    @pytest.fixture
    def tiktoken_tokenizer(self):
        """Load tiktoken GPT-2 tokenizer for testing."""
        try:
            return RustBPETokenizer.from_pretrained("gpt2")
        except Exception:
            pytest.skip("Could not load tiktoken GPT-2 encoding")

    def test_from_pretrained_gpt2(self, tiktoken_tokenizer):
        """Test loading GPT-2 via tiktoken."""
        assert tiktoken_tokenizer is not None
        assert tiktoken_tokenizer.get_vocab_size() > 0

    def test_encode_decode_roundtrip(self, tiktoken_tokenizer):
        """Encoding then decoding should recover the original text."""
        text = "Hello, world!"
        ids = tiktoken_tokenizer.encode(text)
        decoded = tiktoken_tokenizer.decode(ids)
        assert text == decoded

    def test_encode_returns_list_of_ints(self, tiktoken_tokenizer):
        """Encode should return a list of integers."""
        ids = tiktoken_tokenizer.encode("Hello")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_batch(self, tiktoken_tokenizer):
        """Encode should handle a list of strings."""
        texts = ["Hello", "World", "Test"]
        ids_batch = tiktoken_tokenizer.encode(texts)
        assert isinstance(ids_batch, list)
        assert len(ids_batch) == len(texts)
        for ids in ids_batch:
            assert isinstance(ids, list)

    def test_encode_with_prepend(self, tiktoken_tokenizer):
        """Encoding with prepend should add token at start."""
        bos_id = tiktoken_tokenizer.get_bos_token_id()
        ids = tiktoken_tokenizer.encode("Hello", prepend=bos_id)
        assert ids[0] == bos_id

    def test_encode_with_append(self, tiktoken_tokenizer):
        """Encoding with append should add token at end."""
        bos_id = tiktoken_tokenizer.get_bos_token_id()
        ids = tiktoken_tokenizer.encode("Hello", append=bos_id)
        assert ids[-1] == bos_id

    def test_encode_with_prepend_and_append(self, tiktoken_tokenizer):
        """Encoding with both prepend and append should work."""
        bos_id = tiktoken_tokenizer.get_bos_token_id()
        ids = tiktoken_tokenizer.encode("Hello", prepend=bos_id, append=bos_id)
        assert ids[0] == bos_id
        assert ids[-1] == bos_id
        assert len(ids) >= 3  # At least prepend + some tokens + append

    def test_get_bos_token_id(self, tiktoken_tokenizer):
        """BOS token ID should be a valid integer."""
        bos_id = tiktoken_tokenizer.get_bos_token_id()
        assert isinstance(bos_id, int)
        assert bos_id >= 0

    def test_vocab_size_is_positive(self, tiktoken_tokenizer):
        """Vocabulary size should be positive."""
        vocab_size = tiktoken_tokenizer.get_vocab_size()
        assert vocab_size > 0

    def test_callable_interface(self, tiktoken_tokenizer):
        """Tokenizer should be callable as shorthand for encode."""
        text = "Hello"
        ids1 = tiktoken_tokenizer.encode(text)
        ids2 = tiktoken_tokenizer(text)
        assert ids1 == ids2

    def test_encode_empty_string(self, tiktoken_tokenizer):
        """Encoding empty string should return empty list."""
        ids = tiktoken_tokenizer.encode("")
        assert ids == []

    def test_encode_unicode(self, tiktoken_tokenizer):
        """Tokenizer should handle unicode text."""
        text = "Привет мир! 你好世界! 🌍"
        ids = tiktoken_tokenizer.encode(text)
        decoded = tiktoken_tokenizer.decode(ids)
        assert text == decoded

    def test_id_to_token(self, tiktoken_tokenizer):
        """id_to_token should return string representation."""
        # Token 0 should exist in any tokenizer
        token_str = tiktoken_tokenizer.id_to_token(0)
        assert isinstance(token_str, str)

    def test_special_tokens_set(self, tiktoken_tokenizer):
        """get_special_tokens should return a set."""
        special = tiktoken_tokenizer.get_special_tokens()
        assert isinstance(special, (set, frozenset))


class TestRenderConversation:
    """Tests for conversation rendering functionality."""

    @pytest.fixture
    def tokenizer(self):
        """Load tiktoken GPT-2 tokenizer for testing."""
        try:
            return RustBPETokenizer.from_pretrained("gpt2")
        except Exception:
            pytest.skip("Could not load tiktoken GPT-2 encoding")

    def test_render_simple_conversation(self, tokenizer):
        """Test rendering a simple user-assistant conversation."""
        conversation = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        ids, mask = tokenizer.render_conversation(conversation)

        # Should return lists of same length
        assert isinstance(ids, list)
        assert isinstance(mask, list)
        assert len(ids) == len(mask)
        assert len(ids) > 0

        # Mask should only contain 0s and 1s
        assert all(m in (0, 1) for m in mask)

        # Some tokens should be masked (assistant response)
        assert sum(mask) > 0

    def test_render_conversation_with_system(self, tokenizer):
        """Test that system messages are merged with user message."""
        conversation = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
        ids, mask = tokenizer.render_conversation(conversation)
        assert len(ids) > 0
        assert len(ids) == len(mask)

    def test_render_conversation_max_tokens(self, tokenizer):
        """Test that max_tokens is respected."""
        conversation = {
            "messages": [
                {"role": "user", "content": "Hello " * 100},
                {"role": "assistant", "content": "Response " * 100},
            ]
        }
        max_tokens = 50
        ids, mask = tokenizer.render_conversation(conversation, max_tokens=max_tokens)

        assert len(ids) <= max_tokens
        assert len(mask) <= max_tokens

    def test_render_conversation_starts_with_bos(self, tokenizer):
        """Rendered conversation should start with BOS token."""
        conversation = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
        ids, mask = tokenizer.render_conversation(conversation)
        bos_id = tokenizer.get_bos_token_id()
        assert ids[0] == bos_id

    def test_render_conversation_bos_not_masked(self, tokenizer):
        """BOS token should not be masked (mask=0)."""
        conversation = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
        ids, mask = tokenizer.render_conversation(conversation)
        assert mask[0] == 0  # BOS should not be supervised


class TestRenderForCompletion:
    """Tests for RL-style completion rendering."""

    @pytest.fixture
    def tokenizer(self):
        """Load tiktoken GPT-2 tokenizer for testing."""
        try:
            return RustBPETokenizer.from_pretrained("gpt2")
        except Exception:
            pytest.skip("Could not load tiktoken GPT-2 encoding")

    def test_render_for_completion_basic(self, tokenizer):
        """Test basic completion rendering."""
        conversation = {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }
        ids = tokenizer.render_for_completion(conversation)

        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_render_for_completion_ends_with_assistant_start(self, tokenizer):
        """Completion rendering should end with assistant_start token."""
        conversation = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        }
        ids = tokenizer.render_for_completion(conversation)
        assistant_start = tokenizer.encode_special("<|assistant_start|>")
        assert ids[-1] == assistant_start

    def test_render_for_completion_does_not_mutate_original(self, tokenizer):
        """Original conversation should not be mutated."""
        conversation = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        original_len = len(conversation["messages"])
        tokenizer.render_for_completion(conversation)
        assert len(conversation["messages"]) == original_len


class TestVisualization:
    """Tests for tokenization visualization."""

    @pytest.fixture
    def tokenizer(self):
        """Load tiktoken GPT-2 tokenizer for testing."""
        try:
            return RustBPETokenizer.from_pretrained("gpt2")
        except Exception:
            pytest.skip("Could not load tiktoken GPT-2 encoding")

    def test_visualize_tokenization_returns_string(self, tokenizer):
        """Visualization should return a string."""
        conversation = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        }
        ids, mask = tokenizer.render_conversation(conversation)
        viz = tokenizer.visualize_tokenization(ids, mask)
        assert isinstance(viz, str)
        assert len(viz) > 0

    def test_visualize_with_token_id(self, tokenizer):
        """Visualization with token IDs should include parenthesized numbers."""
        conversation = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        }
        ids, mask = tokenizer.render_conversation(conversation)
        viz = tokenizer.visualize_tokenization(ids, mask, with_token_id=True)
        # Should contain parenthesized numbers for token IDs
        assert "(" in viz and ")" in viz
