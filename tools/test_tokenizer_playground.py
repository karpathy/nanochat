#!/usr/bin/env python3
"""
Test script for tokenizer playground - demonstrates functionality without requiring a trained tokenizer.
"""


class MockTokenizer:
    """Mock tokenizer for testing purposes."""

    def __init__(self):
        # Simulate a simple tokenizer with a small vocabulary
        self.vocab = {
            'Hello': 1000,
            ' world': 1001,
            '!': 33,
            'How': 1002,
            ' are': 1003,
            ' you': 1004,
            '?': 63,
        }
        self.special_tokens = {'<|bos|>', '<|user_start|>', '<|user_end|>'}
        self.special_ids = {'<|bos|>': 32000, '<|user_start|>': 32001, '<|user_end|>': 32002}

    def encode(self, text):
        """Simple mock encoding - splits on spaces and punctuation."""
        tokens = []
        current = ""

        for char in text:
            if char in ' !?.,' or not current:
                if current:
                    # Try to find in vocab, otherwise use char ord
                    token_id = self.vocab.get(current, ord(current[0]))
                    tokens.append(token_id)
                    current = ""

                if char in ' !?.,':
                    current = char
                else:
                    current = char
            else:
                current += char

        if current:
            token_id = self.vocab.get(current, ord(current[0]))
            tokens.append(token_id)

        return tokens

    def decode(self, ids):
        """Mock decoding."""
        # Reverse lookup
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        result = ""
        for token_id in ids:
            if token_id in reverse_vocab:
                result += reverse_vocab[token_id]
            elif token_id < 256:
                result += chr(token_id)
            elif token_id in self.special_ids.values():
                for token, tid in self.special_ids.items():
                    if tid == token_id:
                        result += token
                        break
            else:
                result += f"[{token_id}]"
        return result

    def get_special_tokens(self):
        return self.special_tokens

    def encode_special(self, token):
        return self.special_ids.get(token, -1)

    def get_vocab_size(self):
        return 32003


def test_tokenizer_playground():
    """Test the tokenizer playground with mock tokenizer."""
    import sys
    sys.path.insert(0, '/home/user/nanochat')

    from tools.tokenizer_playground import TokenizerPlayground

    # Create mock tokenizer
    tokenizer = MockTokenizer()
    playground = TokenizerPlayground(tokenizer)

    print("=" * 70)
    print("TOKENIZER PLAYGROUND - FUNCTIONALITY TEST")
    print("=" * 70)
    print("\nNote: Using mock tokenizer for demonstration.")
    print("Install and train a real tokenizer for actual use.\n")

    # Test 1: Basic tokenization
    print("\n" + "=" * 70)
    print("TEST 1: Basic Tokenization")
    print("=" * 70)
    playground.tokenize_and_visualize("Hello world!")

    # Test 2: Special tokens
    print("\n" + "=" * 70)
    print("TEST 2: Special Tokens")
    print("=" * 70)
    playground.analyze_special_tokens()

    # Test 3: Vocabulary info
    print("\n" + "=" * 70)
    print("TEST 3: Vocabulary Information")
    print("=" * 70)
    playground.show_vocab_info()

    # Test 4: Comparison
    print("\n" + "=" * 70)
    print("TEST 4: Text Comparison")
    print("=" * 70)
    playground.compare_texts([
        "Hello world!",
        "How are you?",
        "Hi there!",
    ])

    print("\n" + "=" * 70)
    print("✅ All tests completed successfully!")
    print("=" * 70)
    print("\nTo use with a real tokenizer:")
    print("  1. Train a tokenizer: python scripts/tok_train.py")
    print("  2. Run the playground: python tools/tokenizer_playground.py 'Your text here'")
    print("  3. Or use interactive mode: python tools/tokenizer_playground.py -i")
    print()


if __name__ == '__main__':
    test_tokenizer_playground()
