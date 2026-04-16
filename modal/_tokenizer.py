"""
Minimal standalone tokenizer for Modal inference.
Uses tiktoken for fast encoding/decoding with nanochat's special tokens.
"""

import os
import pickle
import tiktoken


# nanochat special tokens
SPECIAL_TOKENS = {
    "<|bos|>": 0,
    "<|user_start|>": 1,
    "<|user_end|>": 2,
    "<|assistant_start|>": 3,
    "<|assistant_end|>": 4,
    "<|python_start|>": 5,
    "<|python_end|>": 6,
    "<|output_start|>": 7,
    "<|output_end|>": 8,
}

# GPT-4 split pattern
SPLIT_PATTERN = r"(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"


class NanochatTokenizer:
    def __init__(self, model_dir: str):
        token_bytes_path = os.path.join(model_dir, "token_bytes.pt")
        tokenizer_pkl_path = os.path.join(model_dir, "tokenizer.pkl")

        if os.path.exists(tokenizer_pkl_path):
            with open(tokenizer_pkl_path, "rb") as f:
                loaded = pickle.load(f)
            # Handle different pickle formats
            if isinstance(loaded, dict):
                mergeable_ranks = loaded
            elif hasattr(loaded, '_mergeable_ranks'):
                # It's a tiktoken Encoding object
                mergeable_ranks = loaded._mergeable_ranks
            else:
                # Try to use it as a pre-built encoder
                self._enc = loaded
                return
        elif os.path.exists(token_bytes_path):
            import torch
            token_bytes = torch.load(token_bytes_path, weights_only=True)
            mergeable_ranks = {bytes(token_bytes[i].tolist()): i for i in range(len(token_bytes))}
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("karpathy/nanochat-d32", "tokenizer.pkl")
            with open(path, "rb") as f:
                mergeable_ranks = pickle.load(f)

        self._enc = tiktoken.Encoding(
            name="nanochat",
            pat_str=SPLIT_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=SPECIAL_TOKENS,
        )

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, allowed_special=set())

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)

    def encode_special(self, token_name: str) -> list[int]:
        return self._enc.encode(token_name, allowed_special="all")

    def get_vocab_size(self) -> int:
        return self._enc.n_vocab


def get_tokenizer(model_dir: str | None = None) -> NanochatTokenizer:
    if model_dir is None:
        model_dir = "/weights/d20"
    return NanochatTokenizer(model_dir)
