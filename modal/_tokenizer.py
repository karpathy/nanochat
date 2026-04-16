"""
Minimal standalone tokenizer for Modal inference.

Loads the pickled tiktoken Encoding from a nanochat tokenizer/ directory and
exposes encode / decode / encode_special methods used by serve.py.
"""

import os
import pickle

import tiktoken


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

# nanochat split pattern (matches nanochat/tokenizer.py)
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class NanochatTokenizer:
    def __init__(self, model_dir: str):
        pkl_path = os.path.join(model_dir, "tokenizer.pkl")
        token_bytes_path = os.path.join(model_dir, "token_bytes.pt")

        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, tiktoken.Encoding):
                self._enc = loaded
                return
            if isinstance(loaded, dict):
                mergeable_ranks = loaded
            elif hasattr(loaded, "_mergeable_ranks"):
                mergeable_ranks = loaded._mergeable_ranks
            else:
                self._enc = loaded
                return
        elif os.path.exists(token_bytes_path):
            import torch
            token_bytes = torch.load(token_bytes_path, weights_only=True)
            mergeable_ranks = {bytes(token_bytes[i].tolist()): i for i in range(len(token_bytes))}
        else:
            raise FileNotFoundError(f"No tokenizer found in {model_dir}")

        # nanochat appends specials at the end of the merge table
        offset = len(mergeable_ranks)
        special_tokens = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        self._enc = tiktoken.Encoding(
            name="nanochat",
            pat_str=SPLIT_PATTERN,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

    def encode(self, text: str) -> list[int]:
        return self._enc.encode_ordinary(text)

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)

    def encode_special(self, token_name: str) -> list[int]:
        return [self._enc.encode_single_token(token_name)]

    def get_vocab_size(self) -> int:
        return self._enc.n_vocab


def get_tokenizer(model_dir: str) -> NanochatTokenizer:
    return NanochatTokenizer(model_dir)
