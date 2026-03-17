"""Tokenizer I/O utilities: load trained tokenizer and token-bytes cache."""

import os

from nanochat.common import tokenizer_dir as _tokenizer_dir
from nanochat.tokenizer.rust_tokenizer import RustBPETokenizer

def get_tokenizer(base_dir: str) -> RustBPETokenizer:
    """Load the trained RustBPETokenizer from the configured base directory."""
    from nanochat.tokenizer.rust_tokenizer import RustBPETokenizer
    return RustBPETokenizer.from_directory(_tokenizer_dir(base_dir))


def get_token_bytes(base_dir: str, device: str = "cpu"):
    """Load the token-bytes cache tensor written by tok_train.

    Returns an int32 tensor of shape (vocab_size,) where each value is the
    number of UTF-8 bytes in that token (0 for special tokens).
    """
    import torch
    tok_dir = _tokenizer_dir(base_dir)
    token_bytes_path = os.path.join(tok_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), (
        f"Token bytes not found at {token_bytes_path}. Run `nanochat data tokenizer train` first."
    )
    with open(token_bytes_path, "rb") as f:
        return torch.load(f, map_location=device)
