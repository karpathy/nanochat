"""BPE tokenizer training and token-bytes cache generation, in the style of GPT-4 tokenizer.."""

import os
import time
from dataclasses import asdict

import torch

from nanochat.common import tokenizer_dir as get_tokenizer_dir
from nanochat.config import Config
from nanochat.dataset import parquets_iter_batched
from nanochat.report import get_report
from nanochat.tokenizer.rust_tokenizer import RustBPETokenizer


def tokenizer_train(config: Config) -> None:
    """Train a BPE tokenizer on the ClimbMix dataset and save it to disk.

    Streams documents from the train split up to ``config.tokenizer.max_chars``
    total characters, crops each document to ``config.tokenizer.doc_cap``, then
    trains a ``RustBPETokenizer`` with ``config.tokenizer.vocab_size`` tokens.
    After training, saves the tokenizer and a ``token_bytes.pt`` cache (bytes
    per token, used for bits-per-byte evaluation) to ``cfg.common.base_dir``.

    Args:
        config: Resolved nanochat config. Uses ``config.common`` and ``config.tokenizer``.
    """
    print(f"max_chars: {config.tokenizer.max_chars:,}")
    print(f"doc_cap: {config.tokenizer.doc_cap:,}")
    print(f"vocab_size: {config.tokenizer.vocab_size:,}")

    # -------------------------------------------------------------------------
    # Text iterator

    def text_iterator():
        """
        1) Flatten the batches into a single iterator
        2) Crop every document to args.doc_cap characters
        3) Break when we've seen args.max_chars characters
        """
        nchars = 0
        for batch in parquets_iter_batched(base_dir=config.common.base_dir, split="train"):
            for doc in batch:
                doc_text = doc
                if len(doc_text) > config.tokenizer.doc_cap:
                    doc_text = doc_text[: config.tokenizer.doc_cap]
                nchars += len(doc_text)
                yield doc_text
                if nchars > config.tokenizer.max_chars:
                    return

    text_iter = text_iterator()

    # -------------------------------------------------------------------------
    # Train the tokenizer
    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(text_iter, config.tokenizer.vocab_size)
    t1 = time.time()
    train_time = t1 - t0
    print(f"Training time: {train_time:.2f}s")

    # -------------------------------------------------------------------------
    # Save the tokenizer to disk
    tok_dir = get_tokenizer_dir(config.common.base_dir)
    tokenizer.save(tok_dir)

    # -------------------------------------------------------------------------
    # Quick inline sanity check
    _sanity_check(tokenizer)

    # -------------------------------------------------------------------------
    # Cache a mapping from token id to number of bytes of that token
    # for efficient evaluation of bits per byte.
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
    token_bytes = []
    for token_id in range(vocab_size):
        token_str = token_strings[token_id]
        if token_str in special_set:
            token_bytes.append(0)
        else:
            id_bytes = len(token_str.encode("utf-8"))
            token_bytes.append(id_bytes)
    token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device="cpu")
    token_bytes_path = os.path.join(tok_dir, "token_bytes.pt")
    with open(token_bytes_path, "wb") as f:
        torch.save(token_bytes, f)
    print(f"Saved token_bytes to {token_bytes_path}")

    # Log to report
    token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
    get_report(config.common.base_dir).log(
        section="Tokenizer training",
        data=[
            asdict(config.tokenizer),
            {"train_time": train_time},
            {"num_special_tokens": len(special_set)},
            {
                "token_bytes_min": int(token_bytes_nonzero.min().item()),
                "token_bytes_max": int(token_bytes_nonzero.max().item()),
                "token_bytes_mean": token_bytes_nonzero.mean().item(),
                "token_bytes_std": token_bytes_nonzero.std().item(),
            },
        ],
    )


def _sanity_check(tokenizer: RustBPETokenizer) -> None:
    test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_text
