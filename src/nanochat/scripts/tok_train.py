"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""

import argparse
import os
import time

import torch

from nanochat.data.dataset import parquets_iter_batched
from nanochat.data.tokenizer import RustBPETokenizer
from nanochat.paths import tokenizer_dir as get_tokenizer_dir
from nanochat.report import get_report


def build_parser():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer")
    parser.add_argument(
        "--max-chars", type=int, default=2_000_000_000, help="Maximum characters to train on (default: 10B)"
    )
    parser.add_argument("--doc-cap", type=int, default=10_000, help="Maximum characters per document (default: 10,000)")
    parser.add_argument("--vocab-size", type=int, default=32768, help="Vocabulary size (default: 32768 = 2^15)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    print(f"max_chars: {args.max_chars:,}")
    print(f"doc_cap: {args.doc_cap:,}")
    print(f"vocab_size: {args.vocab_size:,}")

    # -------------------------------------------------------------------------
    # Text iterator

    def text_iterator():
        """
        1) Flatten the batches into a single iterator
        2) Crop every document to args.doc_cap characters
        3) Break when we've seen args.max_chars characters
        """
        nchars = 0
        for batch in parquets_iter_batched(split="train"):
            for doc in batch:
                doc_text = doc
                if len(doc_text) > args.doc_cap:
                    doc_text = doc_text[: args.doc_cap]
                nchars += len(doc_text)
                yield doc_text
                if nchars > args.max_chars:
                    return

    text_iter = text_iterator()

    # -------------------------------------------------------------------------
    # Train the tokenizer
    t0 = time.time()
    tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
    t1 = time.time()
    train_time = t1 - t0
    print(f"Training time: {train_time:.2f}s")

    # -------------------------------------------------------------------------
    # Save the tokenizer to disk
    tok_dir = get_tokenizer_dir()
    tokenizer.save(tok_dir)

    # -------------------------------------------------------------------------
    # Quick inline sanity check
    test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_text

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
    get_report().log(
        section="Tokenizer training",
        data=[
            vars(args),
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


if __name__ == "__main__":
    main()
