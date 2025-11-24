# RustBPE Tokenizer

This directory contains a high-performance implementation of the Byte Pair Encoding (BPE) algorithm, written in Rust and exposed to Python using PyO3. This tokenizer is used for training the nanochat vocabulary and is designed to be fast, memory-efficient, and parallelized.

## What is BPE?

Byte Pair Encoding is a data compression technique that is widely used in natural language processing for tokenization. It starts with a base vocabulary of individual characters (or bytes) and iteratively merges the most frequent adjacent pairs of tokens into a single new token. This process is repeated for a fixed number of merges, resulting in a vocabulary that can represent common words and subwords as single tokens, while still being able to handle rare words and out-of-vocabulary terms.

## Why Rust?

While the rest of the nanochat codebase is primarily in Python, the BPE training process is computationally intensive and can be a bottleneck. Rust was chosen for this component for several key reasons:

- **Performance:** Rust offers performance comparable to C and C++, which is essential for processing large text corpora quickly.
- **Parallelism:** Rust's ownership model and libraries like Rayon make it easy to write safe and efficient parallel code, allowing the tokenizer to take full advantage of multi-core CPUs.
- **Safety:** Rust's strict compiler and borrow checker prevent common programming errors like null pointer dereferences and data races, leading to more robust and reliable code.
- **Interoperability:** With PyO3, it is straightforward to create Python bindings for Rust code, allowing seamless integration with the rest of the nanochat pipeline.

## Role in nanochat

The `rustbpe` tokenizer is used in the `tok_train.py` script to train a new vocabulary from the pretraining dataset. The trained tokenizer is then used to create a `tiktoken` encoder, which is used for efficient inference in the rest of the nanochat project.
