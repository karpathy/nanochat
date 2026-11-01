# rustbpe

A lightweight Rust library for training GPT-style byte pair encoding
(BPE) tokenizers. rustAgent vendors this crate and uses its native Rust API.

## Rust usage

```rust
let mut tokenizer = rustbpe::Tokenizer::new();
tokenizer.train_from_texts(texts, 32_768, pattern);
let encoded = tokenizer.encode("hello world");
let decoded = tokenizer.decode(encoded)?;
let ranks = tokenizer.get_mergeable_ranks_native();
```

Run the crate tests with `cargo test` from this directory.

## License

MIT
