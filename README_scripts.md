# Custom Debugging Scripts

This directory (and fork) is aimed at adding custom debugging and visualization scripts to `nanochat`.

## Available Scripts

### 1. Token Inspection (`inspect_tokens.py`)
Visualizes the trained vocabulary, showing special tokens and samples of merged tokens.

**How to run:**
From the root directory, run:
```bash
python -m scripts.custom.inspect_tokens
```

**Output:**
- Total vocabulary size.
- List of special tokens and their IDs (e.g., `<|bos|>`, `<|user_start|>`).
- Samples of "merges" (common character combinations).
