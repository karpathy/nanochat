"""
Calculate and report tokenizer evaluation metrics.

This script loads a trained tokenizer and evaluates it against a specified
dataset split (e.g., 'val'). It calculates key metrics and logs them to the
project's reporting system using the `nanochat.report` module.

The calculated metrics include:
1.  Compression Ratio: The ratio of total bytes to total tokens.
2.  Vocabulary Usage: The percentage of the tokenizer's vocabulary used.
3.  Single-Byte Fallback Rate: The percentage of tokens that are raw single bytes.
4.  Token Frequencies: The full count of each token ID in the dataset.

Usage example:
python scripts/tok_metrics.py --split=val
"""
import os
import argparse
import time
import json
from collections import Counter

from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer
from nanochat.dataset import parquets_iter_batched
from nanochat.report import get_report


# 1. --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Calculate tokenizer evaluation metrics.")
parser.add_argument('--split', type=str, default='val', help="Dataset split to evaluate ('train' or 'val')")
args = parser.parse_args()
print(f"Evaluating tokenizer on '{args.split}' split...")

t0 = time.time()

# 2. --- Initialization ---
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
if not os.path.exists(tokenizer_dir) or not os.listdir(tokenizer_dir):
    print(f"Error: Tokenizer directory not found or is empty at {tokenizer_dir}")
    print("Please train a tokenizer first using scripts/tok_train.py")
    exit(1)

print(f"Loading tokenizer from {tokenizer_dir}...")
tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
report = get_report()

token_counts = Counter()
total_tokens = 0
total_bytes = 0
num_documents = 0

# 3. --- Dataset Iteration and Metric Calculation ---
print("Starting dataset iteration...")
iter_t0 = time.time()
try:
    data_iterator = parquets_iter_batched(split=args.split)
    for batch in data_iterator:
        for doc in batch:
            num_documents += 1
            doc_bytes = doc.encode("utf-8")
            total_bytes += len(doc_bytes)
            ids = tokenizer.encode(doc)
            total_tokens += len(ids)
            token_counts.update(ids)

            if num_documents > 0 and num_documents % 1000 == 0:
                print(f"Processed {num_documents:,} documents...")
except FileNotFoundError:
    print(f"Error: Dataset files not found. Did you download them with nanochat/dataset.py?")
    exit(1)
except Exception as e:
    print(f"An error occurred during dataset iteration: {e}")
    exit(1)


iter_t1 = time.time()
print(f"Finished iteration over {num_documents:,} documents in {iter_t1 - iter_t0:.2f}s.")
print("\nAggregating and reporting metrics...")

# 4. --- Metric Aggregation ---
compression_ratio = total_bytes / total_tokens if total_tokens > 0 else 0
used_vocab_size = len(token_counts)
total_vocab_size = tokenizer.get_vocab_size()
vocab_usage_percent = (used_vocab_size / total_vocab_size) * 100 if total_vocab_size > 0 else 0
single_byte_tokens = sum(count for token_id, count in token_counts.items() if token_id < 256)
fallback_rate_percent = (single_byte_tokens / total_tokens) * 100 if total_tokens > 0 else 0

# 5. --- Reporting ---
metrics_to_log = [
    {"split": args.split},
    {"num_documents": num_documents},
    {"total_bytes": total_bytes},
    {"total_tokens": total_tokens},
    {"compression_ratio": compression_ratio},
    {"vocab_size_used": f"{used_vocab_size}/{total_vocab_size}"},
    {"vocab_usage_percent": vocab_usage_percent},
    {"single_byte_fallback_percent": fallback_rate_percent},
]

report.log("Tokenizer Evaluation", metrics_to_log)
print("Logged metrics to 'Tokenizer Evaluation' section of the project report.")

# Save the raw token counts to a separate JSON file
freq_path = os.path.join(tokenizer_dir, f"token_frequencies_{args.split}.json")
print(f"Saving full token frequency map to {freq_path}...")
freq_to_save = {str(k): v for k, v in token_counts.items()}
with open(freq_path, 'w', encoding='utf-8') as f:
    json.dump(freq_to_save, f)

t1 = time.time()
print("\n--- Tokenizer Evaluation Summary ---")
for metric in metrics_to_log:
    for k, v in metric.items():
        if isinstance(v, float):
            print(f"- {k}: {v:.4f}")
        else:
            print(f"- {k}: {v}")
print("------------------------------------")
print(f"Total script time: {t1 - t0:.2f}s")