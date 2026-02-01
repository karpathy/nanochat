#!/usr/bin/env python3
"""
Preprocess AIME datasets from MathArena.
Downloads and formats the data for evaluation.
"""

import json
import os
from datasets import load_dataset


def preprocess_aime(dataset_name, output_file):
    """
    Download and preprocess an AIME dataset.
    
    Args:
        dataset_name: HuggingFace dataset name (e.g., "MathArena/aime_2024_I")
        output_file: Path to output JSONL file
    """
    print(f"Downloading {dataset_name}...")
    ds = load_dataset(dataset_name, split="train")
    
    with open(output_file, "w") as f:
        for example in ds:
            # Format: problem_idx, problem, answer
            record = {
                "question_id": f"{dataset_name.split('/')[-1]}_{example['problem_idx']}",
                "problem": example["problem"].strip(),
                "answer": str(example["answer"]),  # Convert to string for consistency
            }
            f.write(json.dumps(record) + "\n")
    
    print(f"  Saved {len(ds)} examples to {output_file}")


def merge_jsonl(output_file, *input_files):
    merged = []
    for input_file in input_files:
        with open(input_file, "r") as f:
            for line in f:
                merged.append(json.loads(line))
    with open(output_file, "w") as f:
        for record in merged:
            f.write(json.dumps(record) + "\n")
    print(f"  Merged {len(merged)} examples to {output_file}")


def main():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Process AIME 2024 (I and II)
    aime_2024_i = os.path.join(data_dir, "aime_2024_I.jsonl")
    aime_2024_ii = os.path.join(data_dir, "aime_2024_II.jsonl")
    preprocess_aime("MathArena/aime_2024_I", aime_2024_i)
    preprocess_aime("MathArena/aime_2024_II", aime_2024_ii)
    merge_jsonl(os.path.join(data_dir, "aime_2024.jsonl"), aime_2024_i, aime_2024_ii)
    
    # Process AIME 2025 (appears to be combined I + II)
    preprocess_aime("MathArena/aime_2025", os.path.join(data_dir, "aime_2025.jsonl"))
    
    print("\nAll datasets preprocessed successfully!")
    print("Files saved in dev-aime/data/")


if __name__ == "__main__":
    main()
