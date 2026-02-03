"""
Train a custom BPE tokenizer for IaC-GPT.

This tokenizer is optimized for Infrastructure-as-Code syntax, ensuring that
common IaC tokens like '{{ }}', '${ }', 'resource', 'module', 'apiVersion',
etc. are treated efficiently.

Usage:
    python dev/train_iac_tokenizer.py --data-dir data/iac_raw --vocab-size 50257
"""

import os
import argparse
from pathlib import Path
from typing import List
import json
from collections import Counter


def collect_iac_samples(data_dir: Path, max_samples: int = 100000) -> List[str]:
    """Collect sample texts from IaC files for tokenizer training."""
    samples = []
    
    for iac_type in ["terraform", "kubernetes", "ansible", "crossplane", "docker"]:
        category_dir = data_dir / iac_type
        if not category_dir.exists():
            continue
        
        print(f"Collecting samples from {iac_type}...")
        
        for file_path in category_dir.iterdir():
            if not file_path.is_file():
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if len(content) > 100:
                        samples.append(content)
                
                if len(samples) >= max_samples:
                    break
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        if len(samples) >= max_samples:
            break
    
    print(f"Collected {len(samples)} samples for tokenizer training")
    return samples


def get_iac_special_tokens() -> List[str]:
    """Define IaC-specific tokens that should be treated as single units."""
    special_tokens = [
        # Special control tokens (GPT-2 compatible)
        "
