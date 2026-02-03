"""
Repackage scraped IaC data into shards for IaC-GPT training.

This script takes the raw IaC files scraped by scrape_iac_data.py and converts
them into the parquet shard format expected by nanochat's DataLoader.

The data mixture follows the PRD specification:
- 70% Primary Corpus (Terraform, K8s, Ansible, Crossplane, Docker)
- 20% Instruction Set (we'll prepare this separately from TF-Gen/IaC-Eval)
- 10% Documentation (scraped from HashiCorp, k8s.io, CNCF)

This script handles the 70% primary corpus conversion.

Usage:
    python dev/repackage_iac_data.py --input-dir data/iac_raw --output-dir ~/.cache/nanochat/iac_data
"""

import os
import argparse
from pathlib import Path
import time
import pyarrow.parquet as pq
import pyarrow as pa
from typing import List, Dict
from collections import defaultdict


def load_iac_files(input_dir: Path) -> List[str]:
    """
    Load all IaC files and return them as a list of text documents.
    Each file becomes a single document in the training corpus.
    """
    documents = []
    stats = defaultdict(int)
    
    # Process each IaC category
    for iac_type in ["terraform", "kubernetes", "ansible", "crossplane", "docker"]:
        category_dir = input_dir / iac_type
        if not category_dir.exists():
            print(f"Warning: {category_dir} does not exist, skipping...")
            continue
        
        print(f"\nProcessing {iac_type} files...")
        
        for file_path in category_dir.iterdir():
            if not file_path.is_file():
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Skip empty or very small files
                if len(content) < 50:
                    continue
                
                # Add a header to help the model understand context
                # This is important for the model to learn the structure
                header = f"# {iac_type.upper()} Configuration\n# Source: {file_path.name}\n\n"
                document = header + content
                
                documents.append(document)
                stats[iac_type] += 1
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
    
    # Print statistics
    print("\n" + "="*60)
    print("Data Loading Complete")
    print("="*60)
    total = sum(stats.values())
    for iac_type, count in stats.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{iac_type:15s}: {count:6d} files ({percentage:5.1f}%)")
    print(f"{'TOTAL':15s}: {total:6d} files")
    print("="*60)
    
    return documents


def create_synthetic_examples(num_examples: int = 1000) -> List[str]:
    """
    Create synthetic instruction-style examples to teach the model how to respond.
    This is a mini version of what would come from TF-Gen/IaC-Eval datasets.
    """
    examples = []
    
    # Terraform examples
    tf_templates = [
        # VPC Example
        """# User: Create a Terraform module for an AWS VPC with public and private subnets

# Assistant: Here's a Terraform module for an AWS VPC:

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = var.vpc_name
    Environment = var.environment
  }
}

resource "aws_subnet" "public" {
  count             = length(var.public_subnet_cidrs)
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.public_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]

  map_public_ip_on_launch = true

  tags = {
    Name = "${var.vpc_name}-public-${count.index + 1}"
    Type = "Public"
  }
}

resource "aws_subnet" "private" {
  count             = length(var.private_subnet_cidrs)
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.private_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]

  tags = {
    Name = "${var.vpc_name}-private-${count.index + 1}"
    Type = "Private"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.vpc_name}-igw"
  }
}
""",
        # K8s Deployment
        """# User: Create a Kubernetes deployment for a web application with 3 replicas

# Assistant: Here's a Kubernetes deployment manifest:

apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  labels:
    app: web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      labels:
        app: web-app
    spec:
      containers:
      - name: web-app
        image: nginx:latest
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: "64Mi"
            cpu: "250m"
          limits:
            memory: "128Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 80
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
""",
    ]
    
    # Replicate templates to reach target count
    while len(examples) < num_examples:
        examples.extend(tf_templates)
    
    return examples[:num_examples]


def add_documentation_snippets() -> List[str]:
    """
    Add documentation snippets that explain IaC concepts.
    This teaches the model the "why" behind the code.
    """
    docs = [
        """# Terraform Best Practices

## Resource Naming Conventions
Always use descriptive names for resources that indicate their purpose and environment.
Use naming patterns like: {service}-{environment}-{resource_type}

## State Management
- Never commit terraform.tfstate to version control
- Use remote state backends (S3, Terraform Cloud, etc.)
- Enable state locking to prevent concurrent modifications

## Module Organization
- Keep modules focused and reusable
- Use input variables for configuration
- Provide output values for dependent modules
- Document variables with descriptions
""",
        """# Kubernetes Security Best Practices

## Pod Security
- Never run containers as root (use securityContext)
- Set resource limits to prevent resource exhaustion
- Use read-only root filesystems when possible
- Implement network policies to control traffic

## RBAC (Role-Based Access Control)
- Follow principle of least privilege
- Create specific service accounts for applications
- Use namespaces to isolate workloads
- Regularly audit permissions

## Secrets Management
- Never hardcode secrets in manifests
- Use Kubernetes Secrets or external secret managers
- Rotate credentials regularly
- Encrypt secrets at rest
""",
        """# Ansible Playbook Structure

## Inventory Management
Organize hosts into logical groups:
- Development, staging, production environments
- Functional groups (web servers, database servers)
- Geographic locations

## Role Design
- Create reusable roles for common tasks
- Use role dependencies to manage relationships
- Keep roles focused on a single responsibility
- Version control your roles

## Variables and Precedence
Understanding variable precedence (from lowest to highest):
1. Role defaults
2. Inventory variables
3. Playbook variables
4. Extra vars (-e flag)
""",
    ]
    
    return docs * 100  # Replicate to reach ~10% of total corpus


def shuffle_and_shard(documents: List[str], output_dir: Path):
    """
    Shuffle documents and write them to parquet shards.
    Follows the same format as repackage_data_reference.py
    """
    import random
    
    # Shuffle to mix different IaC types
    random.seed(42)
    random.shuffle(documents)
    
    ndocs = len(documents)
    print(f"\nTotal documents to shard: {ndocs}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sharding parameters (matching nanochat's reference)
    chars_per_shard = 250_000_000  # ~250MB of text per shard
    row_group_size = 1024
    
    shard_docs = []
    shard_index = 0
    shard_characters = 0
    total_docs_processed = 0
    total_time_spent = 0
    t0 = time.time()
    
    for doc in documents:
        shard_docs.append(doc)
        shard_characters += len(doc)
        collected_enough_chars = shard_characters >= chars_per_shard
        docs_multiple_of_row_group_size = len(shard_docs) % row_group_size == 0
        
        if collected_enough_chars and docs_multiple_of_row_group_size:
            shard_path = output_dir / f"shard_{shard_index:05d}.parquet"
            shard_table = pa.Table.from_pydict({"text": shard_docs})
            pq.write_table(
                shard_table,
                str(shard_path),
                row_group_size=row_group_size,
                use_dictionary=False,
                compression="zstd",
                compression_level=3,
                write_statistics=False,
            )
            
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            total_docs_processed += len(shard_docs)
            total_time_spent += dt
            
            remaining_docs = ndocs - total_docs_processed
            avg_time_per_doc = total_time_spent / total_docs_processed if total_docs_processed > 0 else 0
            remaining_time = remaining_docs * avg_time_per_doc
            remaining_time_hours = remaining_time / 3600
            
            print(f"Wrote {shard_path.name}: "
                  f"{len(shard_docs)} docs | "
                  f"{shard_characters:,} chars | "
                  f"{dt:.2f}s | "
                  f"remaining: {remaining_time_hours:.2f}h")
            
            shard_docs = []
            shard_characters = 0
            shard_index += 1
    
    # Write remaining documents as the last shard
    if shard_docs:
        # Pad to row_group_size if needed
        while len(shard_docs) % row_group_size != 0:
            shard_docs.append("")  # Empty padding documents
        
        shard_path = output_dir / f"shard_{shard_index:05d}.parquet"
        shard_table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(
            shard_table,
            str(shard_path),
            row_group_size=row_group_size,
            use_dictionary=False,
            compression="zstd",
            compression_level=3,
            write_statistics=False,
        )
        print(f"Wrote final shard {shard_path.name}: {len(shard_docs)} docs | {shard_characters:,} chars")
    
    print(f"\n{'='*60}")
    print(f"Sharding complete!")
    print(f"Total shards created: {shard_index + 1}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Repackage IaC data into training shards")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/iac_raw",
        help="Input directory with scraped IaC files (default: data/iac_raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.expanduser("~/.cache/nanochat/iac_data"),
        help="Output directory for parquet shards (default: ~/.cache/nanochat/iac_data)",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Include synthetic instruction examples (simulates 20% instruction set)",
    )
    parser.add_argument(
        "--include-docs",
        action="store_true",
        help="Include documentation snippets (simulates 10% documentation)",
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("="*60)
    print("IaC Data Repackaging for IaC-GPT")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load primary corpus (70%)
    documents = load_iac_files(input_dir)
    
    if not documents:
        print("ERROR: No documents found! Please run scrape_iac_data.py first.")
        return
    
    # Add synthetic instruction examples (20%)
    if args.include_synthetic:
        print("\nAdding synthetic instruction examples...")
        synthetic = create_synthetic_examples(num_examples=len(documents) // 3)
        documents.extend(synthetic)
        print(f"Added {len(synthetic)} synthetic examples")
    
    # Add documentation (10%)
    if args.include_docs:
        print("\nAdding documentation snippets...")
        docs = add_documentation_snippets()
        documents.extend(docs)
        print(f"Added {len(docs)} documentation snippets")
    
    # Calculate total size
    total_chars = sum(len(doc) for doc in documents)
    total_mb = total_chars / 1_000_000
    print(f"\nTotal corpus size: {total_chars:,} characters ({total_mb:.1f} MB)")
    
    # Shuffle and create shards
    shuffle_and_shard(documents, output_dir)


if __name__ == "__main__":
    main()
