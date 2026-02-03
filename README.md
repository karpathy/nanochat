# IaC-GPT

**Infrastructure-as-Code Specialist LLM**

A domain-specific micro-LLM for DevOps teams, trained on Terraform, Kubernetes, Ansible, Crossplane, and Docker. Built on [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy.

## What is IaC-GPT?

IaC-GPT is a GPT-2 grade (~300M-1.6B param) model trained exclusively on Infrastructure-as-Code. Think of it as a "Senior DevOps Architect" you can run locally - even in air-gapped environments where cloud LLMs are prohibited.

**Target Use Cases:**
- **Boilerplate Generation** - Instant Terraform modules (VPC, EKS, RDS) and Kubernetes manifests
- **Security Auditing** - Detect public S3 buckets, missing tags, over-privileged IAM roles
- **Tool Translation** - Convert Ansible to Crossplane, Terraform to OpenTofu
- **Edge/Air-gapped Operations** - Local CLI assistant where cloud LLMs can't go

**ROI Targets:**
- 30% reduction in infrastructure provisioning lead time
- 20% reduction in cloud misconfiguration incidents
- \>85% compilability rate on generated HCL/YAML

## Quick Start

### 1. Collect IaC Training Data

```bash
# Clone high-quality IaC repositories
bash dev/fast_scrape_iac.sh

# Convert to training shards
uv run python dev/repackage_iac_data.py \
    --input-dir data/iac_raw_cloned \
    --output-dir ~/.cache/nanochat/iac_data \
    --include-synthetic --include-docs

# Link data for training
ln -sf ~/.cache/nanochat/iac_data ~/.cache/nanochat/base_data
```

### 2. Train Custom Tokenizer

```bash
# Duplicate shard for train/val split (if only 1 shard)
cp ~/.cache/nanochat/iac_data/shard_00000.parquet ~/.cache/nanochat/iac_data/shard_00001.parquet

# Train tokenizer optimized for HCL/YAML syntax
uv run python -m scripts.tok_train
```

### 3. Train IaC-GPT

**On 8xH100 (full training, ~3 hours, ~$75):**
```bash
WANDB_RUN=iac-gpt bash runs/iac_speedrun.sh
```

**On Mac/Single GPU (testing only):**
```bash
bash runs/iac_speedrun.sh
```

### 4. Talk to Your Model

```bash
# CLI
uv run python -m scripts.chat_cli -p "Write a Terraform module for an EKS cluster"

# Web UI
uv run python -m scripts.chat_web
```

## Data Pipeline

The IaC training corpus is built from:

| Source | Weight | Description |
|--------|--------|-------------|
| **Primary Corpus** | 70% | Terraform HCL, Kubernetes YAML, Ansible, Crossplane, Dockerfiles |
| **Instruction Set** | 20% | Question → Code patterns for IaC generation |
| **Documentation** | 10% | HashiCorp, Kubernetes.io, CNCF docs |

**Scraped Repositories:**
- terraform-aws-modules/* (VPC, EKS, RDS, IAM, S3)
- terraform-google-modules/* (Network, GKE, SQL)
- Azure/terraform-azurerm-*
- kubernetes/examples, argoproj/argo-cd, istio/istio
- ansible/ansible-examples, geerlingguy/ansible-for-devops
- crossplane/crossplane, upbound/platform-ref-*

## File Structure

```
.
├── dev/
│   ├── fast_scrape_iac.sh          # Clone IaC repositories
│   ├── repackage_iac_data.py       # Convert to training shards
│   └── gen_synthetic_data.py       # Generate identity conversations
├── runs/
│   ├── iac_speedrun.sh             # IaC-GPT training pipeline
│   └── speedrun.sh                 # Original nanochat speedrun
├── scripts/
│   ├── base_train.py               # Pretraining
│   ├── chat_sft.py                 # SFT for DevOps persona
│   ├── chat_cli.py                 # CLI interface
│   └── chat_web.py                 # Web UI
└── nanochat/
    ├── gpt.py                      # Transformer model
    ├── tokenizer.py                # BPE tokenizer
    └── dataloader.py               # Data pipeline
```

## Technical Specs

| Parameter | Value |
|-----------|-------|
| Base Architecture | GPT-2 style Transformer |
| Model Sizes | d6 (tiny), d12 (small), d24 (full) |
| Vocab Size | 32,768 tokens |
| Context Length | 2,048 tokens |
| Optimizer | Muon + AdamW |
| Training Time | ~3 hours on 8xH100 |
| Training Cost | ~$75 USD |

## Credits & Acknowledgements

**IaC-GPT is built on [nanochat](https://github.com/karpathy/nanochat) by [Andrej Karpathy](https://github.com/karpathy).**

nanochat is the simplest experimental harness for training LLMs - minimal, hackable, and covers all stages from tokenization to chat UI. The incredible work on nanochat made it possible to train GPT-2 grade models for ~$75 (down from $50,000 in 2019).

Additional acknowledgements from nanochat:
- [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt) for pretraining optimizations
- [HuggingFace](https://huggingface.co/) for fineweb and smoltalk datasets
- [Lambda](https://lambda.ai/service/gpu-cloud) for compute

## Cite

If you use IaC-GPT, please cite both this project and nanochat:

```bibtex
@misc{iacgpt,
  author = {Nick Moore},
  title = {IaC-GPT: Infrastructure-as-Code Specialist LLM},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/holynakamoto/iacgpt}
}

@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
