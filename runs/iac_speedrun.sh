#!/bin/bash

# IaC-GPT Speedrun: Train a domain-specific LLM for Infrastructure-as-Code
#
# This script trains IaC-GPT on Terraform, Kubernetes, Ansible, Crossplane, and Docker
# Designed for 8xH100 GPUs, target time: < 4 hours
#
# Prerequisites:
#   1. Run data collection: bash dev/fast_scrape_iac.sh
#   2. Run data sharding: python dev/repackage_iac_data.py --input-dir data/iac_raw_cloned --output-dir ~/.cache/nanochat/iac_data
#   3. Symlink data: ln -sf ~/.cache/nanochat/iac_data ~/.cache/nanochat/base_data
#   4. Train tokenizer: python -m scripts.tok_train
#
# Launch:
#   bash runs/iac_speedrun.sh
#   # Or with wandb:
#   WANDB_RUN=iac-gpt bash runs/iac_speedrun.sh
#   # Or in screen:
#   screen -L -Logfile runs/iac_speedrun.log -S iac-gpt bash runs/iac_speedrun.sh

set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# Initialize report
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Verify data and tokenizer exist
echo "=============================================="
echo "IaC-GPT Speedrun"
echo "=============================================="

if [ ! -d "$NANOCHAT_BASE_DIR/base_data" ] && [ ! -L "$NANOCHAT_BASE_DIR/base_data" ]; then
    echo "ERROR: Training data not found at $NANOCHAT_BASE_DIR/base_data"
    echo "Run the data preparation steps first (see script header)"
    exit 1
fi

if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "ERROR: Tokenizer not found at $NANOCHAT_BASE_DIR/tokenizer/"
    echo "Run: python -m scripts.tok_train"
    exit 1
fi

SHARD_COUNT=$(ls -1 $NANOCHAT_BASE_DIR/base_data/*.parquet 2>/dev/null | wc -l)
echo "Found $SHARD_COUNT data shards"
echo "Tokenizer: $NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl"
echo ""

# evaluate the tokenizer
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model pretraining
#
# For IaC-GPT with ~26MB corpus:
# - Using d24 model (same as GPT-2 grade)
# - Reduced data:param ratio since we have less data
# - Multiple epochs over the data to compensate

echo ""
echo "=============================================="
echo "Starting IaC-GPT Pretraining (d24 model)"
echo "=============================================="

# Detect GPU count
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
else
    GPU_COUNT=1
fi

echo "Detected $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -gt 1 ]; then
    # Multi-GPU training with torchrun
    # For small IaC dataset (~26MB), we use more iterations to compensate
    # target-param-data-ratio=5 means we'll loop over data multiple times
    torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.base_train -- \
        --depth=24 \
        --target-param-data-ratio=5 \
        --device-batch-size=16 \
        --run=$WANDB_RUN \
        --model-tag=iac-gpt

    # Evaluate base model
    torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.base_eval -- \
        --device-batch-size=16
else
    # Single GPU or CPU/MPS training
    DEVICE_TYPE=""
    if command -v nvidia-smi &> /dev/null; then
        DEVICE_TYPE="cuda"
    elif [[ "$(uname)" == "Darwin" ]]; then
        DEVICE_TYPE="mps"
    else
        DEVICE_TYPE="cpu"
    fi

    echo "Single device training on: $DEVICE_TYPE"
    echo "Note: For full training, use 8xH100 GPUs"

    # Smaller batch size for single GPU/MPS
    python -m scripts.base_train \
        --depth=12 \
        --device-type=$DEVICE_TYPE \
        --num-iterations=1000 \
        --device-batch-size=4 \
        --run=$WANDB_RUN \
        --model-tag=iac-gpt-small \
        --eval-every=100 \
        --sample-every=200

    python -m scripts.base_eval --device-type=$DEVICE_TYPE
fi

# -----------------------------------------------------------------------------
# SFT: Teach the model the "Senior DevOps Architect" persona
# Uses IaC-specific identity conversations

echo ""
echo "=============================================="
echo "Starting IaC-GPT SFT (DevOps Persona)"
echo "=============================================="

# Generate IaC-specific identity data if not present
IAC_IDENTITY_FILE="$NANOCHAT_BASE_DIR/iac_identity_conversations.jsonl"
if [ ! -f "$IAC_IDENTITY_FILE" ]; then
    echo "Generating IaC-GPT identity conversations..."
    python -c "
import json
import os

# IaC-GPT identity conversations for SFT
conversations = [
    # Identity
    {'messages': [
        {'role': 'user', 'content': 'Who are you?'},
        {'role': 'assistant', 'content': 'I am IaC-GPT, a specialized AI assistant for Infrastructure-as-Code. I can help you write Terraform modules, Kubernetes manifests, Ansible playbooks, and Crossplane compositions. I was trained on high-quality IaC code from production repositories.'}
    ]},
    {'messages': [
        {'role': 'user', 'content': 'What can you help me with?'},
        {'role': 'assistant', 'content': 'I specialize in Infrastructure-as-Code tasks: generating Terraform modules for AWS, GCP, and Azure; writing Kubernetes YAML manifests; creating Ansible playbooks; composing Crossplane resources; and reviewing IaC for security issues like public S3 buckets or overly permissive IAM roles.'}
    ]},
    # Terraform
    {'messages': [
        {'role': 'user', 'content': 'Write a Terraform module for an AWS VPC'},
        {'role': 'assistant', 'content': '''Here is a Terraform module for an AWS VPC with public and private subnets:

\`\`\`hcl
variable \"vpc_cidr\" {
  description = \"CIDR block for VPC\"
  type        = string
  default     = \"10.0.0.0/16\"
}

variable \"environment\" {
  description = \"Environment name\"
  type        = string
}

resource \"aws_vpc\" \"main\" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = \"\${var.environment}-vpc\"
    Environment = var.environment
  }
}

resource \"aws_subnet\" \"public\" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = \"\${var.environment}-public-\${count.index + 1}\"
  }
}

resource \"aws_internet_gateway\" \"main\" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = \"\${var.environment}-igw\"
  }
}

data \"aws_availability_zones\" \"available\" {
  state = \"available\"
}

output \"vpc_id\" {
  value = aws_vpc.main.id
}
\`\`\`

This module creates a VPC with configurable CIDR, two public subnets across availability zones, and an internet gateway.'''}
    ]},
    # Kubernetes
    {'messages': [
        {'role': 'user', 'content': 'Create a Kubernetes deployment for nginx'},
        {'role': 'assistant', 'content': '''Here is a Kubernetes Deployment for nginx with best practices:

\`\`\`yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.25-alpine
        ports:
        - containerPort: 80
        resources:
          requests:
            memory: \"64Mi\"
            cpu: \"100m\"
          limits:
            memory: \"128Mi\"
            cpu: \"200m\"
        livenessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          runAsNonRoot: true
          readOnlyRootFilesystem: true
---
apiVersion: v1
kind: Service
metadata:
  name: nginx
spec:
  selector:
    app: nginx
  ports:
  - port: 80
    targetPort: 80
  type: ClusterIP
\`\`\`

This includes resource limits, health probes, security context, and a Service.'''}
    ]},
    # Security review
    {'messages': [
        {'role': 'user', 'content': 'Review this S3 bucket for security issues: resource \"aws_s3_bucket\" \"data\" { bucket = \"my-data\" acl = \"public-read\" }'},
        {'role': 'assistant', 'content': '''Security issues found:

1. **Public ACL**: The \`acl = \"public-read\"\` makes this bucket publicly accessible. This is a critical security risk.

2. **No encryption**: Server-side encryption is not configured.

3. **No versioning**: Bucket versioning is not enabled for data protection.

Here is a secure version:

\`\`\`hcl
resource \"aws_s3_bucket\" \"data\" {
  bucket = \"my-data\"
}

resource \"aws_s3_bucket_public_access_block\" \"data\" {
  bucket = aws_s3_bucket.data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource \"aws_s3_bucket_server_side_encryption_configuration\" \"data\" {
  bucket = aws_s3_bucket.data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = \"AES256\"
    }
  }
}

resource \"aws_s3_bucket_versioning\" \"data\" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = \"Enabled\"
  }
}
\`\`\`'''}
    ]},
    # Ansible
    {'messages': [
        {'role': 'user', 'content': 'Write an Ansible playbook to install Docker'},
        {'role': 'assistant', 'content': '''Here is an Ansible playbook to install Docker on Ubuntu:

\`\`\`yaml
---
- name: Install Docker
  hosts: all
  become: yes

  tasks:
    - name: Install required packages
      apt:
        name:
          - apt-transport-https
          - ca-certificates
          - curl
          - gnupg
          - lsb-release
        state: present
        update_cache: yes

    - name: Add Docker GPG key
      apt_key:
        url: https://download.docker.com/linux/ubuntu/gpg
        state: present

    - name: Add Docker repository
      apt_repository:
        repo: \"deb [arch=amd64] https://download.docker.com/linux/ubuntu {{ ansible_distribution_release }} stable\"
        state: present

    - name: Install Docker
      apt:
        name:
          - docker-ce
          - docker-ce-cli
          - containerd.io
        state: present
        update_cache: yes

    - name: Start and enable Docker
      systemd:
        name: docker
        state: started
        enabled: yes

    - name: Add user to docker group
      user:
        name: \"{{ ansible_user }}\"
        groups: docker
        append: yes
\`\`\`

This playbook adds the official Docker repository and installs Docker CE.'''}
    ]},
]

# Duplicate and vary for more training data
output_path = '$IAC_IDENTITY_FILE'
with open(output_path, 'w') as f:
    for conv in conversations:
        f.write(json.dumps(conv) + '\n')
    # Add variations
    for i in range(10):
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')

print(f'Generated {len(conversations) * 11} identity conversations')
"
fi

# Run SFT
if [ "$GPU_COUNT" -gt 1 ]; then
    torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.chat_sft -- \
        --device-batch-size=16 \
        --run=$WANDB_RUN

    torchrun --standalone --nproc_per_node=$GPU_COUNT -m scripts.chat_eval -- -i sft
else
    python -m scripts.chat_sft \
        --device-type=$DEVICE_TYPE \
        --device-batch-size=4 \
        --run=$WANDB_RUN

    python -m scripts.chat_eval --device-type=$DEVICE_TYPE -i sft
fi

# -----------------------------------------------------------------------------
# Generate report
python -m nanochat.report generate

echo ""
echo "=============================================="
echo "IaC-GPT Training Complete!"
echo "=============================================="
echo ""
echo "Test your model:"
echo "  python -m scripts.chat_cli -p 'Write a Terraform module for an EKS cluster'"
echo ""
echo "Or start the web UI:"
echo "  python -m scripts.chat_web"
echo ""
echo "Report saved to: report.md"
