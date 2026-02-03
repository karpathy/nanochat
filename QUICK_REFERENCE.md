# IaC-GPT Quick Reference Card

## 🚀 Quick Start (Copy-Paste Ready)

### Prerequisites
```bash
# On your 8xH100 GPU node
export GITHUB_TOKEN="ghp_your_token_here"
cd /Users/nickmoore/nanochat
```

### One-Command Launch
```bash
bash runs/setup_iac_gpt.sh
```

**Time:** 4-5 hours | **Cost:** ~$75 | **Next:** Coffee ☕

---

## 📝 Manual Steps (If Needed)

### Step 1: Scrape Data (30 mins)
```bash
python dev/scrape_iac_data.py --output-dir data/iac_raw --max-repos 500
# Output: ~10k-30k IaC files
```

### Step 2: Create Shards (5 mins)
```bash
python dev/repackage_iac_data.py \
    --input-dir data/iac_raw \
    --output-dir ~/.cache/nanochat/iac_data \
    --include-synthetic --include-docs
# Output: ~50 parquet shards
```

### Step 3: Train Model (3-4 hours)
```bash
bash runs/speedrun_iac.sh
# Output: logs/iac_gpt_d24_*/latest_checkpoint
```

### Step 4: Use It!
```bash
python scripts/iac_cli.py interactive
```

---

## 🛠️ CLI Commands

### Generate Code
```bash
# Terraform
python scripts/iac_cli.py generate --type terraform --service eks --output eks.tf

# Kubernetes
python scripts/iac_cli.py generate --type kubernetes --service deployment --output app.yaml

# Ansible
python scripts/iac_cli.py generate --type ansible --service deploy_app --output deploy.yml
```

### Audit Infrastructure
```bash
python scripts/iac_cli.py audit --path infrastructure/ --report audit.txt
```

### Interactive Mode
```bash
python scripts/iac_cli.py interactive
```

### Web UI
```bash
python -m scripts.chat_web
# Visit: http://YOUR_GPU_NODE_IP:8000
```

---

## 🧪 Evaluation

### Standard CORE Eval
```bash
python -m scripts.base_eval --model logs/iac_gpt_*/latest_checkpoint
# Target: CORE > 0.25
```

### Compilability Test
```bash
# Generate test files
python scripts/iac_cli.py generate --type terraform --service vpc > test.tf

# Validate
terraform init
terraform validate

# Target: >85% success rate
```

---

## 🐛 Troubleshooting

### Rate Limited (GitHub)
```bash
export GITHUB_TOKEN="ghp_your_token"  # Increases from 60 to 5000 req/hour
```

### OOM / CUDA Error
```bash
# Edit runs/speedrun_iac.sh
# Change: --device-batch-size=16
# To:     --device-batch-size=8
```

### No Data Shards
```bash
# Check if scraping worked
ls -lh data/iac_raw/terraform/  # Should have files

# Re-run repackaging
python dev/repackage_iac_data.py --input-dir data/iac_raw --output-dir ~/.cache/nanochat/iac_data --include-synthetic --include-docs
```

---

## 📊 Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Compilability | >85% | Generate 100 files, run `terraform validate` |
| Latency | <200ms/token | Use web UI, measure response time |
| Training Cost | <$100 | Check GPU provider invoice |
| CORE Score | >0.25 | Run `scripts/base_eval.py` |

---

## 📁 File Locations

```
Key Scripts:
├── dev/scrape_iac_data.py         # Step 1: Data collection
├── dev/repackage_iac_data.py      # Step 2: Shard creation
├── runs/speedrun_iac.sh           # Step 3: Training
└── scripts/iac_cli.py             # Step 4: Usage

Documentation:
├── IAC_GPT_README.md              # User guide
├── IMPLEMENTATION_CHECKLIST.md    # PRD tracking
└── PROJECT_SUMMARY.md             # Complete overview

Data:
├── data/iac_raw/                  # Scraped files
└── ~/.cache/nanochat/iac_data/    # Training shards

Models:
└── logs/iac_gpt_d24_*/            # Trained checkpoints
```

---

## 🎯 Common Use Cases

### 1. Generate AWS EKS Cluster
```bash
python scripts/iac_cli.py generate --type terraform --service eks --output eks_cluster.tf
terraform init
terraform plan
```

### 2. Audit Security
```bash
python scripts/iac_cli.py audit --path infrastructure/ --report security_findings.txt
cat security_findings.txt  # Review issues
```

### 3. Ask Questions
```bash
python scripts/iac_cli.py interactive
> How do I secure my Kubernetes deployments?
> Create a Terraform module for an RDS database
> What's the difference between Deployment and StatefulSet?
```

---

## 💰 Cost Breakdown

| Item | Cost |
|------|------|
| Data scraping | Free (GitHub API) |
| GPU compute (8xH100, 3hrs) | $72 |
| Storage (200GB) | $5 |
| **Total** | **~$77** |

**ROI:** Breaks even in 4 days of use!

---

## 🔗 Quick Links

- **Main README:** `IAC_GPT_README.md`
- **PRD Checklist:** `IMPLEMENTATION_CHECKLIST.md`
- **Full Summary:** `PROJECT_SUMMARY.md`
- **Original nanochat:** `README.md`

---

## ⚡ Speed Reference

| Task | Duration |
|------|----------|
| Data scraping | 20-40 mins |
| Shard creation | 5 mins |
| Model training | 3-4 hours |
| Evaluation | 10 mins |
| **Total** | **~4-5 hours** |

---

## 🎓 Example Session

```bash
# Terminal 1: Setup (run once)
export GITHUB_TOKEN="ghp_xxxxx"
bash runs/setup_iac_gpt.sh
# ... wait 4-5 hours ...

# Terminal 2: Use it
python scripts/iac_cli.py interactive

🧑 You: Create a Terraform module for an S3 bucket with encryption

🤖 IaC-GPT: Here's a secure S3 bucket configuration:

resource "aws_s3_bucket" "main" {
  bucket = var.bucket_name
  
  tags = {
    Environment = var.environment
    ManagedBy   = "Terraform"
  }
}

# Enable server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "main" {
  bucket = aws_s3_bucket.main.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block public access - CRITICAL for security
resource "aws_s3_bucket_public_access_block" "main" {
  bucket = aws_s3_bucket.main.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
```

---

## ✅ Pre-Flight Checklist

Before training:
- [ ] GitHub token set: `echo $GITHUB_TOKEN`
- [ ] GPU available: `nvidia-smi` (8x H100)
- [ ] Disk space: `df -h` (>200GB free)
- [ ] In screen/tmux: `screen -S iac_gpt`

After training:
- [ ] Model exists: `ls logs/iac_gpt_*/latest_checkpoint`
- [ ] CORE > 0.25: `python -m scripts.base_eval`
- [ ] CLI works: `python scripts/iac_cli.py interactive`

---

**Ready? Let's build IaC-GPT! 🚀**

```bash
bash runs/setup_iac_gpt.sh
```
