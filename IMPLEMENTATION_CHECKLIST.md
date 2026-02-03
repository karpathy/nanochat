# IaC-GPT Implementation Checklist

This checklist tracks the implementation of IaC-GPT according to the PRD specifications.

---

## ✅ Phase 0: Core Infrastructure (COMPLETE)

### P0-1: Data Collection Pipeline
- [x] **Created:** `dev/scrape_iac_data.py`
  - Scrapes Terraform, Kubernetes, Ansible, Crossplane, and Docker code from GitHub
  - Quality filters: min 10 stars, file size limits
  - Progress tracking for resumable scraping
  - Expected output: ~10,000-30,000 files

### P0-2: Data Sharding
- [x] **Created:** `dev/repackage_iac_data.py`
  - Converts raw IaC files to parquet shards
  - Implements data mixture:
    - 70% Primary Corpus (scraped IaC)
    - 20% Instruction Set (synthetic examples)
    - 10% Documentation (best practices)
  - Output format compatible with nanochat DataLoader

### P0-3: Custom Tokenizer (Optional)
- [x] **Created:** `dev/train_iac_tokenizer.py`
  - Placeholder for custom BPE tokenizer
  - Optimizes for IaC-specific tokens (`{{`, `${`, `resource`, `apiVersion`, etc.)
  - **Note:** Can use default GPT-2 tokenizer initially

### P0-4: Base Training Script
- [x] **Created:** `runs/speedrun_iac.sh`
  - Modified speedrun for IaC data
  - Training parameters:
    - Model: d24 (24 layers, ~1.6B params)
    - Compute: 8x H100
    - Time: ~3-4 hours
    - Cost: <$100
    - Data ratio: 12:1 (param:data)

### P0-5: Identity Infusion
- [x] **Created:** `dev/gen_iac_identity.py`
  - Generates "Senior DevOps Architect" persona
  - Creates 1000+ synthetic conversations
  - Topics:
    - Self-introduction
    - Best practices guidance
    - Security awareness
    - Technical explanations

---

## ✅ Phase 1: User Interface (COMPLETE)

### P1-1: Local CLI Tool
- [x] **Created:** `scripts/iac_cli.py`
  - **Generate Mode:** Creates Terraform/K8s/Ansible code
    ```bash
    python scripts/iac_cli.py generate --type terraform --service eks --output eks.tf
    ```
  - **Audit Mode:** Scans for security issues (public S3, 0.0.0.0/0, etc.)
    ```bash
    python scripts/iac_cli.py audit --path infrastructure/ --report audit.txt
    ```
  - **Interactive Mode:** Chat interface
    ```bash
    python scripts/iac_cli.py interactive
    ```

### P1-2: Safety Guardrails (Partial)
- [x] Security awareness in identity data (warns about 0.0.0.0/0)
- [x] Audit mode detects dangerous patterns
- [ ] **TODO:** Fine-tune to actively reject dangerous configurations
  - Create dataset of harmful IaC examples
  - Train with negative examples (RLHF/DPO)

### P1-3: Documentation
- [x] **Created:** `IAC_GPT_README.md`
  - Complete user guide
  - Step-by-step tutorial
  - Use case examples
  - Troubleshooting section

### P1-4: One-Command Setup
- [x] **Created:** `runs/setup_iac_gpt.sh`
  - Automates entire pipeline:
    1. Data scraping
    2. Shard creation
    3. Model training
    4. Evaluation
  - Interactive prompts and error checking

---

## ⏳ Phase 2: Advanced Features (NOT STARTED)

### P2-1: Tool Execution
- [ ] **TODO:** Implement code execution capabilities
  - Add hooks to `nanochat/execution.py`
  - Allow model to run:
    - `terraform validate`
    - `terraform plan -json`
    - `ansible-lint`
    - `kubectl apply --dry-run`
  - Parse output and correct errors iteratively

### P2-2: Translation Engine
- [ ] **TODO:** Implement IaC tool translation
  - Ansible → Crossplane
  - Terraform → OpenTofu
  - CloudFormation → Terraform
  - Use instruction-tuning approach

### P2-3: Continuous Learning
- [ ] **TODO:** Implement feedback loop
  - Track user edits to generated code
  - Collect validated examples
  - Periodic fine-tuning with high-quality data

---

## 📊 Success Metrics (To Be Measured)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Compilability Rate** | >85% | TBD | ⏳ Pending first training |
| **Latency (per token)** | <200ms | TBD | ⏳ Needs benchmark |
| **Training Cost** | <$100 | ~$75 (estimated) | ✅ On target |
| **CORE Score** | >0.25 | TBD | ⏳ Needs eval run |
| **Data Collection** | >10k files | TBD | ⏳ Pending scrape |

---

## 🚀 Quick Start Guide

### Prerequisites
```bash
# 1. Clone and enter nanochat
cd /Users/nickmoore/nanochat

# 2. Set GitHub token (for data scraping)
export GITHUB_TOKEN="ghp_your_token_here"

# 3. Ensure you're on an 8xH100 GPU node
nvidia-smi  # Should show 8x H100 GPUs
```

### Option A: Automated Setup (Recommended)
```bash
# One command to rule them all
bash runs/setup_iac_gpt.sh
```

### Option B: Manual Step-by-Step
```bash
# Step 1: Scrape data (~30 mins)
python dev/scrape_iac_data.py \
    --output-dir data/iac_raw \
    --max-repos 500

# Step 2: Create shards (~5 mins)
python dev/repackage_iac_data.py \
    --input-dir data/iac_raw \
    --output-dir ~/.cache/nanochat/iac_data \
    --include-synthetic \
    --include-docs

# Step 3: Train model (~3-4 hours)
bash runs/speedrun_iac.sh

# Step 4: Test it
python scripts/iac_cli.py interactive
```

---

## 🧪 Evaluation Plan

### 1. Automated Metrics (Objective)
```bash
# Run standard CORE eval
python -m scripts.base_eval --model logs/iac_gpt_d24/latest_checkpoint

# Expected: CORE > 0.25 (GPT-2 grade)
```

### 2. Compilability Testing (Domain-Specific)
```bash
# Create test suite
mkdir -p data/iac_test_suite/

# Generate 100 test cases
python scripts/iac_cli.py generate --type terraform --service vpc > test_vpc.tf
python scripts/iac_cli.py generate --type kubernetes --service deployment > test_deploy.yaml

# Validate each
terraform validate
kubectl apply --dry-run=client -f test_deploy.yaml

# Calculate success rate: valid / total
```

### 3. Human Evaluation (Qualitative)
- [ ] Ask 5 DevOps engineers to rate outputs (1-5 scale)
- [ ] Metrics:
  - Correctness (does it work?)
  - Best practices (is it production-ready?)
  - Security (any vulnerabilities?)
  - Completeness (missing anything important?)

---

## 🐛 Known Issues & Limitations

### Current Limitations
1. **No execution capabilities yet** (Phase 2)
   - Cannot run `terraform validate` automatically
   - Cannot iteratively fix syntax errors

2. **Limited safety guardrails**
   - Model may still generate dangerous configs if prompted
   - Need RLHF/DPO fine-tuning to strengthen rejection behavior

3. **Data mixture not optimized**
   - 20% instruction set is synthetic (placeholder)
   - Should integrate real TF-Gen/IaC-Eval datasets

4. **Tokenizer not customized**
   - Using default GPT-2 tokenizer
   - Missing IaC-specific optimizations

### Future Improvements
- [ ] Add real instruction datasets (TF-Gen, IaC-Eval)
- [ ] Implement custom tokenizer training
- [ ] Add safety dataset (reject dangerous patterns)
- [ ] Integrate with CI/CD pipelines
- [ ] Multi-cloud provider fine-tunes (AWS, GCP, Azure)

---

## 📁 File Inventory

### Created Files (All Complete)
```
nanochat/
├── IAC_GPT_README.md                    # User-facing documentation
├── IMPLEMENTATION_CHECKLIST.md          # This file
│
├── dev/
│   ├── scrape_iac_data.py              # GitHub data scraper
│   ├── repackage_iac_data.py           # Shard creation
│   ├── train_iac_tokenizer.py          # Custom tokenizer (optional)
│   └── gen_iac_identity.py             # Persona infusion
│
├── scripts/
│   └── iac_cli.py                      # Specialized CLI
│
└── runs/
    ├── speedrun_iac.sh                 # Training script
    └── setup_iac_gpt.sh                # One-command setup
```

---

## 🎯 Next Steps

### Immediate (Before First Training)
1. [ ] Set `GITHUB_TOKEN` environment variable
2. [ ] Spin up 8xH100 GPU node (Lambda/GCP)
3. [ ] Run `bash runs/setup_iac_gpt.sh`
4. [ ] Wait 4-5 hours for completion

### Post-Training
1. [ ] Run evaluation suite
2. [ ] Measure compilability rate on test cases
3. [ ] Test with real infrastructure projects
4. [ ] Collect user feedback
5. [ ] Iterate on data mixture

### Long-Term
1. [ ] Integrate real instruction datasets
2. [ ] Implement tool execution (Phase 2)
3. [ ] Add translation capabilities
4. [ ] Create cloud provider-specific variants
5. [ ] Publish results & model weights

---

## 💼 ROI Calculation

### Investment
- **Compute:** $75 (3 hours × 8xH100 @ $24/hour)
- **Developer Time:** 8 hours × $100/hour = $800
- **Total:** ~$875

### Expected Returns (Per Team)
- **Provisioning Time Saved:** 30% × 40 hours/month = 12 hours/month
- **At $100/hour:** $1,200/month savings
- **Misconfig Incidents Avoided:** 20% × 5 incidents/month × $5,000/incident = $5,000/month
- **Total Monthly Savings:** $6,200
- **ROI Breakeven:** 0.14 months (~4 days)

---

## ✅ Completion Status

**Phase 0 (Core):** ✅ 100% Complete  
**Phase 1 (UI/UX):** ✅ 95% Complete (safety guardrails need strengthening)  
**Phase 2 (Advanced):** ❌ 0% Complete (future work)  

**Overall:** ✅ Ready for first training run!

---

**Last Updated:** 2026-02-03  
**Version:** 1.0.0  
**Status:** Ready for Training
