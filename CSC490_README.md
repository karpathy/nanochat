# CSC490 Part 2 — Ablation Study Setup & Run Guide

## Environment Setup

```bash
# Install uv if needed
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (including modal as dev dep)
uv sync --dev

# Activate the venv
source .venv/bin/activate

# Build the rustbpe tokenizer (requires Rust)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# If you have conda activated, unset it first:
# unset CONDA_PREFIX && uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

---

## Modal Setup (one-time)

```bash
# Authenticate with Modal (creates ~/.modal.toml)
uv run modal setup

# Create the secret with your API keys
uv run modal secret create nanochat-secrets \
    WANDB_API_KEY=<your_wandb_key> \
    HF_TOKEN=hf_<your_hf_token> \
    WANDB_ENTITY=<your_wandb_username>
    HF_TOKEN=hf_<your_hf_token> \
    WANDB_ENTITY=<your_wandb_username>
```

---

## Part 2 — Ablation Studies (yoyoliuuu)
## Part 2 — Ablation Studies (yoyoliuuu)

### Full pipeline (first time — downloads data, trains tokenizer, runs all 3 ablations)

```bash
# --detach keeps the pipeline alive even if you close your terminal
uv run modal run --detach nanochat_modal.py::main
```

This runs all 5 stages server-side on Modal:
1. Download 12 FineWeb-EDU shards (~20 min, CPU)
2. Train BPE tokenizer (~5 min, A10G)
3. `picochat-baseline` — relu², RoPE 10K (~51 min, A10G)
4. `picochat-swiglu` — SwiGLU, RoPE 10K (~55 min, A10G)
5. `picochat-mtp` — relu², MTP 1-step, w=0.3 (~66 min, A10G)

Monitor progress: https://wandb.ai/yoyoliuuu/nanochat

### Re-run individual ablations (data + tokenizer already on volume)

```bash
uv run modal run nanochat_modal.py::run_baseline
uv run modal run nanochat_modal.py::run_swiglu
uv run modal run nanochat_modal.py::run_mtp
uv run modal run nanochat_modal.py::run_rope500k   # supplemental ablation
```

### Ablation Configurations
### Ablation Configurations

| Run name            | mlp_type | rope_base | num_mtp_steps | Role        |
|---------------------|----------|-----------|---------------|-------------|
| picochat-baseline   | relu2    | 10,000    | 0             | Baseline    |
| picochat-swiglu     | swiglu   | 10,000    | 0             | Ablation A  |
| picochat-mtp        | relu2    | 10,000    | 1             | Ablation B  |
| picochat-rope500k   | relu2    | 500,000   | 0             | Supplemental|

All runs use: `depth=8`, `n_embd=512`, `max_seq_len=512`, `device_batch_size=16`, A10G GPU.

### Cost Reference (A10G @ ~$1.10/hr)
### Cost Reference (A10G @ ~$1.10/hr)

| Stage                   | Duration  | Cost/run | × 3 seeds   |
|-------------------------|-----------|----------|-------------|
| Data + tokenizer        | ~25 min   | ~$0.11   | one-time    |
| picochat-baseline       | ~51 min   | ~$0.94   | ~$2.82      |
| picochat-swiglu         | ~55 min   | ~$1.00   | ~$3.00      |
| picochat-mtp            | ~66 min   | ~$1.21   | ~$3.63      |
| picochat-rope500k       | ~51 min   | ~$0.94   | ~$2.82      |
| Stage                   | Duration  | Cost/run | × 3 seeds   |
|-------------------------|-----------|----------|-------------|
| Data + tokenizer        | ~25 min   | ~$0.11   | one-time    |
| picochat-baseline       | ~51 min   | ~$0.94   | ~$2.82      |
| picochat-swiglu         | ~55 min   | ~$1.00   | ~$3.00      |
| picochat-mtp            | ~66 min   | ~$1.21   | ~$3.63      |
| picochat-rope500k       | ~51 min   | ~$0.94   | ~$2.82      |
| **Total (3 seeds each)**|           |          | **~$12.38** |

---

## Part 3 — Context Window Extension (alvinay73)

```bash
# Run full pipeline
uv run modal run ctx_modal.py

# Or run stages individually
uv run modal run ctx_modal.py::run_ctx_stage1
uv run modal run ctx_modal.py::run_ctx_stage2
uv run modal run ctx_modal.py::run_ctx_evals
```
