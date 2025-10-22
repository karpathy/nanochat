# Nanochat Developer Onboarding Guide

> A comprehensive guide for developers getting started with the nanochat codebase

## Table of Contents
- [Introduction](#introduction)
- [Project Philosophy](#project-philosophy)
- [Architecture Overview](#architecture-overview)
- [Development Environment Setup](#development-environment-setup)
- [Codebase Structure](#codebase-structure)
- [Key Concepts](#key-concepts)
- [Training Pipeline Deep Dive](#training-pipeline-deep-dive)
- [Common Development Tasks](#common-development-tasks)
- [Testing](#testing)
- [Debugging Tips](#debugging-tips)
- [Performance Considerations](#performance-considerations)
- [Contributing Guidelines](#contributing-guidelines)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

---

## Introduction

### What is Nanochat?

Nanochat is a full-stack implementation of a ChatGPT-like Large Language Model (LLM) designed to be:
- **Accessible**: Train a complete LLM for ~$100 in ~4 hours
- **Minimal**: ~8,300 lines of clean, readable code
- **Educational**: Designed as the capstone for the LLM101n course
- **Hackable**: No complex abstractions or configuration frameworks
- **End-to-end**: From raw text to web-served chatbot in one codebase

### Key Statistics
- **Default Model**: d32 (32-layer Transformer, ~1.9B parameters)
- **Training Data**: 38B tokens from FineWeb-Edu
- **Training Cost**: ~$100 (speedrun) to ~$1000 (extended)
- **Hardware**: Optimized for 8xH100 GPU node
- **Codebase**: 44 files, ~8,300 lines of code
- **Dependencies**: ~2,000 lines in uv.lock

---

## Project Philosophy

Understanding these principles will help you make design decisions aligned with the project's goals:

### 1. Simplicity Over Flexibility
- **No giant configuration objects**: Hyperparameters are simple variables at the top of scripts
- **No model factories**: Direct class instantiation
- **No if-then-else monsters**: Clean, linear code paths

### 2. Readability Over Abstraction
- Code should be readable top-to-bottom
- Prefer explicit over implicit
- Comments explain "why", not "what"

### 3. Hackability Over Generality
- This is a "strong baseline" not a "framework"
- Designed to be forked and modified
- Specific implementations over generic interfaces

### 4. Performance Within Reason
- Optimize where it matters (training loops, tokenization)
- Don't sacrifice readability for marginal gains
- Use Rust for truly performance-critical code (tokenizer)

### 5. Dependency Minimalism
- Only include essential dependencies
- Prefer standard library when possible
- Each dependency must justify its inclusion

---

## Architecture Overview

### High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        NANOCHAT PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

1. DATA PREPARATION
   ├─ Download FineWeb-Edu dataset (HuggingFace)
   ├─ Train BPE tokenizer (rustbpe)
   └─ Tokenize and cache data shards

2. BASE PRETRAINING (~1.5 hours)
   ├─ Initialize GPT model
   ├─ Train on raw text (language modeling)
   └─ Save checkpoint: base.pt

3. MIDTRAINING (~1.5 hours)
   ├─ Load base.pt
   ├─ Continue training with mixed task data
   └─ Save checkpoint: mid.pt

4. SUPERVISED FINETUNING (~0.5 hours)
   ├─ Load mid.pt
   ├─ Finetune on instruction datasets
   │  └─ MMLU, ARC, GSM8K, SmolTalk
   └─ Save checkpoint: sft.pt

5. REINFORCEMENT LEARNING (Optional)
   ├─ Load sft.pt
   ├─ Train reward model
   ├─ PPO-style RL training
   └─ Save checkpoint: rl.pt

6. INFERENCE & SERVING
   ├─ Load sft.pt (or rl.pt)
   ├─ CLI chat interface
   └─ Web server with REST API
```

### System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                         COMPONENTS                              │
└────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌──────────────────┐     ┌──────────────┐
│   RustBPE       │────▶│   Tokenizer      │────▶│   Dataset    │
│   (Rust)        │     │   (Python)       │     │   Loader     │
└─────────────────┘     └──────────────────┘     └──────────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        GPT MODEL                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐       │
│  │  Embedding   │──▶│ Transformer  │──▶│  LM Head     │       │
│  │  Layer       │   │  Blocks (32) │   │  (Unembedding)│      │
│  └──────────────┘   └──────────────┘   └──────────────┘       │
│                           │                                     │
│                     ┌─────┴─────┐                              │
│                     │           │                              │
│            ┌────────▼────┐  ┌───▼────────┐                    │
│            │  Attention  │  │    MLP     │                    │
│            │  (MQA)      │  │  (ReLU²)   │                    │
│            └─────────────┘  └────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        OPTIMIZERS                                │
│  ┌─────────────────────┐        ┌──────────────────────┐       │
│  │  Muon               │        │  DistAdamW           │       │
│  │  (Transformer       │        │  (Embeddings,        │       │
│  │   blocks)           │        │   LM Head)           │       │
│  └─────────────────────┘        └──────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION & SERVING                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Benchmarks  │  │  CLI Chat    │  │  Web Server  │         │
│  │  (MMLU, ARC) │  │  Interface   │  │  (FastAPI)   │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Development Environment Setup

### Prerequisites
- **Python**: 3.10 or higher
- **Rust**: Latest stable (for rustbpe tokenizer)
- **CUDA**: 12.8 (for GPU training)
- **uv**: Fast Python package manager

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync

# Build Rust tokenizer
cd rustbpe
maturin develop --release
cd ..

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import rustbpe; print('RustBPE imported successfully')"
```

### Development on Different Platforms

#### GPU Development (Linux with CUDA)
```bash
# Default configuration works out of the box
uv sync
```

#### CPU Development (Any platform)
```bash
# Use the CPU-friendly run script
bash dev/runcpu.sh  # Reduced model sizes and iterations
```

#### Mac with Apple Silicon (MPS)
```bash
# Automatic MPS detection
# Reduce batch sizes in scripts if you encounter memory issues
```

#### Single GPU Development
```bash
# Simply omit torchrun command
# Example: Instead of
torchrun --standalone --nproc_per_node=8 -m scripts.base_train

# Run:
python -m scripts.base_train
# Code automatically uses gradient accumulation
```

### Recommended Tools
- **IDE**: VSCode with Python extension
- **Debugging**: pdb, PyTorch debugger
- **Profiling**: PyTorch Profiler, nvidia-smi
- **Monitoring**: wandb (Weights & Biases)
- **Version Control**: Git

---

## Codebase Structure

### Directory Layout

```
nanochat/
├── nanochat/                 # Core library modules
│   ├── __init__.py          # Package initialization
│   ├── gpt.py               # GPT model architecture (307 lines)
│   ├── engine.py            # Inference engine with KV cache (343 lines)
│   ├── tokenizer.py         # Tokenization utilities (395 lines)
│   ├── dataset.py           # Data loading from HuggingFace (128 lines)
│   ├── muon.py              # Muon optimizer (187 lines)
│   ├── adamw.py             # Distributed AdamW (77 lines)
│   ├── checkpoint_manager.py # Checkpoint save/load (186 lines)
│   ├── execution.py         # Safe Python code execution (351 lines)
│   ├── common.py            # Utilities (DDP, logging, etc.) (186 lines)
│   ├── core_eval.py         # CORE metric evaluation (311 lines)
│   ├── configurator.py      # CLI config override (104 lines)
│   └── report.py            # Training report generation (408 lines)
│
├── scripts/                  # Training & inference pipelines
│   ├── tok_train.py         # Train BPE tokenizer (106 lines)
│   ├── tok_eval.py          # Evaluate tokenizer (265 lines)
│   ├── base_train.py        # Base pretraining (350 lines)
│   ├── base_loss.py         # Evaluate base loss (79 lines)
│   ├── base_eval.py         # Evaluate base on CORE (186 lines)
│   ├── mid_train.py         # Midtraining (307 lines)
│   ├── chat_sft.py          # Supervised finetuning (282 lines)
│   ├── chat_rl.py           # Reinforcement learning (331 lines)
│   ├── chat_eval.py         # Chat evaluation (254 lines)
│   ├── chat_cli.py          # CLI chat interface (105 lines)
│   └── chat_web.py          # Web server & UI (415 lines)
│
├── tasks/                    # Evaluation benchmarks
│   ├── common.py            # Task framework (377 lines)
│   ├── mmlu.py              # MMLU benchmark (170 lines)
│   ├── arc.py               # ARC benchmark (144 lines)
│   ├── gsm8k.py             # GSM8K math problems (273 lines)
│   ├── humaneval.py         # Code generation (231 lines)
│   ├── smoltalk.py          # Conversational data (92 lines)
│   └── customjson.py        # Custom JSON tasks (59 lines)
│
├── rustbpe/                  # Rust BPE tokenizer
│   ├── src/
│   │   └── lib.rs           # Main Rust implementation (559 lines)
│   ├── Cargo.toml           # Rust dependencies
│   └── pyproject.toml       # Maturin build config
│
├── dev/                      # Development utilities
│   ├── nanochat.png         # Logo
│   ├── runcpu.sh            # CPU-friendly training script
│   └── web/                 # Web UI assets
│       ├── index.html       # Chat UI (187 lines)
│       └── style.css        # Styles (192 lines)
│
├── tests/                    # Test suite
│   └── test_rustbpe.py      # Tokenizer tests
│
├── speedrun.sh              # 4-hour $100 training run
├── run1000.sh               # 33-hour $1000 training run
├── pyproject.toml           # Python project config
├── uv.lock                  # Dependency lockfile
└── README.md                # Main documentation
```

### File Size & Complexity Guide

| File | Lines | Complexity | Purpose |
|------|-------|------------|---------|
| `nanochat/gpt.py` | 307 | Medium | Core model architecture |
| `nanochat/engine.py` | 343 | Medium | Inference with KV cache |
| `nanochat/tokenizer.py` | 395 | Medium | Tokenization abstractions |
| `nanochat/execution.py` | 351 | High | Sandboxed code execution |
| `nanochat/report.py` | 408 | Low | Report generation |
| `scripts/base_train.py` | 350 | High | Pretraining loop |
| `scripts/chat_web.py` | 415 | Medium | Web server & UI |
| `tasks/common.py` | 377 | Medium | Task framework |
| `rustbpe/src/lib.rs` | 559 | High | BPE tokenizer (Rust) |

---

## Key Concepts

### 1. Model Architecture (GPT)

#### GPTConfig
```python
@dataclass
class GPTConfig:
    vocab_size: int = 50304          # Vocabulary size
    n_layer: int = None              # Calculated from depth
    n_head: int = None               # Calculated from n_embd
    n_embd: int = None               # Calculated from depth * 64
    sequence_len: int = 2048         # Max sequence length
    depth: int = 32                  # Depth multiplier (d32 = 32 layers)
    rope_theta: float = 50000.0      # RoPE base frequency
    use_mqa: bool = True             # Multi-Query Attention
```

#### Key Architectural Choices

**Rotary Position Embeddings (RoPE)**
- Replaces absolute positional embeddings
- Better length extrapolation
- Encodes relative position information in attention

**Multi-Query Attention (MQA)**
- Single key/value head, multiple query heads
- Dramatically faster inference (smaller KV cache)
- Minimal quality degradation vs. Multi-Head Attention

**ReLU² Activation**
```python
def forward(self, x):
    x = self.fc(x)
    x = torch.relu(x)
    x = x * x  # ReLU squared
    x = self.proj(x)
    return x
```

**QK Normalization**
- Normalizes query and key vectors before attention
- Improves training stability
- Better gradient flow

**RMSNorm (Root Mean Square Normalization)**
```python
def forward(self, x):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-5)
    return x / rms
```

**Logit Softcapping**
```python
logits = 15.0 * torch.tanh(logits / 15.0)  # Cap at ±15
```
- Prevents extreme logit values
- Stabilizes training
- Improves numerical stability

### 2. Tokenization

#### Special Tokens
```python
SPECIAL_TOKENS = {
    '<|bos|>': 50257,              # Beginning of sequence
    '<|user_start|>': 50258,       # User message start
    '<|user_end|>': 50259,         # User message end
    '<|assistant_start|>': 50260,  # Assistant message start
    '<|assistant_end|>': 50261,    # Assistant message end
    '<|python_start|>': 50262,     # Python code block start
    '<|python_end|>': 50263,       # Python code block end
    '<|output_start|>': 50264,     # Tool output start
    '<|output_end|>': 50265,       # Tool output end
}
```

#### Conversation Format
```
<|bos|><|user_start|>What is 2+2?<|user_end|><|assistant_start|>Let me calculate that.
<|python_start|>
print(2 + 2)
<|python_end|>
<|output_start|>
4
<|output_end|>
The answer is 4.<|assistant_end|>
```

#### RustBPE vs tiktoken vs HuggingFace
- **RustBPE**: Training BPE tokenizers (fast, minimal)
- **tiktoken**: Inference (OpenAI's tokenizer, extremely fast)
- **HuggingFace**: Fallback for flexibility

### 3. Optimizers

#### Muon Optimizer
Used for transformer block parameters (Q, K, V, MLP weights):

```python
# Key features:
# 1. Momentum-based update
# 2. Orthogonalization via Newton-Schulz iteration
# 3. Runs in bfloat16 on GPU
# 4. Separate learning rates for 2D (weights) and 1D (norms) params

# Typical usage:
muon = Muon(
    model.parameters(),
    lr=0.01,              # Learning rate for 2D params
    momentum=0.95,
    norm_lr_scale=0.1,    # Scale for 1D params (0.001 effective LR)
)
```

**Why Muon?**
- Faster convergence than AdamW for transformers
- Better generalization
- Lower memory footprint (no second moment)

#### DistAdamW Optimizer
Used for embedding and output layers:

```python
# Key features:
# 1. Distributed AdamW with ZeRO-2 style sharding
# 2. Optimizer states sharded across GPUs
# 3. Gradient reduction before update

# Typical usage:
adamw = DistAdamW(
    [model.transformer.wte, model.lm_head],
    lr=0.004,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.0,  # Usually no decay for embeddings
)
```

**Why Different Optimizers?**
- Embeddings need different learning dynamics
- Muon requires matrix parameters (2D)
- AdamW better for embeddings (1D-like behavior)

### 4. Data Loading

#### FineWeb-Edu Dataset
```python
# Dataset info:
# - 100B tokens of educational web text
# - Filtered for quality and educational value
# - Stored as parquet files on HuggingFace
# - Each shard: ~250M characters

# Download shards:
python -m nanochat.dataset -n 8  # Download 8 shards (~2B chars)
```

#### Data Pipeline
```
HuggingFace → Parquet Files → Tokenization → .npy Cache → DataLoader
```

#### Distributed Data Loading
```python
# Each GPU gets a unique slice of data
# No overlap, deterministic sampling
dataset = ShardedDataset(
    data_dir='data',
    sequence_len=2048,
    rank=rank,           # GPU rank
    world_size=world_size,  # Total GPUs
)
```

### 5. Distributed Training

#### DDP (Distributed Data Parallel)
```python
# Initialize distributed training
from nanochat.common import setup_ddp

rank, world_size, device = setup_ddp()

# Wrap model
model = DDP(model, device_ids=[rank])

# Each GPU processes different batch
# Gradients averaged across GPUs automatically
```

#### Gradient Accumulation
```python
# Effective batch size = device_batch_size * grad_accum_steps * world_size
# Example: 32 * 4 * 8 = 1024 samples per update

for micro_step in range(grad_accum_steps):
    # Forward pass
    loss = model(batch) / grad_accum_steps  # Scale loss

    # Backward pass
    loss.backward()  # Accumulate gradients

# Update (once per macro step)
optimizer.step()
optimizer.zero_grad()
```

### 6. KV Cache (Inference Optimization)

#### What is KV Cache?
During autoregressive generation, we recompute the same key/value projections repeatedly. KV cache stores them:

```python
# Without cache: O(n²) for n tokens
for i in range(n):
    # Recompute all previous K, V projections
    k = project_key(tokens[:i+1])
    v = project_value(tokens[:i+1])
    output[i] = attention(q[i], k, v)

# With cache: O(n)
cache_k, cache_v = [], []
for i in range(n):
    k_new = project_key(tokens[i])
    v_new = project_value(tokens[i])
    cache_k.append(k_new)
    cache_v.append(v_new)
    output[i] = attention(q[i], cache_k, cache_v)
```

#### KVCache Implementation
```python
class KVCache:
    def __init__(self, max_length, n_layer, n_head, head_dim, device):
        self.cache_k = torch.zeros(n_layer, max_length, n_head, head_dim)
        self.cache_v = torch.zeros(n_layer, max_length, n_head, head_dim)
        self.pos = 0  # Current position

    def update(self, layer_idx, k, v):
        seq_len = k.size(1)
        self.cache_k[layer_idx, self.pos:self.pos+seq_len] = k
        self.cache_v[layer_idx, self.pos:self.pos+seq_len] = v
        return self.cache_k[layer_idx, :self.pos+seq_len], \
               self.cache_v[layer_idx, :self.pos+seq_len]
```

### 7. Checkpointing

#### Checkpoint Format
```python
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'config': config.__dict__,
    'step': current_step,
    'epoch': current_epoch,
    'rng_state': torch.get_rng_state(),
    # ... metadata
}
```

#### Loading Checkpoints
```python
from nanochat.checkpoint_manager import load_checkpoint

model, optimizer, metadata = load_checkpoint(
    'checkpoints/base.pt',
    model,
    optimizer,
    device
)
start_step = metadata.get('step', 0)
```

---

## Training Pipeline Deep Dive

### Phase 1: Tokenizer Training

**Script**: `scripts/tok_train.py`

```bash
# Train BPE tokenizer on sample of FineWeb-Edu
python -m scripts.tok_train --vocab_size=50304 --sample_chars=100000000

# Output: tokenizer/tok50304.model (rustbpe format)
```

**What happens:**
1. Download sample text from FineWeb-Edu
2. Train BPE merges using rustbpe
3. Add special tokens for chat
4. Export to tiktoken-compatible format
5. Save vocabulary and merges

**Key parameters:**
- `vocab_size`: Target vocabulary size (default: 50304)
- `sample_chars`: Number of characters to sample for training

### Phase 2: Base Pretraining

**Script**: `scripts/base_train.py`

```bash
# Single GPU
python -m scripts.base_train --depth=32 --device_batch_size=32

# Multi-GPU (8xGPU)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=32
```

**What happens:**
1. Initialize GPT model from scratch
2. Load tokenized FineWeb-Edu data shards
3. Train on language modeling objective (predict next token)
4. Periodic evaluation on validation set
5. Save checkpoints every N steps
6. Log to wandb (if enabled)

**Key hyperparameters:**
```python
# Model
depth = 32                    # Model depth (d32 = 32 layers)
sequence_len = 2048          # Sequence length

# Training
num_iterations = 6000        # Total training steps
device_batch_size = 32       # Per-GPU batch size
grad_accum_steps = 4         # Gradient accumulation steps
# Effective batch = 32 * 4 * 8 = 1024

# Optimization
lr_muon = 0.01              # LR for Muon (transformer blocks)
lr_adamw = 0.004            # LR for AdamW (embeddings, LM head)
warmup_iters = 0            # Warmup steps
weight_decay = 0.0          # Weight decay

# Evaluation
val_interval = 100          # Evaluate every N steps
save_interval = 1000        # Save checkpoint every N steps
```

**Training dynamics:**
- Loss typically starts around 10-11 (random init)
- Drops to ~3-4 after pretraining
- Validation loss should track training loss closely

### Phase 3: Midtraining

**Script**: `scripts/mid_train.py`

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
```

**What happens:**
1. Load base pretrained checkpoint
2. Mix general text with instruction-following data
3. Continue pretraining with task mixture
4. Prepare model for supervised finetuning

**Data mixture:**
```python
# Example mixture (simplified)
mixture = TaskMixture({
    'fineweb': 0.7,          # 70% general text
    'mmlu': 0.1,             # 10% MMLU
    'arc': 0.05,             # 5% ARC
    'gsm8k': 0.1,            # 10% math
    'smoltalk': 0.05,        # 5% conversations
})
```

**Why midtraining?**
- Smooth transition from base → SFT
- Model sees task formats before finetuning
- Better final performance than direct SFT

### Phase 4: Supervised Finetuning (SFT)

**Script**: `scripts/chat_sft.py`

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
```

**What happens:**
1. Load midtrained checkpoint
2. Finetune on instruction-following datasets
3. Train to predict only assistant responses (mask user prompts)
4. Evaluate on multiple benchmarks
5. Save SFT checkpoint

**Task mixture:**
```python
tasks = TaskMixture([
    ('mmlu', 0.3),           # Multi-task language understanding
    ('arc_easy', 0.15),      # Science questions (easy)
    ('arc_challenge', 0.15), # Science questions (hard)
    ('gsm8k', 0.2),          # Grade school math
    ('smoltalk', 0.2),       # Conversational data
])
```

**Loss masking:**
```python
# Only compute loss on assistant tokens
# User tokens are masked (loss = 0)
loss_mask = (labels != IGNORE_INDEX)
loss = F.cross_entropy(logits, labels, reduction='none')
loss = (loss * loss_mask).sum() / loss_mask.sum()
```

**Key differences from base training:**
- Lower learning rate (10x smaller)
- Shorter training (fewer iterations)
- Task-specific evaluation
- Chat formatting with special tokens

### Phase 5: Reinforcement Learning (Optional)

**Script**: `scripts/chat_rl.py`

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
```

**What happens:**
1. Load SFT checkpoint
2. Generate completions for prompts
3. Score with reward model (or heuristic)
4. Update policy using PPO-style RL
5. Periodic evaluation

**Why RL?**
- Align to human preferences
- Reduce hallucinations
- Improve helpfulness, harmlessness
- Better instruction following

**Challenges:**
- Requires reward model or human feedback
- Can destabilize training
- Needs careful hyperparameter tuning

---

## Common Development Tasks

### Task 1: Train a Small Model Locally

```bash
# 1. Reduce model size and iterations
python -m scripts.tok_train --vocab_size=10000

# 2. Train tiny base model (single GPU, CPU-friendly)
python -m scripts.base_train \
    --depth=8 \
    --device_batch_size=4 \
    --num_iterations=100 \
    --sequence_len=512

# 3. Skip mid/SFT, go straight to inference
python -m scripts.chat_cli
```

### Task 2: Add a New Evaluation Task

```python
# File: tasks/mytask.py

from tasks.common import Task

class MyTask(Task):
    def __init__(self):
        # Load your dataset
        self.data = load_my_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return a dict with 'prompt', 'label', and metadata."""
        item = self.data[idx]
        return {
            'prompt': item['question'],
            'label': item['answer'],
            'metadata': {'category': item['category']}
        }

    def evaluate(self, model, tokenizer, device):
        """Run evaluation and return metrics dict."""
        correct = 0
        total = 0

        for item in self:
            # Generate prediction
            pred = generate_prediction(model, item['prompt'])

            # Check correctness
            if pred == item['label']:
                correct += 1
            total += 1

        return {'accuracy': correct / total}
```

**Register task:**
```python
# In scripts/chat_eval.py or similar
from tasks.mytask import MyTask

# Add to evaluation suite
tasks = {
    'mmlu': MMLU(),
    'mytask': MyTask(),  # Add here
}
```

### Task 3: Modify Model Architecture

**Example: Add Dropout**

```python
# File: nanochat/gpt.py

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln_1 = RMSNorm(config.n_embd)
        self.ln_2 = RMSNorm(config.n_embd)

        # ADD: Dropout layers
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, kv_cache=None):
        # Attention
        attn_out, kv_cache = self.attn(self.ln_1(x), kv_cache)
        x = x + self.dropout(attn_out)  # ADD: dropout

        # MLP
        x = x + self.dropout(self.mlp(self.ln_2(x)))  # ADD: dropout

        return x, kv_cache

# Update GPTConfig
@dataclass
class GPTConfig:
    # ... existing fields
    dropout: float = 0.1  # ADD: dropout probability
```

**Remember to:**
1. Update all scripts that instantiate GPT
2. Add CLI argument for dropout
3. Test that model still trains
4. Document the change

### Task 4: Add Custom Dataset

```python
# File: nanochat/custom_dataset.py

import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, sequence_len):
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len

        # Load your data
        with open(data_path, 'r') as f:
            self.texts = f.readlines()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        tokens = self.tokenizer.encode(text)

        # Truncate/pad to sequence_len
        if len(tokens) > self.sequence_len:
            tokens = tokens[:self.sequence_len]
        else:
            tokens = tokens + [0] * (self.sequence_len - len(tokens))

        # Convert to tensor
        return torch.tensor(tokens, dtype=torch.long)
```

**Use in training:**
```python
# In scripts/base_train.py

from nanochat.custom_dataset import CustomDataset

# Replace existing dataset
train_dataset = CustomDataset(
    data_path='data/my_data.txt',
    tokenizer=tokenizer,
    sequence_len=sequence_len,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=device_batch_size,
    shuffle=True,
)
```

### Task 5: Debug Training Issues

**Common issues and solutions:**

#### Loss is NaN
```python
# Check for:
# 1. Learning rate too high
lr = 0.01  # Try reducing to 0.001

# 2. Gradient explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. Invalid data
assert not torch.isnan(batch).any(), "NaN in input data"
assert not torch.isinf(batch).any(), "Inf in input data"
```

#### Loss not decreasing
```python
# Check for:
# 1. Learning rate too low
print(f"LR: {optimizer.param_groups[0]['lr']}")

# 2. Data loading issue
batch = next(iter(train_loader))
print(f"Batch shape: {batch.shape}, min: {batch.min()}, max: {batch.max()}")

# 3. Optimizer not updating
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"No gradient for {name}")
```

#### OOM (Out of Memory)
```python
# Solutions:
# 1. Reduce device_batch_size
device_batch_size = 16  # Was 32

# 2. Reduce sequence_len
sequence_len = 1024  # Was 2048

# 3. Use gradient checkpointing
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    for block in self.blocks:
        x = checkpoint(block, x)  # Trade compute for memory
    return x

# 4. Clear cache periodically
if step % 100 == 0:
    torch.cuda.empty_cache()
```

### Task 6: Profile Performance

```python
# File: scripts/profile_train.py

import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    # Run training for a few steps
    for step in range(10):
        loss = train_step(model, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Print results
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Export Chrome trace
prof.export_chrome_trace("trace.json")
# Open trace.json in chrome://tracing
```

### Task 7: Add Weights & Biases Logging

```python
# Already integrated! Just set environment variable:
export WANDB_PROJECT="nanochat"
export WANDB_ENTITY="your-username"

# Or in script:
import wandb

wandb.init(
    project="nanochat",
    config={
        'depth': depth,
        'lr': lr,
        'batch_size': device_batch_size,
    }
)

# Log metrics
wandb.log({
    'train/loss': loss.item(),
    'train/lr': optimizer.param_groups[0]['lr'],
}, step=step)
```

---

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_rustbpe.py -v

# Run with output
python -m pytest tests/test_rustbpe.py -v -s

# Run specific test
python -m pytest tests/test_rustbpe.py::test_encode -v
```

### Writing Tests

```python
# File: tests/test_model.py

import torch
from nanochat.gpt import GPT, GPTConfig

def test_model_forward():
    """Test forward pass."""
    config = GPTConfig(
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=128,
        sequence_len=64,
    )
    model = GPT(config)

    # Create dummy input
    batch_size = 4
    seq_len = 32
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits = model(x)

    # Check output shape
    assert logits.shape == (batch_size, seq_len, config.vocab_size)

def test_model_generation():
    """Test autoregressive generation."""
    config = GPTConfig(vocab_size=100, depth=2)
    model = GPT(config)
    model.eval()

    # Generate
    prompt = torch.tensor([[1, 2, 3]])
    output = model.generate(prompt, max_new_tokens=10)

    # Check output shape
    assert output.shape == (1, 13)  # 3 + 10

def test_kv_cache():
    """Test KV cache correctness."""
    from nanochat.engine import KVCache

    config = GPTConfig(vocab_size=100, depth=2)
    model = GPT(config)

    prompt = torch.tensor([[1, 2, 3, 4, 5]])

    # Generate without cache
    with torch.no_grad():
        logits_no_cache = model(prompt)

    # Generate with cache
    kv_cache = KVCache(config)
    with torch.no_grad():
        logits_cache, _ = model(prompt, kv_cache=kv_cache)

    # Should be identical
    assert torch.allclose(logits_no_cache, logits_cache, atol=1e-5)
```

---

## Debugging Tips

### 1. Enable Detailed Logging

```python
# File: nanochat/common.py

import logging

# Set to DEBUG for verbose output
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### 2. Inspect Model Internals

```python
# Add hooks to inspect activations
def print_activation(name):
    def hook(module, input, output):
        print(f"{name}: {output.shape}, mean={output.mean():.4f}, std={output.std():.4f}")
    return hook

# Register hooks
model.transformer.h[0].attn.register_forward_hook(print_activation('layer0_attn'))
model.transformer.h[0].mlp.register_forward_hook(print_activation('layer0_mlp'))
```

### 3. Gradient Checks

```python
# Check gradient flow
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: grad_norm={grad_norm:.6f}")
    else:
        print(f"{name}: NO GRADIENT")
```

### 4. Memory Profiling

```python
import torch

def print_memory_stats():
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Call periodically
print_memory_stats()
```

### 5. Debugging Distributed Training

```python
# Only print from rank 0
if rank == 0:
    print(f"Step {step}, Loss: {loss.item()}")

# Synchronize all processes
torch.distributed.barrier()

# Check if all GPUs have same gradients (should after DDP backward)
if rank == 0:
    for name, param in model.named_parameters():
        # Gather gradients from all ranks
        # They should be identical after DDP backward
        pass
```

---

## Performance Considerations

### 1. Batch Size Tuning

**Find optimal batch size:**
```python
# Start with largest batch size that fits in memory
device_batch_size = 32

# If OOM, reduce by half
device_batch_size = 16  # or 8, 4, 2, 1

# Compensate with gradient accumulation
# effective_batch = device_batch_size * grad_accum_steps * world_size
# Keep effective_batch constant by adjusting grad_accum_steps
```

**Memory vs. speed tradeoff:**
- Larger batch size → Better GPU utilization → Faster training
- Smaller batch size → Less memory → Can fit larger models

### 2. Mixed Precision Training

```python
# Use automatic mixed precision (AMP)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()

    # Forward pass in mixed precision
    with autocast():
        loss = model(batch)

    # Backward with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Already enabled in nanochat:**
- Model runs in bfloat16 automatically on supported GPUs
- No need for gradient scaling with bfloat16

### 3. Data Loading Optimization

```python
# Use multiple workers
train_loader = DataLoader(
    dataset,
    batch_size=device_batch_size,
    num_workers=4,  # Load data in parallel
    pin_memory=True,  # Faster CPU→GPU transfer
    prefetch_factor=2,  # Prefetch batches
)
```

### 4. Compilation (PyTorch 2.0+)

```python
# Compile model for faster execution
model = torch.compile(model, mode='max-autotune')

# Or for less aggressive optimization
model = torch.compile(model, mode='reduce-overhead')
```

**Trade-offs:**
- First run is slower (compilation time)
- Subsequent runs are faster
- May not work with all code patterns

### 5. Inference Optimization

```python
# Use KV cache (already implemented)
# Use smaller precision
model = model.half()  # fp16

# Or even int8 quantization (requires additional setup)
# Use batch inference
prompts = ["prompt1", "prompt2", "prompt3"]
# Process in parallel instead of sequentially
```

---

## Contributing Guidelines

### Code Style

1. **Follow PEP 8** with exceptions:
   - Line length: 100 characters (not 79)
   - Use descriptive variable names

2. **Type hints encouraged** but not required:
   ```python
   def train_step(model: GPT, batch: torch.Tensor) -> float:
       ...
   ```

3. **Docstrings for public APIs**:
   ```python
   def generate(self, prompt, max_new_tokens=50):
       """
       Generate text autoregressively.

       Args:
           prompt (torch.Tensor): Input token IDs, shape (batch, seq_len)
           max_new_tokens (int): Number of tokens to generate

       Returns:
           torch.Tensor: Generated token IDs, shape (batch, seq_len + max_new_tokens)
       """
   ```

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Maintenance

**Examples:**
```
feat(gpt): add dropout to transformer blocks

- Add dropout config parameter
- Apply dropout after attention and MLP
- Update all training scripts to support dropout

Closes #123
```

```
fix(tokenizer): handle empty string encoding

Previously crashed with IndexError when encoding empty string.
Now returns empty token list.
```

### Pull Request Process

1. **Fork and create branch**:
   ```bash
   git checkout -b feat/my-feature
   ```

2. **Make changes and test**:
   ```bash
   python -m pytest tests/ -v
   ```

3. **Commit with clear messages**:
   ```bash
   git commit -m "feat: add new evaluation task"
   ```

4. **Push and create PR**:
   ```bash
   git push origin feat/my-feature
   # Create PR on GitHub
   ```

5. **PR description should include**:
   - What problem does this solve?
   - What changes were made?
   - How to test the changes?
   - Any breaking changes?

### What to Contribute

**Welcomed contributions:**
- Bug fixes
- Performance improvements
- New evaluation tasks
- Documentation improvements
- Training optimizations
- Inference speedups

**Not welcomed:**
- Complex abstraction layers
- Excessive configurability
- Large dependency additions
- Breaking changes to core APIs

---

## Troubleshooting

### Issue: Import Error

```
ModuleNotFoundError: No module named 'rustbpe'
```

**Solution:**
```bash
cd rustbpe
maturin develop --release
cd ..
```

### Issue: CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce `device_batch_size`:
   ```bash
   python -m scripts.base_train --device_batch_size=16  # or 8, 4, 2
   ```

2. Reduce `sequence_len`:
   ```bash
   python -m scripts.base_train --sequence_len=1024
   ```

3. Reduce model `depth`:
   ```bash
   python -m scripts.base_train --depth=16
   ```

### Issue: Slow Data Loading

```
Training is slow, GPU utilization is low
```

**Solution:**
```bash
# Pre-download all data shards before training
python -m nanochat.dataset -n 100

# Increase DataLoader workers (in script)
num_workers = 4  # More parallel data loading
```

### Issue: Loss is NaN

```
Step 100: loss = nan
```

**Solutions:**
1. Check for bad data:
   ```python
   assert not torch.isnan(batch).any()
   ```

2. Reduce learning rate:
   ```python
   lr = 0.001  # Was 0.01
   ```

3. Add gradient clipping:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

4. Check for inf/nan in logits:
   ```python
   assert torch.isfinite(logits).all(), "Non-finite logits"
   ```

### Issue: Distributed Training Hangs

```
Stuck at "Initializing distributed training..."
```

**Solutions:**
1. Check all GPUs are accessible:
   ```bash
   nvidia-smi
   ```

2. Set timeout:
   ```bash
   export NCCL_TIMEOUT=3600  # 1 hour
   ```

3. Debug with single GPU first:
   ```bash
   python -m scripts.base_train  # No torchrun
   ```

### Issue: Tokenizer Vocabulary Mismatch

```
RuntimeError: Embedding size mismatch
```

**Solution:**
```bash
# Make sure vocab_size in config matches tokenizer
python -c "import tiktoken; enc = tiktoken.get_encoding('gpt2'); print(enc.n_vocab)"

# Update GPTConfig.vocab_size to match
```

### Issue: Checkpoint Loading Fails

```
KeyError: 'model'
```

**Solutions:**
1. Check checkpoint format:
   ```python
   import torch
   ckpt = torch.load('checkpoint.pt')
   print(ckpt.keys())  # Should have 'model', 'optimizer', 'config'
   ```

2. Use checkpoint_manager:
   ```python
   from nanochat.checkpoint_manager import load_checkpoint
   model, optimizer, metadata = load_checkpoint('checkpoint.pt', model, optimizer, device)
   ```

---

## Resources

### Official Documentation
- **README.md**: Main project documentation
- **Discussions**: https://github.com/karpathy/nanochat/discussions
- **Issues**: https://github.com/karpathy/nanochat/issues

### Related Projects
- **nanoGPT**: https://github.com/karpathy/nanoGPT (pretraining only)
- **modded-nanoGPT**: https://github.com/KellerJordan/modded-nanogpt (speedrun optimizations)

### Datasets
- **FineWeb-Edu**: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- **SmolTalk**: https://huggingface.co/datasets/HuggingFaceTB/smoltalk

### Papers & References

**Transformer Architecture:**
- Attention Is All You Need (Vaswani et al., 2017)
- RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)
- Fast Transformer Decoding: One Write-Head is All You Need (Shazeer, 2019) - Multi-Query Attention

**Optimization:**
- Muon Optimizer (Jordan et al., 2024)
- Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)

**Training Techniques:**
- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (Rajbhandari et al., 2020)
- Mixed Precision Training (Micikevicius et al., 2017)

### Community
- **Discord**: (Check repo for link)
- **Twitter**: Follow @karpathy for updates

### Tools
- **Weights & Biases**: https://wandb.ai/ (experiment tracking)
- **DeepWiki**: https://deepwiki.com/ (code exploration)
- **files-to-prompt**: https://github.com/simonw/files-to-prompt (package codebase for LLMs)

---

## Quick Reference

### Common Commands

```bash
# Setup
uv venv && source .venv/bin/activate && uv sync
cd rustbpe && maturin develop --release && cd ..

# Training pipeline
python -m scripts.tok_train
torchrun --standalone --nproc_per_node=8 -m scripts.base_train
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft

# Inference
python -m scripts.chat_cli
python -m scripts.chat_web

# Evaluation
python -m scripts.base_eval
python -m scripts.chat_eval

# Testing
python -m pytest tests/ -v
```

### File Quick Access

| Task | File |
|------|------|
| Modify model architecture | `nanochat/gpt.py` |
| Change training loop | `scripts/base_train.py`, `scripts/chat_sft.py` |
| Add evaluation task | `tasks/` |
| Update tokenizer | `nanochat/tokenizer.py`, `rustbpe/src/lib.rs` |
| Modify web UI | `dev/web/index.html`, `scripts/chat_web.py` |
| Add optimizer | `nanochat/` (create new file) |
| Update hyperparameters | Top of training scripts |

---

## Conclusion

You now have a comprehensive understanding of the nanochat codebase! Remember:

1. **Start small**: Train tiny models locally before scaling up
2. **Read the code**: ~8K lines is totally digestible
3. **Ask questions**: Use Discussions on GitHub
4. **Experiment**: Fork and hack away
5. **Share**: Contribute back improvements

The best way to learn is to train your own model. Start with `speedrun.sh` on a GPU node, or `dev/runcpu.sh` locally, and watch the magic happen.

Happy hacking! 🚀
