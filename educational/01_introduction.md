# Introduction to nanochat: Building a ChatGPT from Scratch

## What is nanochat?

nanochat is a complete, minimal implementation of a Large Language Model (LLM) similar to ChatGPT. Unlike most LLM projects that rely on heavy external frameworks, nanochat is built from scratch with minimal dependencies, making it perfect for learning how modern LLMs actually work.

**Key Philosophy:**
- **From Scratch**: Implement core algorithms yourself rather than using black-box libraries
- **Minimal Dependencies**: Only essential libraries (PyTorch, tokenizers, etc.)
- **Educational**: Clean, readable code that you can understand completely
- **Full Stack**: Everything from tokenization to web serving
- **Practical**: Actually trains a working model for ~$100

## What You'll Learn

By studying this repository, you will understand:

1. **Tokenization**: How text is converted to numbers using Byte Pair Encoding (BPE)
2. **Model Architecture**: The Transformer architecture with modern improvements
3. **Training Pipeline**:
   - **Pretraining**: Learning language patterns from raw text
   - **Midtraining**: Specialized training on curated data
   - **Supervised Fine-Tuning (SFT)**: Teaching the model to chat
   - **Reinforcement Learning (RL)**: Optimizing for quality
4. **Optimization**: Advanced optimizers like Muon and AdamW
5. **Evaluation**: Measuring model performance
6. **Inference**: Running the trained model efficiently
7. **Deployment**: Serving the model via a web interface

## Repository Structure

```
nanochat/
├── nanochat/              # Core library
│   ├── gpt.py            # GPT model architecture
│   ├── tokenizer.py      # BPE tokenizer wrapper
│   ├── dataloader.py     # Data loading and tokenization
│   ├── engine.py         # Inference engine
│   ├── adamw.py          # AdamW optimizer
│   ├── muon.py           # Muon optimizer
│   └── ...
├── rustbpe/              # High-performance Rust tokenizer
│   └── src/lib.rs        # BPE implementation in Rust
├── scripts/              # Training and evaluation scripts
│   ├── base_train.py     # Pretraining script
│   ├── mid_train.py      # Midtraining script
│   ├── chat_sft.py       # Supervised fine-tuning
│   ├── chat_rl.py        # Reinforcement learning
│   └── chat_web.py       # Web interface
├── tasks/                # Evaluation benchmarks
├── tests/                # Unit tests
└── speedrun.sh           # Complete pipeline script
```

## The Training Pipeline

nanochat implements the complete modern LLM training pipeline:

### 1. Tokenization (tok_train.py)
First, we need to convert text into numbers. We train a **Byte Pair Encoding (BPE)** tokenizer on a corpus of text. This creates a vocabulary of ~32,000 tokens that efficiently represent common words and subwords.

**Time**: ~10 minutes on CPU

### 2. Base Pretraining (base_train.py)
The model learns to predict the next token in sequences of text. This is where most of the "knowledge" is learned - language patterns, facts, reasoning abilities, etc.

**Data**: ~10 billion tokens from FineWeb (high-quality web text)
**Objective**: Next-token prediction
**Time**: ~2-4 hours on 8×H100 GPUs
**Cost**: ~$100

### 3. Midtraining (mid_train.py)
Continue pretraining on a smaller, more curated dataset to improve quality and reduce the need for instruction following data.

**Data**: ~1 billion high-quality tokens
**Time**: ~30 minutes
**Cost**: ~$12

### 4. Supervised Fine-Tuning (chat_sft.py)
Teach the model to follow instructions and chat like ChatGPT. We train on conversation examples.

**Data**: ~80,000 conversations from SmolTalk
**Objective**: Predict assistant responses given user prompts
**Time**: ~15 minutes
**Cost**: ~$6

### 5. Reinforcement Learning (chat_rl.py)
Further optimize the model using reinforcement learning to improve response quality.

**Technique**: Self-improvement via sampling and filtering
**Time**: ~10 minutes
**Cost**: ~$4

## Key Technical Features

### Modern Architecture Choices

The GPT model in nanochat includes modern improvements over the original GPT-2:

1. **Rotary Position Embeddings (RoPE)**: Better position encoding
2. **RMSNorm**: Simpler, more efficient normalization
3. **Multi-Query Attention (MQA)**: Faster inference
4. **QK Normalization**: Stability improvement
5. **ReLU² Activation**: Better than GELU for small models
6. **Untied Embeddings**: Separate input/output embeddings
7. **Logit Softcapping**: Prevents extreme logits

### Efficient Implementation

- **Mixed Precision**: BF16 for most operations
- **Gradient Accumulation**: Larger effective batch sizes
- **Distributed Training**: Multi-GPU support with DDP
- **Compiled Models**: PyTorch compilation for speed
- **Streaming Data**: Memory-efficient data loading
- **Rust Tokenizer**: Fast tokenization with parallel processing

## Mathematical Notation

Throughout this guide, we'll use the following notation:

- $d_{model}$: Model dimension (embedding size)
- $n_{layers}$: Number of Transformer layers
- $n_{heads}$: Number of attention heads
- $d_{head}$: Dimension per attention head ($d_{model} / n_{heads}$)
- $V$: Vocabulary size
- $T$ or $L$: Sequence length
- $B$: Batch size
- $\theta$: Model parameters
- $\mathcal{L}$: Loss function
- $p(x)$: Probability distribution

## Prerequisites

To fully understand this material, you should have:

**Essential:**
- Python programming
- Basic linear algebra (matrices, vectors, dot products)
- Basic calculus (derivatives, chain rule)
- Basic probability (distributions, expectation)

**Helpful but not required:**
- PyTorch basics
- Deep learning fundamentals
- Transformer architecture awareness

Don't worry if you're not an expert! We'll explain everything step by step.

## How to Use This Guide

The educational materials are organized as follows:

1. **01_introduction.md** (this file): Overview and context
2. **02_mathematical_foundations.md**: Math concepts you need
3. **03_tokenization.md**: BPE algorithm and implementation
4. **04_transformer_architecture.md**: The GPT model structure
5. **05_attention_mechanism.md**: Self-attention in detail
6. **06_training_process.md**: How training works
7. **07_optimization.md**: Advanced optimizers (Muon, AdamW)
8. **08_implementation_details.md**: Code walkthrough
9. **09_evaluation.md**: Measuring model performance
10. **10_rust_implementation.md**: High-performance Rust tokenizer

Each section builds on previous ones, so it's best to read them in order.

## Running the Code

To get started with nanochat:

```bash
# Clone the repository
git clone https://github.com/karpathy/nanochat.git
cd nanochat

# Install dependencies (requires Python 3.10+)
pip install uv
uv sync

# Run the complete pipeline (requires 8×H100 GPUs)
bash speedrun.sh
```

For learning purposes, you can also:

```bash
# Run tests
python -m pytest tests/ -v

# Train tokenizer only
python -m scripts.tok_train

# Train small model on 1 GPU
python -m scripts.base_train --depth=6
```

## Next Steps

In the next section, we'll cover the **Mathematical Foundations** - all the math concepts you need to understand how LLMs work, explained from first principles.

Let's begin! 🚀
