# Nanochat Complete Documentation Guide

Welcome! This documentation will teach you everything you need to know about Large Language Models (LLMs) and how this nanochat codebase works. **No prior knowledge assumed** - we'll start from the basics and build up to the complete system.

## 📚 Documentation Structure

This guide is organized into 8 comprehensive documents. **Read them in order** if you're completely new to LLMs:

### 1. [Introduction to LLMs and PyTorch](01_introduction.md)
**Start here if you're new to AI/ML**
- What is a Large Language Model?
- What is PyTorch and why do we use it?
- How does machine learning work?
- Overview of the nanochat system
- Key concepts and terminology

### 2. [Tokenization: From Text to Numbers](02_tokenization.md)
**How computers understand text**
- Why we can't just use characters
- What is Byte Pair Encoding (BPE)?
- How the tokenizer is trained
- Special tokens for conversations
- Code walkthrough: `tokenizer.py` and `rustbpe/`

### 3. [The GPT Architecture](03_architecture.md)
**Deep dive into the neural network**
- What is a Transformer?
- Attention mechanisms explained simply
- Layer-by-layer breakdown of the model
- How embeddings work
- Mathematical operations (explained for beginners)
- Code walkthrough: `gpt.py`

### 4. [The Training Pipeline](04_training.md)
**How the model learns**
- What is training vs fine-tuning?
- The complete training pipeline (pretraining → mid-training → SFT → RL)
- Loss functions and optimization
- Distributed training on multiple GPUs
- Checkpointing and resuming training
- Code walkthrough: `base_train.py`, `chat_sft.py`, etc.

### 5. [Inference: How Text Generation Works](05_inference.md)
**How the model generates responses**
- Autoregressive generation explained
- Sampling strategies (temperature, top-p, top-k)
- KV cache optimization
- The conversation format
- Code walkthrough: `engine.py`

### 6. [Tools and Capabilities](06_tools.md)
**Advanced features**
- Calculator tool integration
- Python code execution
- How the model decides when to use tools
- Special tokens for tool use
- Code walkthrough: `execution.py`

### 7. [Evaluation and Benchmarks](07_evaluation.md)
**Measuring model quality**
- What is CORE score?
- Task-based evaluation (ARC, GSM8K, MMLU, HumanEval)
- Bits-per-byte metrics
- How to interpret results
- Code walkthrough: `core_eval.py`, `tasks/`

### 8. [Quick Start Guide](08_quickstart.md)
**Getting up and running**
- Installation and setup
- Running your first training
- Using the web interface
- Customizing the model
- Troubleshooting common issues

### 9. [Feature Implementation Guide](09_feature_implementation_guide.md)
**Hands-on learning through building features**
- 10 beginner-friendly features to add to nanochat
- Complete implementations with detailed explanations
- Step-by-step coding instructions
- Learn by doing: tokenization, training, inference tools
- See also: [Part 2](09_feature_implementation_guide_part2.md) for additional features

## 🛠️ Available Tools

The `tools/` directory contains ready-to-use utilities for learning and experimentation:

- **tokenizer_playground.py** - Interactive tokenizer visualization
  - See how text is tokenized in real-time with color-coding
  - Understand token boundaries and special tokens
  - Compare tokenization efficiency of different texts
  - Perfect for understanding how "Hello world" becomes `[1000, 1001, 33]`

- **model_calculator.py** - Calculate model size, memory, and training costs
  - Estimate parameters for any model configuration
  - Predict GPU memory requirements
  - Calculate training time and FLOPs
  - Perfect for understanding model scaling

More tools coming soon! See the [Feature Implementation Guide](09_feature_implementation_guide.md) for features you can build.

## 🎯 How to Use This Documentation

### If you're completely new to LLMs:
1. Read documents 1-8 in order
2. Don't skip sections - each builds on previous knowledge
3. Try the code examples as you go
4. Re-read sections that are confusing (this is complex stuff!)
5. Build features from document 9 to practice what you learned

### If you have some ML background:
1. Skim document 1
2. Focus on documents 2-5 for implementation details
3. Reference documents 6-7 as needed
4. Jump to document 9 to build practical tools

### If you just want to run the code:
1. Go straight to document 8 (Quick Start)
2. Use tools from the `tools/` directory
3. Reference other docs when you need to understand specifics

### If you learn best by building:
1. Read document 1-2 for basics
2. Jump to document 9 and start building features
3. Reference other docs as needed while implementing

## 🔑 Key Concepts You'll Learn

By the end of this documentation, you'll understand:

- **Tokenization**: How "Hello world" becomes `[15496, 995]`
- **Embeddings**: How numbers become vectors in high-dimensional space
- **Attention**: How the model "looks at" previous words
- **Training**: How the model learns patterns from data
- **Generation**: How the model creates new text one token at a time
- **Fine-tuning**: How we teach the model to chat
- **Evaluation**: How we know if the model is good

## 🛠️ Code Organization Reference

Quick reference for where to find things:

```
nanochat/
├── nanochat/              # Core library
│   ├── gpt.py            # The neural network model (see doc 3)
│   ├── tokenizer.py      # Text → numbers (see doc 2)
│   ├── engine.py         # Text generation (see doc 5)
│   ├── dataloader.py     # Data loading for training (see doc 4)
│   ├── execution.py      # Code execution tool (see doc 6)
│   └── core_eval.py      # Model evaluation (see doc 7)
├── scripts/               # Entry points
│   ├── base_train.py     # Pretraining (see doc 4)
│   ├── chat_sft.py       # Fine-tuning (see doc 4)
│   ├── chat_web.py       # Web interface (see doc 8)
│   └── chat_eval.py      # Run benchmarks (see doc 7)
├── rustbpe/               # Tokenizer training (see doc 2)
├── tasks/                 # Evaluation tasks (see doc 7)
├── docs/                  # Complete documentation (you are here!)
└── tools/                 # Learning and utility tools (see doc 9)
    ├── tokenizer_playground.py  # Interactive tokenizer visualization
    └── model_calculator.py      # Model size & cost calculator
```

## 📖 Glossary

Quick reference for terms you'll encounter:

- **LLM**: Large Language Model - a neural network trained on text
- **Token**: A piece of text (usually part of a word) that the model processes
- **Embedding**: Converting a token into a vector of numbers
- **Transformer**: The architecture that modern LLMs use
- **Attention**: Mechanism for looking at relevant previous tokens
- **Parameter**: A number in the model that gets learned during training
- **Checkpoint**: A saved snapshot of the model's parameters
- **Loss**: How wrong the model's predictions are (lower is better)
- **Inference**: Using a trained model to generate text
- **Fine-tuning**: Additional training to specialize the model
- **SFT**: Supervised Fine-Tuning - training on instruction-response pairs
- **RL**: Reinforcement Learning - training using reward signals

## 💡 Learning Tips

1. **Don't memorize formulas** - Focus on understanding what they do
2. **Run the code** - Seeing it work helps understanding
3. **Modify things** - Change parameters and see what happens
4. **Take breaks** - This is dense material; process it in chunks
5. **Ask questions** - Use comments in code to note confusion points
6. **Draw diagrams** - Visualize data flowing through the system

## 🚀 Next Steps

Ready to start? Head to [Document 1: Introduction to LLMs and PyTorch](01_introduction.md)!

---

**Note**: This documentation was created to be self-contained. You should be able to understand the entire system using only these documents and the code. If something is unclear, re-read the section and trace through the code - the understanding will come with time and practice.

Good luck on your LLM learning journey! 🎓
