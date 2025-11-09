# Nanochat Learning Fork - Additional Features

This is a learning-focused fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) designed for beginners who want to deeply understand LLMs and PyTorch. All additions focus on educational value and hands-on learning.

## About This Fork

**Original Repository**: [karpathy/nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy

**This Fork's Purpose**:
- Provide comprehensive beginner-friendly documentation
- Add practical tools for learning and experimentation
- Enable hands-on feature building to understand LLM internals
- Maintain simplicity and minimal dependencies

**Target Audience**: Complete beginners to LLMs and PyTorch who want to learn by doing.

## What's Added

### 📚 Comprehensive Documentation (`docs/`)

Eight detailed guides that teach LLMs from the ground up, assuming NO prior knowledge:

1. **Introduction to LLMs and PyTorch** - What are LLMs? What is PyTorch? How does ML work?
2. **Tokenization** - How text becomes numbers, BPE algorithm explained
3. **Architecture** - Complete Transformer architecture breakdown with code walkthroughs
4. **Training Pipeline** - How models learn, distributed training, optimizers
5. **Inference** - Text generation, sampling strategies, KV cache optimization
6. **Tools and Capabilities** - Calculator and code execution features
7. **Evaluation** - CORE score, benchmarks, metrics explained
8. **Quick Start Guide** - Installation, first training, troubleshooting
9. **Feature Implementation Guide** - 10 features you can build to learn

**Key Principle**: All documentation is self-contained. You should be able to understand the entire system using ONLY these docs and the code, with no external resources needed.

### 🛠️ Learning Tools (`tools/`)

Practical utilities for understanding model behavior and planning experiments:

#### ✅ Implemented Features

##### 1. Model Size & Cost Calculator (`model_calculator.py`)
Calculate parameters, memory, and training costs for any model configuration.

**What it does:**
- Counts parameters for all model components (embeddings, attention, MLP)
- Estimates memory requirements (fp32, fp16, training, inference)
- Predicts training time and computational cost (FLOPs)
- Provides educational insights about parameter distribution

**Why it's useful:**
- Understand how model size scales with dimensions
- Avoid GPU OOM errors by predicting memory needs
- Plan training experiments and timelines
- Learn where parameters come from in Transformers

**Usage:**
```bash
# Use preset configurations
python tools/model_calculator.py --preset nanochat-tiny
python tools/model_calculator.py --preset gpt2-small

# Custom configuration
python tools/model_calculator.py --depth 12 --hidden-dim 768 --vocab-size 32000

# Customize training parameters
python tools/model_calculator.py --preset gpt2-small --batch-size 32 --total-tokens 20000000000
```

**Example output:**
```
======================================================================
MODEL SIZE & COST CALCULATOR
======================================================================

📊 MODEL CONFIGURATION
----------------------------------------------------------------------
  Layers (depth):        6
  Hidden dimension:      384
  Vocabulary size:       32,000
  Attention heads:       6

🔢 PARAMETER BREAKDOWN
----------------------------------------------------------------------
  Token embeddings:           12,288,000 params
  Per-layer breakdown:
    - Attention:                 589,824 params
    - MLP:                     1,179,648 params
    - LayerNorm:                   1,536 params
    - Total per layer:         1,771,008 params
  All 6 layers:             10,626,048 params
  Final LayerNorm:                   768 params
  LM head:                    12,288,000 params
  ────────────────────────────────────────────────────────────
  TOTAL PARAMETERS:           35,202,816 params
                                   35.20 M

💾 MEMORY REQUIREMENTS
----------------------------------------------------------------------
  Model weights (fp32):        0.13 GB
  Model weights (fp16):        0.07 GB
  Training (fp32+opt):         0.52 GB
  Inference (fp16):            0.07 GB

⏱️  TRAINING ESTIMATES
----------------------------------------------------------------------
  Training tokens:       10.0B tokens
  Batch size:            64 sequences
  Sequence length:       1024 tokens
  Tokens per batch:      65,536
  Total steps:           152,587
  Throughput:            100,000 tokens/sec
  Training time:         27.8 hours (1.2 days)
  Total FLOPs:           2112.2 PetaFLOPs

💡 LEARNING INSIGHTS
----------------------------------------------------------------------
  • Embeddings use ~34.9% of parameters
  • Attention layers use ~10.1% of parameters
  • MLP layers use ~20.1% of parameters
  • Training needs ~4x more memory than inference
  • Each parameter sees 284 tokens during training
======================================================================
```

**Dependencies:** None (Python standard library only)

**Learning outcomes:**
- Understand parameter counting in Transformers
- Learn about memory requirements for different precisions
- See how batch size and sequence length affect training
- Calculate FLOPs for computational cost estimation

#### 🔜 Planned Features (See `docs/09_feature_implementation_guide.md`)

2. **Interactive Tokenizer Playground** - Visualize tokenization with colors, compare tokenizers
3. **Training Progress Dashboard** - Real-time visualization of training metrics
4. **Dataset Inspector** - Validate and analyze training data
5. **Checkpoint Browser & Comparator** - Explore saved models and compare performance
6. **Generation Parameter Explorer** - Experiment with temperature, top-k, top-p
7. **Training Resume Helper** - Easily resume training from checkpoints
8. **Simple Attention Visualizer** - See what the model attends to
9. **Learning Rate Finder** - Find optimal learning rate automatically
10. **Conversation Template Builder** - Create and test custom chat templates

## Design Principles

All additions follow these principles:

1. **Educational First** - Every feature teaches you something about LLMs
2. **Minimal Dependencies** - Use standard library when possible, avoid bloat
3. **Simple Implementation** - Code should be readable by beginners
4. **No GPU Required** (for tools) - Learning tools work on any machine
5. **Self-Contained** - Documentation explains everything needed

## How to Use This Fork

### For Learning:
1. Read the documentation in `docs/` sequentially
2. Use the tools in `tools/` to experiment
3. Build the features from `docs/09_feature_implementation_guide.md`
4. Modify and extend features to deepen understanding

### For Experimentation:
1. Use `tools/model_calculator.py` to plan your experiment
2. Follow `docs/08_quickstart.md` to run training
3. Use the web interface or CLI to test your model
4. Build additional tools as needed

### For Contributing:
This is a personal learning fork. Feel free to fork it further for your own learning journey!

## Differences from Original

### Added:
- Complete beginner documentation (8 guides + 1 feature guide)
- Learning tools directory with utilities
- Feature implementation guide with 10 hands-on projects

### Unchanged:
- All core nanochat functionality
- Training pipeline and scripts
- Model architecture
- Evaluation framework

### Philosophy:
The original nanochat is minimalist and production-focused. This fork adds a comprehensive learning layer on top without modifying the core system.

## Status

- ✅ Documentation: Complete (9 guides covering all aspects)
- ✅ Tools: 1/10 features implemented
- 🔄 Actively adding more learning features

## Acknowledgments

Huge thanks to [Andrej Karpathy](https://github.com/karpathy) for creating nanochat - a beautifully simple and educational LLM implementation that makes learning accessible.

This fork wouldn't exist without his excellent work on making AI education approachable.

## License

Same as original nanochat repository (MIT License).

---

**Note**: This is a learning fork. For production use, refer to the [original nanochat repository](https://github.com/karpathy/nanochat).
