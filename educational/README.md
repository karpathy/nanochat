# Educational Guide to nanochat

This folder contains a comprehensive educational guide to understanding and building your own Large Language Model (LLM) from scratch, using nanochat as a reference implementation.

## What's Included

This guide covers everything from mathematical foundations to practical implementation:

### 📚 Core Materials

1. **[01_introduction.md](01_introduction.md)** - Overview of nanochat and the LLM training pipeline
2. **[02_mathematical_foundations.md](02_mathematical_foundations.md)** - All the math you need (linear algebra, probability, optimization)
3. **[03_tokenization.md](03_tokenization.md)** - Byte Pair Encoding (BPE) algorithm with detailed code walkthrough
4. **[04_transformer_architecture.md](04_transformer_architecture.md)** - GPT model architecture and components
5. **[05_attention_mechanism.md](05_attention_mechanism.md)** - Deep dive into self-attention with implementation details
6. **[06_training_process.md](06_training_process.md)** - Complete training pipeline from data loading to checkpointing
7. **[07_optimization.md](07_optimization.md)** - Advanced optimizers (Muon + AdamW) with detailed explanations
8. **[08_putting_it_together.md](08_putting_it_together.md)** - Practical implementation guide and debugging tips

### 🎯 Who This Is For

- **Beginners**: Start from first principles with clear explanations
- **Intermediate**: Deep dive into implementation details and code
- **Advanced**: Learn cutting-edge techniques (RoPE, Muon, MQA)

## How to Use This Guide

### Sequential Reading (Recommended for Beginners)

Read in order from 01 to 08. Each section builds on previous ones:

```
Introduction → Math → Tokenization → Architecture →
Attention → Training → Optimization → Implementation
```

### Topic-Based Reading (For Experienced Practitioners)

Jump directly to topics of interest:
- **Want to understand tokenization?** → Read `03_tokenization.md`
- **Need to implement attention?** → Read `05_attention_mechanism.md`
- **Optimizing training?** → Read `07_optimization.md`

### Code Walkthrough (Best for Implementation)

Read alongside the nanochat codebase:
1. Read a section (e.g., "Transformer Architecture")
2. Open the corresponding file (`nanochat/gpt.py`)
3. Follow along with the code examples
4. Modify and experiment

## Compiling to PDF

To create a single PDF document from all sections:

```bash
cd educational
python compile_to_pdf.py
```

This will generate `nanochat_educational_guide.pdf`.

**Requirements:**
- Python 3.7+
- pandoc
- LaTeX distribution (e.g., TeX Live, MiKTeX)

Install dependencies:
```bash
# macOS
brew install pandoc
brew install basictex  # or MacTeX for full distribution

# Ubuntu/Debian
sudo apt-get install pandoc texlive-full

# Python packages
pip install pandoc
```

## Key Features of This Guide

### 🎓 Educational Approach
- **From first principles**: Assumes only basic Python and math knowledge
- **Progressive complexity**: Start simple, build up gradually
- **Concrete examples**: Real code from nanochat, not pseudocode

### 💻 Code-Focused
- **Deep code explanations**: Every important function is explained line-by-line
- **Implementation patterns**: Learn best practices and design patterns
- **Debugging tips**: Common pitfalls and how to avoid them

### 🔬 Comprehensive
- **Mathematical foundations**: Understand the "why" behind every technique
- **Modern techniques**: RoPE, MQA, Muon optimizer, softcapping
- **Full pipeline**: From raw text to deployed chatbot

### 🚀 Practical
- **Runnable examples**: All code can be tested immediately
- **Optimization tips**: Make training fast and efficient
- **Scaling guidance**: From toy models to production systems

## What You'll Learn

By the end of this guide, you'll understand:

✅ How tokenization works (BPE algorithm)
✅ Transformer architecture in detail
✅ Self-attention mechanism (with RoPE, MQA)
✅ Training loop and data pipeline
✅ Advanced optimization (Muon + AdamW)
✅ Mixed precision training (BF16)
✅ Distributed training (DDP)
✅ Evaluation and metrics
✅ How to implement your own LLM

## Prerequisites

**Essential:**
- Python programming
- Basic linear algebra (matrices, vectors, dot products)
- Basic calculus (derivatives, chain rule)
- Basic probability (distributions)

**Helpful but not required:**
- PyTorch basics
- Deep learning fundamentals
- Familiarity with Transformers

## Additional Resources

### Official Documentation
- [nanochat GitHub](https://github.com/karpathy/nanochat)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [HuggingFace Transformers](https://huggingface.co/transformers/)

### Related Projects
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Pretraining only
- [minGPT](https://github.com/karpathy/minGPT) - Educational GPT
- [llm.c](https://github.com/karpathy/llm.c) - GPT in C/CUDA

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) - GPT-3
- [Training Compute-Optimal LLMs](https://arxiv.org/abs/2203.15556) - Chinchilla scaling laws

## Contributing

Found an error or want to improve the guide?
1. Open an issue on the main nanochat repository
2. Suggest improvements or clarifications
3. Share what topics you'd like to see covered

## License

This educational material follows the same MIT license as nanochat.

## Acknowledgments

This guide is based on the nanochat implementation by Andrej Karpathy. All code examples are from the nanochat repository.

Special thanks to the open-source community for making LLM education accessible!

---

**Happy learning! 🚀**

If you find this guide helpful, please star the [nanochat repository](https://github.com/karpathy/nanochat)!
