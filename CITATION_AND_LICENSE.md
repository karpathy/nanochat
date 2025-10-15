# Citation and License Information

## 📖 Citation

If you find **nanochat** helpful in your research, please cite:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## 📄 License

This project is licensed under the **MIT License**.

### What This Means

You are free to:
- ✅ Use the code for any purpose (commercial or non-commercial)
- ✅ Modify the code
- ✅ Distribute the code
- ✅ Sublicense the code
- ✅ Use it in proprietary software

The only requirements are:
- Include the original copyright notice
- Include the MIT License text

### MIT License Text

```
MIT License

Copyright (c) 2025 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgements

### Original nanochat Project

This implementation builds upon the excellent **nanochat** project created by **Andrej Karpathy**.

**nanochat** is:
> The best ChatGPT that $100 can buy.

A full-stack implementation of an LLM like ChatGPT in a single, clean, minimal, hackable, dependency-lite codebase.

### New Features in This Fork/Branch

This implementation adds:
- **Mamba Architecture** - State Space Models with linear complexity
- **RAG/REFRAG** - Retrieval-Augmented Generation with multi-hop support
- **Hybrid Models** - Mix Transformer and Mamba blocks
- **Comprehensive Documentation** - 5,000+ lines of guides and tutorials

### Dependencies and Acknowledgements

#### Core Dependencies
- **PyTorch** - Deep learning framework
- **NumPy** - Numerical computing
- **HuggingFace Datasets** - Dataset management

#### Mamba Implementation
- **mamba-ssm** - Official Mamba implementation by Gu & Dao
- **causal-conv1d** - Causal convolution kernels
- **Triton** - GPU kernel optimization

#### RAG Implementation
- **sentence-transformers** - Dense retrieval embeddings
- **FAISS** - Efficient similarity search (Facebook AI)
- **rank-bm25** - BM25 sparse retrieval

### Research Papers

#### Mamba Architecture
```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

#### Retrieval-Augmented Generation
```bibtex
@article{lewis2020retrieval,
  title={Retrieval-augmented generation for knowledge-intensive nlp tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt{\"a}schel, Tim and others},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={9459--9474},
  year={2020}
}
```

## 💡 Attribution Guidelines

### When Using This Code

If you use this implementation in your work, we appreciate if you:

1. **Cite the original nanochat project** (see citation above)
2. **Mention the specific features you used** (e.g., "using the Mamba implementation from nanochat")
3. **Link to the repository** (https://github.com/karpathy/nanochat)

### Example Attribution

In your README or documentation:

```markdown
This project uses [nanochat](https://github.com/karpathy/nanochat) 
by Andrej Karpathy for LLM training, specifically leveraging the 
Mamba architecture and RAG fine-tuning capabilities.
```

In your paper:

```latex
We utilize the nanochat framework \cite{nanochat} with Mamba 
architecture for efficient sequence modeling.
```

## 🤝 Contributing

Contributions to nanochat should maintain:
- **Simplicity** - Clean, minimal code
- **Readability** - Educational quality
- **Hackability** - Easy to modify
- **Documentation** - Well-explained

See the main README.md for contribution guidelines.

## 📧 Contact

For questions about the original nanochat project:
- GitHub: https://github.com/karpathy/nanochat
- See the repository for contact information

For questions about the Mamba/RAG implementation:
- See the documentation in this repository
- Open an issue on GitHub

## 🌟 Support the Project

If you find nanochat useful:
- ⭐ Star the repository on GitHub
- 📖 Cite it in your research
- 🤝 Contribute improvements
- 📣 Share it with others

## 📚 Further Reading

### Original nanochat
- Repository: https://github.com/karpathy/nanochat
- Discussions: https://github.com/karpathy/nanochat/discussions

### This Implementation
- Start: `START_HERE.md`
- Mamba: `QUICKSTART_MAMBA.md`
- RAG: `RAG_QUICKSTART.md`
- Features: `FEATURES.md`

---

**Remember**: This is an MIT License project. You're free to use it, modify it, and build upon it. We just ask that you cite the original work and maintain the license notices. 🙏

**Version**: 1.0.0
**Date**: January 15, 2025

