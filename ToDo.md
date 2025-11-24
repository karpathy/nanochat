# Remaining Tasks & Roadmap

## 🚀 Optimization & Strix Halo Specifics
- [ ] **MXFP4 Investigation**: Research and implement OCP Microscaling (MXFP4) support for inference using AMD Quark, once the ecosystem matures for APUs.
- [ ] **System Tuner Expansion**: Enhance `scripts/tune_system.py` to auto-tune:
    - Learning rates and schedules.
    - Optimizer hyperparameters (momentum, weight decay).
    - Compilation flags (`torch.compile` modes).
- [ ] **Torch Compile Dynamics**: Investigate `dynamic=True` vs `False` in `scripts/base_train.py` for variable sequence lengths on RDNA 3.5.
- [ ] **Distributed Tuning**: Benchmark RCCL vs Gloo backends specifically for APU-based distributed setups (if scaling to multi-node APUs).

## 🛠 Codebase Maintenance & Tech Debt
- [ ] **DDP Detection**: Refactor `is_ddp()` in `nanochat/common.py` to use a more robust detection method.
- [ ] **Tokenizer Efficiency**: Optimize `prepend_id` insertion in `nanochat/tokenizer.py` (currently uses `list.insert(0)`, which is O(N)).
- [ ] **Liger Kernels**: Experiment with [Liger Kernels](https://github.com/linkedin/Liger-Kernel) or chunked cross-entropy in `nanochat/gpt.py` to reduce memory usage.
- [ ] **Checkpointing**:
    - Fix potentially redundant model re-initialization in `checkpoint_manager.py`.
    - Ensure optimizer state saving across ranks is robust (`scripts/base_train.py`).
- [ ] **Evaluation Cleanup**: Refactor `scripts/base_eval.py` to remove heavy dependencies (like pandas) and simplify file handling.
- [ ] **AdamW Warmup**: Experiment with short warmup periods for AdamW parameters (`scripts/base_train.py` TODO).

## ✨ New Features
- [ ] **Model Export**:
    - Add a script to export checkpoints to **GGUF** format for efficient inference on Strix Halo NPU (via llama.cpp).
    - Add HuggingFace `safetensors` export support.
- [ ] **Inference Server**: Create a production-ready API server (FastAPI) to serve the model, replacing the simple `chat_cli.py`.
- [ ] **RLHF Expansion**: Extend Reinforcement Learning (RL) support beyond the current GSM8K-only implementation.
- [ ] **Advanced UI**: Develop a more robust chat interface (React/Web) or integrate with existing open-source UIs (e.g., Open WebUI).
- [ ] **Data Pipeline**:
    - Add data integrity verification for downloaded shards.
    - Optimize data loading for APU unified memory architectures.
