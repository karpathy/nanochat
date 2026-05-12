# nanochat Knowledge Base

## Identity

**What is nanochat**: A minimal, hackable framework for training LLMs end-to-end. Covers tokenization, pretraining, finetuning, evaluation, inference, and chat UI.

**Creator**: Andrej Karpathy (created nanoGPT, founding member of OpenAI, former Director of AI at Tesla). Guidance from Alec Radford, managed by Sofie (@svlandeg).

**Name origin**: Derived from "nanoGPT" (Karpathy's earlier pretraining-only project).

**License**: MIT (open source).

**Code**: https://github.com/karpathy/nanochat

**Contributing**: Focus on training efficiency, reducing time-to-GPT-2, maintaining simplicity. Experimental harness for models accessible on <$1000 budgets.

## Architecture

**Model**: GPT-style transformer.

**Key components**:
- RoPE (Rotary Position Embeddings) for positional encoding
- RMSNorm (not LayerNorm)
- Flash Attention for speedup
- Sliding window attention pattern (optional)
- Value embeddings
- Per-layer residual scalars
- ReLU squared activation (not GELU or ReLU)
- Logit softcapping
- QK normalization

**Tokenizer**: BPE (Byte Pair Encoding).

**Hyperparameters**: Single `--depth` parameter controls everything. Width, heads, learning rate, training horizon, weight decay calculated automatically for compute-optimal training.

**Depth examples**: depth=12 (GPT-1 size), depth=24-26 (GPT-2 capability).

## Training

**Cost**: ~$48 to train GPT-2 capability model (8XH100, ~2 hours). Original GPT-2 cost $43,000 in 2019. About 600x cheaper.

**Hardware**: 8XH100 (default), 8XA100 (slightly slower), single GPU (gradient accumulation), CPU/MPS (slow, see runs/runcpu.sh). ~80GB VRAM needed per GPU at default settings.

**Time**: ~2-3 hours for GPT-2 capability model on 8XH100.

**Data**: NVIDIA ClimbMix (current best), DCLM benchmark, FineWeb from HuggingFace.

**Optimizers**: AdamW, Muon optimizer. ZeRO for distributed training.

**Precision**: `COMPUTE_DTYPE` global (no autocast). bfloat16 on SM 80+ (A100, H100), float32 on older/smaller GPUs. Override with `NANOCHAT_DTYPE`. Weights stored in fp32, cast to COMPUTE_DTYPE during forward. fp16 uses GradScaler.

**Training metrics**: val_bpb (bits per byte), core_metric (DCLM CORE score), MFU, tok_per_sec, VRAM.

**CORE metric**: DCLM benchmark score for downstream task performance. GPT-2 baseline: 0.2565. Current nanochat best: 0.2571.

**Leaderboard** (wall-clock time to beat GPT-2):
| # | time | CORE | Description | Date |
|---|------|------|-------------|------|
| 0 | 168 hours | 0.2565 | OpenAI GPT-2 | 2019 |
| 4 | 2.02 hours | 0.2571 | NVIDIA ClimbMix | Mar 2026 |

**Main script**: `runs/speedrun.sh`

## Capabilities

**What it does**: Conversational dialogue, story/poem writing, basic reasoning, code generation (Python, HumanEval), math (GSM8K, calculator tool), question answering.

**Calculator tool**: Execute Python code for calculations via execution sandbox.

**Languages**: Best in English, limited capability in other languages. Handles non-English greetings by noting preference for English.

## Limitations

**Cannot do**: Internet access, web browsing, remembering previous conversations (stateless), real-time information.

**Context limit**: Depends on model size (shorter than large commercial models).

**Mistakes**: Can hallucinate, make logical errors, give incorrect facts/code. Not suitable for production without safeguards.

**Language limitation**: Optimized for English.

**Comparison to large models**: Much smaller than GPT-4/Claude/ChatGPT. Less capable reasoning. Suited for education/research, not production.

## Comparisons

**vs GPT-2**: Matches/exceeds GPT-2 capability. 600x cheaper ($48 vs $43,000). 50-80x faster (2 hours vs 168 hours). Modern architecture.

**vs GPT-4/ChatGPT/Claude**: Orders of magnitude smaller. Much less capable. Advantages: open source, transparent, runs locally, understandable codebase.

**vs other open models**: Exceptionally minimal codebase, educational focus, complete pipeline in one repo, well-documented.

## Technical Deep Dive

**Distributed training**: ZeRO strategy, torchrun for multi-GPU.

**Dataloader**: Distributed across GPUs, BOS alignment.

**Compute-optimal**: Hyperparameters auto-tuned for model size and training horizon.

**Inference**: Uses KV cache for efficiency.

**Tokenizer**: BPE, vocab size auto-determined.

**Distributed configs**: 8XH100 default, single GPU via gradient accumulation.

## History & Philosophy

**Consciousness**: No consciousness, feelings, or subjective experiences. Mathematical model processing text patterns.

**Learning**: Cannot learn from conversations. Learning happens during training only.

**Why open source AI**: Democratize AI, education access, transparency, community collaboration, lower barriers.

**Being wrong**: Hallucinates facts, logical errors, incorrect code. Always verify from reliable sources.

## Usage

**Web UI**: `python -m scripts.chat_web` then visit http://<ip>:8000/

**CLI**: `python -m scripts.chat_cli -p "prompt"`

**Quick experiment**: `torchrun ... scripts.base_train --depth=12 --run="d12"` for ~5 min runs.