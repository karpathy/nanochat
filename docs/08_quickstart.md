# Quick Start Guide

This guide will get you up and running with nanochat - from installation to training your first model and using the web interface.

## Table of Contents
1. [Installation](#installation)
2. [Quick Test: Pre-trained Model](#quick-test-pre-trained-model)
3. [Training Your First Model](#training-your-first-model)
4. [Using the Web Interface](#using-the-web-interface)
5. [Command-Line Chat](#command-line-chat)
6. [Customizing Training](#customizing-training)
7. [Common Issues and Solutions](#common-issues-and-solutions)

---

## Installation

### Requirements

**Hardware:**
- For training: NVIDIA GPU with 16GB+ VRAM (H100, A100, or RTX 4090)
- For inference: Any GPU, or even CPU (but slower)

**Software:**
- Python 3.10+
- CUDA 12+ (for GPU training)
- ~50GB disk space (for datasets and checkpoints)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/nanochat.git
cd nanochat
```

### Step 2: Install Dependencies

**Using uv (recommended):**
```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install deps
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

**Using pip:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Step 3: Verify Installation

```bash
# Test import
python -c "import nanochat; print('nanochat installed successfully!')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Expected output:**
```
nanochat installed successfully!
CUDA available: True
```

---

## Quick Test: Pre-trained Model

If you want to test the system before training, you can use a pre-trained tokenizer and model (if available).

### Option 1: Download Pre-trained Model

```bash
# Download from release (if available)
wget https://github.com/.../nanochat-d20.tar.gz
tar -xzf nanochat-d20.tar.gz -C out/
```

### Option 2: Use OpenAI's GPT-2

For quick testing without training:

```bash
# This will use tiktoken's GPT-2 tokenizer
python -c "
from nanochat.tokenizer import RustBPETokenizer
tokenizer = RustBPETokenizer.from_pretrained('gpt2')
print(tokenizer.encode('Hello, world!'))
"
```

---

## Training Your First Model

### The Easy Way: speedrun.sh

The fastest way to train a complete model:

```bash
# Train a ~$100 model in 4 hours
bash speedrun.sh
```

**What this does:**
1. Downloads and prepares training data
2. Trains a tokenizer
3. Pretrains a base model (d20, 561M parameters)
4. Fine-tunes for chat
5. Evaluates on benchmarks
6. Launches web UI

**Expected timeline:**
- Setup: 15 minutes (data download)
- Tokenizer training: 5 minutes
- Base pretraining: 3-4 hours
- Fine-tuning: 30 minutes
- Evaluation: 30 minutes
- **Total: ~5 hours**

### Step-by-Step Training

If you want more control, run each stage manually:

#### 1. Download Training Data

```bash
# Download FineWeb dataset (~100GB)
python -m scripts.download_data \
    --num_shards=100 \
    --output_dir=out/data
```

#### 2. Train Tokenizer

```bash
# Train BPE tokenizer on 2B characters
python -m scripts.tok_train \
    --vocab_size=65536 \
    --num_chars=2000000000 \
    --output_dir=out/tokenizer
```

**Output:**
```
Processing sequences from iterator...
Processed 50000000 sequences, 12000 unique chunks
Starting BPE training: 65280 merges to compute
Progress: 50% (32640/65280 merges)
Progress: 100% (65280/65280 merges)
Saved tokenizer to out/tokenizer/tokenizer.pkl
```

#### 3. Pretrain Base Model

```bash
# Train on 1 GPU
python -m scripts.base_train \
    --run=my_first_model \
    --depth=20 \
    --num_iterations=5400

# Train on 8 GPUs (distributed)
torchrun --nproc_per_node=8 -m scripts.base_train \
    --run=my_first_model \
    --depth=20 \
    --num_iterations=5400
```

**Monitor training:**
```
Step 00000 | Validation bpb: 2.1234
Step 00250 | Validation bpb: 1.4567
Step 00500 | Validation bpb: 1.3210
...
Step 05400 | Validation bpb: 1.2145
Step 05400 | CORE metric: 0.4567
```

**Checkpoints saved to:** `out/base_checkpoints/d20/`

#### 4. Fine-tune for Chat

```bash
# Supervised fine-tuning
python -m scripts.chat_sft \
    --base_checkpoint=out/base_checkpoints/d20 \
    --num_iterations=2000

# Optional: Reinforcement learning
python -m scripts.chat_rl \
    --sft_checkpoint=out/chat_checkpoints/d20_sft \
    --num_iterations=1000
```

**Checkpoints saved to:** `out/chat_checkpoints/d20_sft/`

#### 5. Evaluate

```bash
# Evaluate chat model
python -m scripts.chat_eval \
    --checkpoint_dir=out/chat_checkpoints/d20_sft \
    --max_per_task=100
```

**Output:**
```
Task: arc_challenge     Accuracy: 0.45
Task: gsm8k             Accuracy: 0.38
Task: mmlu              Accuracy: 0.32
Task: humaneval         Pass@1: 0.08
Overall: 0.41
```

---

## Using the Web Interface

### Launch the Server

```bash
# Start web server
python -m scripts.chat_web \
    --checkpoint_dir=out/chat_checkpoints/d20_sft \
    --port=8000
```

**Output:**
```
Loading model from out/chat_checkpoints/d20_sft...
Model loaded successfully!
Starting server on http://localhost:8000
```

### Access the UI

1. Open browser to `http://localhost:8000`
2. You'll see a ChatGPT-like interface
3. Type a message and press Enter!

**Example conversation:**
```
You: What is 2+2?
Assistant: Let me calculate that. <|python_start|>2+2<|python_end|><|output_start|>4<|output_end|> The answer is 4.

You: Tell me a fun fact about space.
Assistant: Did you know that a day on Venus is longer than a year on Venus? Venus takes about 243 Earth days to rotate once on its axis, but only about 225 Earth days to orbit the Sun!

You: Thanks!
Assistant: You're welcome! Let me know if you have any other questions.
```

### Configuration Options

```bash
# Custom port
python -m scripts.chat_web --port=5000

# Different temperature
python -m scripts.chat_web --temperature=1.2

# Max tokens per response
python -m scripts.chat_web --max_tokens=500
```

---

## Command-Line Chat

For a simpler interface without a web browser:

```bash
# Interactive CLI chat
python -m scripts.chat_cli \
    --checkpoint_dir=out/chat_checkpoints/d20_sft
```

**Usage:**
```
Model loaded successfully!

User: What is the capital of France?
Assistant: The capital of France is Paris.

User: What about Germany?
Assistant: The capital of Germany is Berlin.

User: exit
Goodbye!
```

**Options:**
```bash
# Higher temperature (more creative)
python -m scripts.chat_cli --temperature=1.2

# Lower temperature (more focused)
python -m scripts.chat_cli --temperature=0.5

# Generate multiple samples
python -m scripts.chat_cli --num_samples=3
```

---

## Customizing Training

### Training a Smaller Model (for testing)

```bash
# Tiny model for quick testing
python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=512 \
    --device_batch_size=1 \
    --total_batch_size=512 \
    --num_iterations=100
```

**Use case:** Test your setup without waiting hours

### Training a Larger Model

```bash
# d32 (1.9B parameters) for better quality
bash run1000.sh  # ~$800, 41.6 hours
```

Or manually:
```bash
torchrun --nproc_per_node=8 -m scripts.base_train \
    --depth=32 \
    --num_iterations=10000 \
    --total_batch_size=524288
```

### Custom Hyperparameters

```bash
# Change learning rates
python -m scripts.base_train \
    --matrix_lr=0.01 \
    --embedding_lr=0.1 \
    --unembedding_lr=0.002

# Change batch size
python -m scripts.base_train \
    --total_batch_size=262144 \
    --device_batch_size=16

# Change context length
python -m scripts.base_train \
    --max_seq_len=4096  # Longer context
```

### Adding Custom Training Data

**For fine-tuning:**

1. Create JSONL file with conversations:
```json
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm doing well, thank you!"}]}
```

2. Point training script to your data:
```python
# In scripts/chat_sft.py, modify data loading:
data_loader = load_custom_data("path/to/your/data.jsonl")
```

3. Run fine-tuning:
```bash
python -m scripts.chat_sft \
    --base_checkpoint=out/base_checkpoints/d20 \
    --custom_data=path/to/your/data.jsonl
```

### Generating Synthetic Data

Use the generator to create custom examples:

```bash
# Generate synthetic conversations
python dev/gen_synthetic_data.py \
    --num_examples=1000 \
    --output=out/synthetic_data.jsonl
```

Edit `dev/gen_synthetic_data.py` to customize the personality/style.

---

## Common Issues and Solutions

### Issue: CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

1. **Reduce batch size:**
```bash
python -m scripts.base_train --device_batch_size=16  # Instead of 32
```

2. **Use gradient accumulation:**
```bash
# Maintain same total batch size with less memory
python -m scripts.base_train \
    --device_batch_size=8 \
    --total_batch_size=524288  # Will use more accumulation steps
```

3. **Train smaller model:**
```bash
python -m scripts.base_train --depth=12  # Instead of 20
```

4. **Reduce context length:**
```bash
python -m scripts.base_train --max_seq_len=1024  # Instead of 2048
```

### Issue: Tokenizer Not Found

**Error:**
```
FileNotFoundError: out/tokenizer/tokenizer.pkl not found
```

**Solution:**
Train tokenizer first:
```bash
python -m scripts.tok_train --vocab_size=65536
```

Or point to existing tokenizer:
```bash
export NANOCHAT_BASE_DIR=/path/to/your/tokenizer/parent/dir
python -m scripts.base_train
```

### Issue: Data Download Fails

**Error:**
```
ConnectionError: Failed to download dataset
```

**Solutions:**

1. **Check internet connection**

2. **Retry with fewer shards:**
```bash
python -m scripts.download_data --num_shards=10  # Instead of 100
```

3. **Use local data:**
Place text files in `out/data/` and modify data loading in scripts.

### Issue: Training is Very Slow

**Symptoms:** < 100 tokens/second

**Solutions:**

1. **Use torch.compile:**
Already enabled by default in `base_train.py`

2. **Check GPU utilization:**
```bash
nvidia-smi -l 1  # Monitor GPU usage
```

If GPU usage is low, increase batch size.

3. **Use more GPUs:**
```bash
torchrun --nproc_per_node=8 -m scripts.base_train  # Instead of 1
```

4. **Check data loading:**
If you see "waiting for data", increase data loader workers.

### Issue: Model Generates Gibberish

**Symptoms:** Model outputs random characters or repeats endlessly

**Possible causes:**

1. **Training not converged:** Train longer or check if loss is decreasing

2. **Wrong checkpoint:** Make sure you're loading the right checkpoint

3. **Tokenizer mismatch:** Use the same tokenizer for training and inference

4. **Temperature too high:** Lower it:
```bash
python -m scripts.chat_cli --temperature=0.7
```

### Issue: Model Won't Stop Generating

**Symptoms:** Model generates thousands of tokens without stopping

**Solutions:**

1. **Set max_tokens:**
```bash
python -m scripts.chat_cli --max_tokens=100
```

2. **Check for `<|assistant_end|>` in training data**

3. **Fine-tune with proper conversation format**

### Issue: Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'nanochat'
```

**Solution:**

1. **Install in editable mode:**
```bash
pip install -e .
```

2. **Activate virtual environment:**
```bash
source venv/bin/activate  # or .venv/bin/activate
```

3. **Run as module:**
```bash
python -m scripts.base_train  # Instead of python scripts/base_train.py
```

---

## Next Steps

Now that you have a working setup:

1. **Experiment with different model sizes** (depth=12, 20, 26, 32)
2. **Try different hyperparameters** (learning rates, batch sizes)
3. **Create custom training data** for your use case
4. **Evaluate on different benchmarks** to understand capabilities
5. **Modify the architecture** in `gpt.py` (if you're adventurous!)

### Learning Resources

- **Read the other documentation**: Start with `docs/01_introduction.md`
- **Explore the code**: `nanochat/gpt.py` is only ~300 lines!
- **Check the examples**: `dev/` directory has useful scripts
- **Read the research**: Papers on transformers, BPE, RLHF

### Getting Help

- **Check existing issues**: https://github.com/nanochat/issues
- **Read error messages carefully**: They often tell you exactly what's wrong
- **Enable verbose logging**: Add `--verbose` to commands
- **Try smaller experiments first**: Debug on tiny models before scaling up

---

## Summary

**Quick commands reference:**

```bash
# Install
uv sync

# Train everything (easy mode)
bash speedrun.sh

# Train step-by-step
python -m scripts.tok_train
python -m scripts.base_train
python -m scripts.chat_sft

# Use the model
python -m scripts.chat_web  # Web interface
python -m scripts.chat_cli  # Command-line

# Evaluate
python -m scripts.chat_eval
```

**Key files:**
- `speedrun.sh` - Complete training pipeline
- `scripts/base_train.py` - Pretraining
- `scripts/chat_sft.py` - Fine-tuning
- `scripts/chat_web.py` - Web interface
- `nanochat/gpt.py` - Model architecture

**Good luck with your LLM training journey!** 🚀

---

## Congratulations!

You've completed the nanochat documentation. You now understand:
- ✅ What LLMs are and how they work
- ✅ Tokenization with BPE
- ✅ Transformer architecture
- ✅ Training pipeline (pretraining → SFT → RL)
- ✅ Inference and generation
- ✅ Tool integration
- ✅ Evaluation metrics
- ✅ How to train and use models

**You're ready to build your own ChatGPT-like model from scratch!**