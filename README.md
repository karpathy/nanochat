# nanochat

![nanochat logo](dev/nanochat.png)

> The best ChatGPT that $100 can buy.

This repo is a full-stack implementation of an LLM like ChatGPT in a single, clean, minimal, hackable, dependency-lite codebase. nanochat is designed to run on a single 8XH100 node via scripts like [speedrun.sh](speedrun.sh), that run the entire pipeline start to end. This includes tokenization, pretraining, finetuning, evaluation, inference, and web serving over a simple UI so that you can talk to your own LLM just like ChatGPT. nanochat will become the capstone project of the course LLM101n being developed by Eureka Labs.

## Quick start

The fastest way to feel the magic is to run the speedrun script [speedrun.sh](speedrun.sh), which trains and inferences the $100 tier of nanochat. On an 8XH100 node at $24/hr, this gives a total run time of about 4 hours. Boot up a new 8XH100 GPU box from your favorite provider (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training script:

```bash
bash speedrun.sh
```

Alternatively, since the script runs for 4 hours, I like to launch it like this inside a new screen session `speedrun` (and also log output to `speedrun.log`):

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

See the [screen cheatsheet](https://gist.github.com/jctosta/af918e1618682638aa82) if you are less familiar. You can watch it go inside the screen session, or detach with `Ctrl-a d` and `tail speedrun.log` to view progress. Now wait 4 hours. Once it's done, you can talk to your LLM via the ChatGPT-like web UI. Make sure again that your local uv virtual environment is active (run `source .venv/bin/activate`), and serve it:

```bash
python -m scripts.chat_web
```

And then visit the URL shown. Make sure to access it correctly, e.g. on Lambda use the public IP of the node you're on, followed by the port, so for example [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/), etc. Then talk to your LLM as you'd normally talk to ChatGPT! Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. The speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener :).

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

You can also `cat report.md` file which appeared in the project directory and contains the "report card" of the run, i.e. a bunch of evaluations and metrics. At the very end, you'll see a summary table, for example:

---

- Characters: 333,989
- Lines: 8,304
- Files: 44
- Tokens (approx): 83,497
- Dependencies (uv.lock lines): 2,004

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2219   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy        | -        | 0.3561   | 0.3876   | -        |
| GSM8K           | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval       | -        | 0.0671   | 0.0854   | -        |
| MMLU            | -        | 0.3111   | 0.3151   | -        |
| ChatCORE        | -        | 0.0730   | 0.0884   | -        |

Total wall clock time: 3h51m

---

(Your table might be missing the RL number by default). For a lot more information around the speedrun script and what to look for and expect, please refer to the walkthrough that I posted in Discussions of the repo: ["Introducing nanochat: The best ChatGPT that $100 can buy"](https://github.com/karpathy/nanochat/discussions/1).

## Bigger models

Unsurprisingly, $100 is not enough to train a highly performant ChatGPT clone. In fact, LLMs are famous for their multi-million dollar capex. For our purposes, I think there are two more scales of interest. First is the ~$300 tier d26 model (i.e. depth=26) that trains in ~12 hours, which slightly outperforms GPT-2 CORE score. Second is the $1000 tier (~41.6 hours), just because it's a nice round number. But both of these are not yet fully supported and therefore not attached here in the master branch yet.

That said, to give a sense, the example changes needed for the [speedrun.sh](speedrun.sh) file to train a GPT-2 grade model d26 only involve three changes:

```bash
...
# you'll need to download more data shards for pretraining
# get the number of parameters, multiply 20 to get tokens, multiply by 4.8 to get chars,
# divide by 250 million to get number of shards. todo need to improve this...
python -m nanochat.dataset -n 450 &
...
# use --depth to increase model size. to not oom, halve device batch size 32 -> 16:
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16
...
# make sure to use the same later during midtraining:
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
```

That's it! The biggest thing to pay attention to is making sure you have enough data shards to train on (the code will loop and do more epochs over the same training set otherwise, decreasing learning speed a bit), and managing your memory/VRAM, primarily by decreasing the `device_batch_size` until things fit (the scripts automatically compensates by increasing the number of gradient accumulation loops, simply turning parallel compute to sequential compute).

And a bit more about computing environments that will run nanochat:

- The code will run just fine on the Ampere 8XA100 GPU node as well, but a bit slower.
- All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait 8 times longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device_batch_size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't implemented this out of the box so it might take a bit of tinkering.

## Questions

nanochat is designed to be short and sweet. One big advantage of this is that we can package up all of the files together and copy paste them to your favorite LLM to ask arbitrary questions. As an example, I like to package up the repo using the [files-to-prompt](https://github.com/simonw/files-to-prompt) utility like so:

```bash
files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml > packaged.txt
```

This includes all py, rs, html, toml, sh files, excludes the `rustbpe/target` folder, and chooses the cxml output format. Everything is written to the `packaged.txt` file, which atm measures ~330KB (i.e. well below ~100K tokens for a state of the art LLM), and ~8K lines of code in 45 files.

Alternatively, I recommend using [DeepWiki](https://deepwiki.com/) from Devin/Cognition to ask questions of this repo. In the URL of this repo, simply change github.com to deepwiki.com, and you're off.

## Tests

I haven't invested too much here but some tests exist, especially for the tokenizer. Run e.g. as:

```bash
python -m pytest tests/test_rustbpe.py -v -s
```

## For Students

nanochat is designed as an educational full-stack LLM implementation. If you're learning about how modern language models work from tokenization to deployment, this section will guide you through the codebase systematically.

### Learning Path

The best way to understand nanochat is to follow the same order as the training pipeline. Here's the recommended reading sequence:

#### **Phase 1: Foundations (Start Here)**

1. **`nanochat/common.py`** - Common utilities, distributed setup, logging
   - *What to learn*: How distributed training is initialized, basic helper functions
   - *Key concepts*: DDP (Distributed Data Parallel), device management, logging patterns

2. **`nanochat/tokenizer.py`** - Text tokenization and the BPE algorithm
   - *What to learn*: How text becomes numbers that neural networks can process
   - *Key concepts*: Byte Pair Encoding (BPE), vocabulary, special tokens
   - *Related*: `rustbpe/src/lib.rs` (Rust implementation for speed)

3. **`scripts/tok_train.py`** - Tokenizer training script
   - *What to learn*: How to train a tokenizer from scratch on your dataset
   - *Try it*: Run `python -m scripts.tok_train --max_chars=2000000000` (after downloading data)

#### **Phase 2: Model Architecture**

4. **`nanochat/gpt.py`** ⭐ **CORE FILE**
   - *What to learn*: The Transformer architecture with modern improvements
   - *Key concepts*:
     - Rotary embeddings (RoPE) for positional encoding
     - QK normalization for training stability
     - Multi-Query Attention (MQA) for efficient inference
     - ReLU² activation function
     - RMSNorm (no learnable parameters)
   - *Architecture highlights*:
     - `CausalSelfAttention`: The attention mechanism
     - `MLP`: Feed-forward network with ReLU² activation
     - `Block`: One transformer layer (attention + MLP)
     - `GPT`: The full model putting it all together

5. **`nanochat/muon.py`** and **`nanochat/adamw.py`** - Optimizers
   - *What to learn*: How different parameters need different optimization strategies
   - *Key insight*: Muon optimizer for matrix parameters, AdamW for embeddings
   - *Why dual optimizers?*: Different parameter types benefit from different update rules

#### **Phase 3: Data & Training**

6. **`nanochat/dataset.py`** - Dataset downloading and preparation
   - *What to learn*: How to download and manage large training datasets (FineWeb)
   - *Key concepts*: Data sharding, streaming, efficient storage

7. **`nanochat/dataloader.py`** - Data loading during training
   - *What to learn*: How to efficiently feed data to the model during training
   - *Key concepts*: Tokenization on-the-fly, distributed data loading, batching

8. **`scripts/base_train.py`** ⭐ **CORE FILE**
   - *What to learn*: The complete pretraining loop
   - *Key concepts*:
     - Gradient accumulation for large batch sizes
     - Mixed precision training (bfloat16)
     - Learning rate schedules
     - Checkpointing
     - Distributed training coordination
   - *Try it*: Read through the main training loop starting from `for step in range(num_iterations + 1):`

#### **Phase 4: Evaluation**

9. **`nanochat/loss_eval.py`** - Training/validation loss evaluation
   - *What to learn*: How to measure model perplexity on held-out data
   - *Key concepts*: Bits per byte (BPB), perplexity

10. **`nanochat/core_eval.py`** - CORE benchmark evaluation
    - *What to learn*: How to evaluate language modeling capability
    - *Key concepts*: Next-token prediction accuracy as a metric

11. **`tasks/*.py`** - Task-specific evaluations
    - `tasks/arc.py` - Reasoning benchmark
    - `tasks/gsm8k.py` - Math word problems
    - `tasks/humaneval.py` - Code generation
    - `tasks/mmlu.py` - General knowledge
    - `tasks/smoltalk.py` - Conversational ability
    - *What to learn*: How to evaluate LLMs on different capabilities

#### **Phase 5: Inference & Serving**

12. **`nanochat/engine.py`** ⭐ **CORE FILE**
    - *What to learn*: Efficient text generation with KV caching
    - *Key concepts*:
      - KV cache for fast autoregressive generation
      - Sampling strategies (temperature, top-k)
      - Tool use (calculator integration)
      - Batch generation
    - *Cool feature*: The calculator tool demonstrates how LLMs can use tools during generation

13. **`scripts/chat_cli.py`** - Command-line chat interface
    - *What to learn*: How to build a simple chat interface
    - *Try it*: `python -m scripts.chat_cli -p "Why is the sky blue?"`

14. **`scripts/chat_web.py`** - Web-based chat interface
    - *What to learn*: How to serve an LLM over HTTP
    - *Try it*: `python -m scripts.chat_web` (after training)

#### **Phase 6: Advanced Training**

15. **`scripts/mid_train.py`** - Midtraining
    - *What to learn*: Teaching the model special tokens and conversational format
    - *Key insight*: Bridge between pretraining and task-specific finetuning

16. **`scripts/chat_sft.py`** - Supervised Fine-Tuning
    - *What to learn*: Adapting the model to follow instructions
    - *Key concepts*: Instruction tuning, chat templates

17. **`scripts/chat_rl.py`** - Reinforcement Learning
    - *What to learn*: Using RL to improve specific capabilities (math)
    - *Key concepts*: Reward models, policy optimization

#### **Phase 7: Infrastructure**

18. **`nanochat/checkpoint_manager.py`** - Model checkpointing
    - *What to learn*: How to save and load model weights efficiently

19. **`nanochat/report.py`** - Automated reporting
    - *What to learn*: How to track experiments and generate reports

20. **`nanochat/configurator.py`** - Configuration management
    - *What to learn*: Command-line argument parsing for ML experiments

### Key Architectural Decisions & Why

1. **Rotary Embeddings instead of learned positional embeddings**
   - *Why?*: Better length generalization, no extra parameters
   - *Where?*: `gpt.py` - see the `apply_rotary_emb()` function

2. **Untied embeddings** (separate input and output embedding matrices)
   - *Why?*: More expressive, worth the extra parameters
   - *Where?*: `gpt.py` - `GPT` class has separate `wte` and `lm_head` parameters

3. **QK Normalization**
   - *Why?*: Training stability, prevents attention logits from exploding
   - *Where?*: `gpt.py` - in `CausalSelfAttention.forward()` after rotary embeddings

4. **Multi-Query Attention (MQA)**
   - *Why?*: Faster inference with minimal quality loss
   - *Where?*: `gpt.py` - `GPTConfig` has separate `n_head` and `n_kv_head`, see `repeat_kv()` function

5. **ReLU² activation**
   - *Why?*: Better than GELU for smaller models, simple and effective
   - *Where?*: `gpt.py` - `MLP.forward()` uses `F.relu(x).square()`

6. **Dual optimizer strategy** (Muon + AdamW)
   - *Why?*: Matrix parameters and embeddings benefit from different optimization
   - *Where?*: `gpt.py` - see `GPT.setup_optimizers()` method

7. **Logit soft-capping**
   - *Why?*: Prevents extreme logit values, improves training stability
   - *Where?*: `gpt.py` - in `GPT.forward()`, search for "softcap"

### The Complete Pipeline Visualized

```
1. Data Preparation
   ├─ Download FineWeb shards (dataset.py)
   ├─ Train BPE tokenizer (tok_train.py)
   └─ Tokenize data on-the-fly (dataloader.py)
   
2. Pretraining
   ├─ Initialize model (gpt.py)
   ├─ Setup optimizers (muon.py, adamw.py)
   ├─ Train on tokens (base_train.py)
   └─ Evaluate on CORE (base_eval.py)
   
3. Midtraining
   ├─ Load base checkpoint
   ├─ Train on formatted data (mid_train.py)
   └─ Evaluate on chat tasks (chat_eval.py)
   
4. Fine-tuning
   ├─ Supervised learning (chat_sft.py)
   ├─ [Optional] RL training (chat_rl.py)
   └─ Final evaluation (chat_eval.py)
   
5. Deployment
   ├─ Load best checkpoint
   ├─ Serve via CLI (chat_cli.py)
   └─ Serve via Web (chat_web.py)
```

### Concepts to Master

As you read through the code, make sure you understand these fundamental concepts:

**Tokenization:**
- Why we need tokenization
- How BPE works (greedy merge of most frequent pairs)
- Special tokens and their purpose

**Model Architecture:**
- Self-attention mechanism (Q, K, V matrices)
- Causal masking (can only attend to past tokens)
- Residual connections (x + attention(x))
- Layer normalization (RMSNorm variant)
- Why we stack many layers

**Training:**
- Gradient descent and backpropagation
- Loss function (cross-entropy for next token prediction)
- Learning rate schedules (warmup + cosine decay)
- Gradient accumulation (simulating larger batches)
- Mixed precision training (bfloat16 for speed)

**Distributed Training:**
- Data parallelism (same model, different data shards)
- Gradient synchronization across GPUs
- All-reduce operations

**Inference:**
- Autoregressive generation (one token at a time)
- KV caching (reuse past computations)
- Sampling strategies (temperature, top-k)

### Recommended Experiments

Once you've read through the code, try these experiments to deepen understanding:

1. **Modify the tokenizer vocabulary size** - See how it affects compression and training
2. **Change model depth** - Train a smaller/larger model, observe parameter count vs. performance
3. **Experiment with batch sizes** - Understand the speed/memory tradeoff
4. **Try different sampling temperatures** - See how it affects generation creativity
5. **Implement a simple evaluation task** - Add your own benchmark in `tasks/`
6. **Add a new tool** - Extend the calculator to support more operations

### Quick Start for Learning

If you just want to understand the core without running anything:

1. Read `gpt.py` - Understand the Transformer architecture
2. Read `engine.py` - Understand how generation works
3. Read `base_train.py` - Understand the training loop

These three files (~1000 lines total) contain the essence of how modern LLMs work.

### Resources for Deeper Learning

- **Attention paper**: "Attention Is All You Need" (Vaswani et al.)
- **GPT-2 paper**: "Language Models are Unsupervised Multitask Learners"
- **Rotary embeddings**: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- **Andrej's videos**: Neural Networks: Zero to Hero series on YouTube
- **LLM101n course**: The course this project was built for (when released)

## Contributing

nanochat is nowhere finished. The goal is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there will be no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a concrete ChatGPT clone and its report card.

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
