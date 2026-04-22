# samosaChaat — Training Roadmap v2

**Purpose**: a self-contained plan to take the model from its current state (d24-sft-r6, 97% probe pass, noticeable rough edges) to something that genuinely feels *smart* and *alive*.

**Author**: Manmohan Sharma. **Model**: `nanochat-d24` / samosaChaat / 1.38 B params / 16 K context.

Read this top to bottom when you next allocate GPU time. Everything you need — infrastructure, credentials, datasets, commands, evaluation gates — is in here.

---

## 0. Read me first

If you just got 8× H100s allocated and want to ship a better model today, your order of operations is:

1. SSH in, sync the repo, pull weights from HF (§3 below).
2. Run **Phase A** (joint Think+Tool SFT) — 2 hours, biggest single-round win.
3. Evaluate. If you have more time, run **Phase B** (expanded reasoning SFT) — another 2 hours.
4. If you have a full day and ~$300 budget, run **Phase C** (extended pretraining) — 12-18 hours.
5. **Phase D** (DPO) polishes tone and removes lingering HTML/format artifacts — 3 hours.
6. **Phase E** (scale to d32) is only worth doing after A–D have diminishing returns.

---

## 1. Current state (April 2026)

### Model
- **Production checkpoint**: `chatsft_checkpoints/d24-sft-r6/model_000754.pt` on HF, val_bpb **0.2635**, 32/33 on probe suite.
- **Base pretrain**: `base_checkpoints/d24/model_005568.pt`, 5.84 B tokens on ClimbMix, val_bpb 0.72.
- **Continued pretrain**: `base_checkpoints/d24-cpt/model_010000.pt`, val_bpb 0.365, 2 K context.
- **16 K extension**: `base_checkpoints/d24-cpt-16k/model_001200.pt`, val_bpb 0.526.

### What works
- Persona / identity / Manmohan attribution: 100%
- Tool use (with classifier or force toggle): 100%
- India / domain knowledge: 100%
- Basic math, chat format, creative format: 100%

### What doesn't
| Bug | Root cause |
|---|---|
| Factual hallucination (GDP, prices, random names) | Base pretrain is 5× under Chinchilla-optimal (5.84 B vs ~28 B for 1.38 B params) |
| Multi-step arithmetic / day-of-week | Only ~3.5 k reasoning SFT rows; industry runs 100 k+ |
| Can't chain `<think>` + `<|python_start|>` | Training data had them as disjoint patterns — never together |
| `<b>` / `<i>` / `Answer:` / `![placeholder]` leaks | Noisy UltraChat/WildChat rows weren't filtered hard enough |
| Multi-turn follow-ups ("tell me more about him") | Thin multi-turn coverage in SFT |
| Model loops after `<|output_end|>` | Training tool-use examples didn't always terminate with `<|assistant_end|>` |
| Creative tasks (haiku, jokes) mediocre | ~224 creative examples, way too little |

---

## 2. Infrastructure recap

Everything lives in three places: **HuggingFace** (weights + data + docs), **GitHub** (code + CI/CD), **Modal** (inference). Training runs on rented GPUs (Prime Intellect / Hyperbolic).

### Credentials (all still valid — rotate at your discretion)

```bash
# HuggingFace
HF_TOKEN       = hf_<WRITE_TOKEN>            # read
HF_WRITE_TOKEN = hf_<WRITE_TOKEN>             # write

# Search / LLM APIs (for data generation)
TAVILY_API_KEY      = tvly-<YOUR_TAVILY_KEY>
OPENAI_API_KEY      = sk-proj-<REDACTED>
ANTHROPIC_API_KEY   = sk-ant-api03-<REDACTED>

# Modal — use ~/.modal.toml on a machine authed as manmohan659
# token_id: ak-<YOUR_MODAL_TOKEN_ID>  (secret in ~/.modal.toml)
```

### Machines

| Where | What | How to reach |
|---|---|---|
| 8× H100 (Prime Intellect) | Training GPU | `ssh -i ~/.ssh/gpu_servers ubuntu@<IP>` — IP rotates, set when spinning up |
| Modal (manmohan659) | Production inference, L4 GPU | App `samosachaat-inference`. Deploy: `modal deploy modal/serve.py` |
| EC2 `52.10.243.118` (AWS us-west-2) | Production frontend + chat-api + auth + nginx | `ssh -i ~/Documents/FinalSemester/DevOps/manmohan.pem ubuntu@52.10.243.118` |

### Repos

| Repo | Contents |
|---|---|
| `ManmohanSharma/nanochat-d24` (HF model) | All `base_checkpoints/*`, `chatsft_checkpoints/*`, `tokenizer/`, `scripts/training_pipeline/`, `datasets/`, `evals/`, `README.md`, `TRAINING_REPORT.md` |
| `ManmohanSharma/nanochat-d24-training-data` (HF dataset) | 40 immutable parquet shards, 18 GB — the original base pretrain + CPT corpus. **Do not re-shard.** |
| `github.com/manmohan659/nanochat` | All training code, modal serving, frontend, chat-api, CI/CD |

### Cold-start a new 8× H100 box

```bash
# On the fresh box, get python tooling
sudo apt-get update -qq && sudo apt-get install -y python3-pip python3-dev
pip3 install --user torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip3 install --user tiktoken tokenizers huggingface_hub wandb rustbpe psutil \
  tabulate kernels torchao einops regex matplotlib zstandard pandas transformers datasets openai modal

# Clone
git clone https://github.com/manmohan659/nanochat.git ~/work/nanochat
cd ~/work/nanochat

# Pull training pipeline scripts (they live in the HF repo, not git)
python3 -c "
import os
from huggingface_hub import hf_hub_download
tok = 'hf_<WRITE_TOKEN>'
for f in ['scripts/base_cpt.py', 'scripts/training_pipeline/resume_from_hf.py',
          'scripts/training_pipeline/hf_push_worker.py',
          'scripts/training_pipeline/eval_suite_v2.py',
          'scripts/training_pipeline/launch_cpt.sh']:
    p = hf_hub_download('ManmohanSharma/nanochat-d24', f, token=tok)
    dest = os.path.join(os.path.expanduser('~/work/nanochat'), f)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.abspath(p) != os.path.abspath(dest):
        import shutil; shutil.copy2(p, dest)
    print(f'pulled {f}')
"

# Stash API keys
cat > ~/.api_keys <<'EOF'
export HF_TOKEN='hf_<WRITE_TOKEN>'
export HF_WRITE_TOKEN='hf_<WRITE_TOKEN>'
export TAVILY_API_KEY='tvly-<YOUR_TAVILY_KEY>'
export OPENAI_API_KEY='sk-proj-...'
export ANTHROPIC_API_KEY='sk-ant-api03-...'
EOF
chmod 600 ~/.api_keys
echo '[ -f ~/.api_keys ] && source ~/.api_keys' >> ~/.bashrc

# Pull training data
python3 ~/work/nanochat/scripts/training_pipeline/resume_from_hf.py
# This fetches: 40 parquet shards + latest checkpoint + tokenizer
```

---

## 3. The plan

Six phases, ordered by impact-per-cost. Each phase is independently shippable: you can stop after any of them and still have a better model than today.

### Phase A — Joint Think + Tool SFT ⏱️ 2 hours ~$15

**Goal**: fix the #1 visible bug: the model picks *either* `<think>` *or* `<|python_start|>` but never both. Fixes temporal reasoning on current-event queries too, because the model learns to think *about* whether to search.

**Data generation** — synthesize 3,000 conversations via gpt-4o-mini:

```python
# ~/work/scripts/gen_joint_think_tool.py
# Prompt the teacher to emit strict format:
#   <think>brief reasoning about whether tool is needed</think>
#   <|python_start|>{"tool":"web_search","arguments":{...}}<|python_end|>
#   <|output_start|>{plausible Tavily result}<|output_end|>
#   {final grounded answer}
#
# Three sub-patterns (1000 each):
#  A. think → web_search → answer (time-sensitive facts)
#  B. think → calculator → answer (arithmetic/finance)
#  C. think → direct answer (no tool needed — think still closes cleanly)
```

Topic banks to vary:
- Current events: elections, sports, weather, CEOs, prices, news
- Math: tips, CAGR, compound interest, basic algebra
- Mixed: "is X true today?" where the model decides to search or not

**Critical invariants for every conv:**
- `<think>` opens and closes with `</think>` — answer never inside
- tool call + result appear only after `</think>`
- conv terminates cleanly (`<|assistant_end|>` added at tokenization time by `chat_sft.py`)

**Filter**: reject any sample with answer-inside-think, missing close-tag, or more than one `<|output_start|>`.

**SFT launch** (continues from r6):
```bash
# First, move r6 into base_checkpoints so chat_sft can load it
cp -r ~/.cache/nanochat/chatsft_checkpoints/d24-sft-r6 \
      ~/.cache/nanochat/base_checkpoints/d24-sft-r7-init

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
  --run=dummy --model-tag=d24-sft-r7-init --model-step=754 \
  --load-optimizer=0 --max-seq-len=4096 --device-batch-size=4 \
  --total-batch-size=524288 --init-lr-frac=0.2 --warmdown-ratio=0.5 \
  --eval-every=50 --mmlu-epochs=0 --gsm8k-epochs=0 \
  --extra-train-jsonl=~/work/sft_data/r7_joint_train.jsonl \
  --extra-val-jsonl=~/work/sft_data/r7_joint_val.jsonl
```

**Mix**:
- joint_think_tool × 6 (18 k rows — the fix)
- reasoning_v2_clean × 2 (keep existing think behavior)
- tool_use × 2 (keep direct-tool behavior)
- creator × 20 (identity retention)
- identity × 3 (identity retention)
- desserts × 2 (domain retention)
- small quality sample (~3 k) for chat breadth

**Eval gate**: probe suite must stay at 95%+ AND these new probes must pass:
- "what's the weather in Seoul" in Think mode → calls web_search, answer outside `</think>`
- "calculate a 17% tip on $45 in think mode" → thinks, calls calculator, gives $7.65
- "how do airplanes fly" in Think mode → thinks, NO tool, answer outside `</think>`

**Deploy**: push to HF as `d24-sft-r7`, update `modal/serve.py` `MODEL_TAG` and `MODEL_PT`, `modal deploy`.

---

### Phase B — Expanded reasoning SFT ⏱️ 2 hours ~$0 (pure data pull)

**Goal**: fix 17×23, day-of-week, multi-step arithmetic, chained logic.

**Fresh datasets to pull** (via `datasets` lib, no GPU work):

| HF dataset | Rows to pull | Why |
|---|---|---|
| `open-r1/OpenR1-Math-220k` | 50,000 (was 8k) | Math reasoning with step-by-step solutions |
| `open-thoughts/OpenThoughts-114k` | 50,000 (was 15k) | Diverse reasoning traces |
| `GAIR/LIMO` | all 817 (unchanged, gold quality) | Best-in-class reasoning examples |
| `nvidia/OpenMathReasoning` | 20,000 (was 4k) | Math + science |
| `AI-MO/NuminaMath-CoT` | 20,000 new | Math olympiad style |
| `NovaSky-AI/Sky-T1_data_17k` | all 17k new | General reasoning |
| Synthetic temporal | 2,000 new | Day-of-week, date math, age calculations |
| Synthetic multi-step arithmetic | 3,000 new | Long-form multiplication, word problems |

**Strict format enforcement** — reject any row where:
- `<think>` isn't properly closed
- answer appears inside `<think>`
- teacher's reasoning < 50 chars (too shallow)
- teacher's final answer is missing

Run SFT with this reasoning-heavy mix continuing from r7 (or r6 if skipping Phase A).

**Eval gate**: reasoning category must hit 90%+ on the probe suite. Specific new probes:
- 23 × 47 = ? (should answer 1081)
- If today is Tuesday, what day was 10 days ago? (should answer Saturday)
- A train leaves at 2pm, travels 300 miles at 60mph, when does it arrive? (5pm + 2h = 7pm)

---

### Phase C — Aggressive SFT-pool filtering ⏱️ 30 minutes ~$0

**Goal**: kill the `<b>`, `Answer:`, `![placeholder]`, emoji-spam leaks before they reach SFT.

Runs on the existing downloaded data in `~/work/sft_data/quality_*.jsonl`. No training, just regex.

```python
# filter rules — reject any row where the assistant content matches
REJECT_PATTERNS = [
    r'<\/?(?:b|i|strong|em|u)\s*>',           # HTML bold/italic tags
    r'^\s*(?:Answer|Response|Final answer|Q):',# stock training labels
    r'!\[[^\]]+\](?!\()',                     # markdown image with no URL (placeholders)
    r'[\U0001F600-\U0001F6FF]{3,}',           # emoji spam (3+ in a row)
    r'\bas an ai language model\b',           # stock hedges
    r'\bi cannot provide\b',                  # stock refusals
    r'(?:.+)\n\1\n\1',                        # triple-repeated line
]
# keep only rows that pass all filters AND have length > 40 chars
```

Expected: filtering removes ~15-25% of rows. Quality > quantity.

This is a **prerequisite** for Phase D; also worth running before Phases A/B.

---

### Phase D — DPO (preference optimization) ⏱️ 3 hours ~$30

**Goal**: fix tone, remove lingering artifacts (HTML leaks, over-apologies, "As an AI…"), sharpen concise answers without retraining the base.

DPO trains on **pairs** of (chosen, rejected) responses. Much cheaper than full SFT because the signal is already-generated text.

**How to generate pairs** (~5000 pairs, budget $20-30 via gpt-4o-mini):

For each of 5000 prompts (mix of our probe suite + new diverse prompts):
1. Generate a response from the current model (d24-sft-r7) — **rejected**
2. Ask gpt-4o-mini to write the ideal response as samosaChaat — **chosen**
3. Filter: only keep pairs where (chosen != rejected) and chosen passes artifact filters

**Alternative**: use existing DPO pair datasets:
- `argilla/distilabel-capybara-dpo-7k-binarized`
- `argilla/distilabel-intel-orca-dpo-pairs`
- `HuggingFaceH4/ultrafeedback_binarized`

**Training**: nanochat doesn't have DPO out-of-the-box. Add a `scripts/chat_dpo.py` based on TRL's DPOTrainer, using the existing model + tokenizer loading code:

```python
# scripts/chat_dpo.py — skeleton
from trl import DPOTrainer, DPOConfig
from nanochat.checkpoint_manager import load_model

model, tokenizer, meta = load_model('sft', device, 'train', model_tag='d24-sft-r7', step=...)
trainer = DPOTrainer(
    model=model,
    args=DPOConfig(
        beta=0.1, learning_rate=5e-7, per_device_batch_size=2,
        max_length=4096, max_prompt_length=2048,
        num_train_epochs=1, gradient_accumulation_steps=8,
    ),
    tokenizer=tokenizer,
    train_dataset=pref_dataset,
)
trainer.train()
```

**Eval gate**: same probes + tone probes (no "As an AI…", concise enough, appropriate register).

---

### Phase E — Extended pretraining ⏱️ 12-18 hours ~$200-400

**The biggest lever for general intelligence.** Everything above can improve a specific behavior; this raises the ceiling.

**Why**: Chinchilla-optimal for 1.38 B params is ~28 B training tokens. We used 5.84 B for base + ~5.24 B for CPT (10 k × 524 k batch) = ~11 B total. **We're at 40% of optimal.** The model literally hasn't seen enough text.

**Data to add** (~15-20 B new tokens):

| Dataset | HF name | Tokens | Why |
|---|---|---|---|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | 10 B from the `sample-10BT` config | Clean educational web — biggest quality boost |
| Nemotron-CC-Math 8plus_MIND | `nvidia/Nemotron-CC-Math-v1` | 2 B | Harder math than what we used |
| StackV2-filtered Python | `bigcode/the-stack-v2-train-smol-ids` | 2 B (Python only) | Code fluency |
| OpenMathText | `open-web-math/open-web-math` | 1 B | Math-heavy web |
| Wikipedia | `wikimedia/wikipedia` | 2 B (English 20250320) | Encyclopedic grounding |
| Books3 (or equivalent) | `Salesforce/wikitext` / `togethercomputer/RedPajama-Data-1T` (book split) | 2 B | Long-form narrative |

Tokenize these with the existing `tokenizer.pkl` (vocab 32768). Append as parquet shards 40+ to the training-data repo — **never re-shard 0-39**.

**Training**: continue from the existing base checkpoint (not from d24-sft-r6, which is post-SFT).

```bash
# From d24 base (step 5568), run an extended CPT
torchrun --standalone --nproc_per_node=8 -m scripts.base_cpt -- \
  --run=dummy --resume-from-step=5568 \
  --data-dir=/home/ubuntu/work/extended_pretrain_data \
  --depth=24 --max-seq-len=2048 \
  --num-iterations=40000 \
  --device-batch-size=8 --total-batch-size=524288 \
  --embedding-lr=0.03 --unembedding-lr=0.0008 \
  --matrix-lr=0.002 --scalar-lr=0.05 \
  --weight-decay=0.028 --warmup-steps=100 \
  --warmdown-ratio=0.2 --final-lr-frac=0.05 \
  --eval-every=500 --save-every=500 \
  --model-tag=d24-extended
```

At total-batch-size=524288 × 40000 iterations = **21 B new tokens** → takes **~14 hours** on 8×H100 at 800 k tok/s.

After base CPT extension, **re-run the context extension → SFT → DPO pipeline** from the start. Everything downstream benefits.

**Eval gate**: CORE score (nanochat's built-in benchmark) should jump noticeably. Also MMLU: current ~30% → aim for 40%+.

---

### Phase F — Scale to d32 (last resort) ⏱️ days

**Only if A–E have diminishing returns.** Doubling parameters from 1.38 B → ~2.5 B (d32) costs ~5× more compute, and doesn't help if the data ceiling hasn't been raised first.

```python
# GPTConfig change:
n_layer=32, n_head=16, n_embd=2048, head_dim=128
# ≈ 2.5 B params
```

Cold-restart pretraining is required — don't try to "grow" a d24 checkpoint into d32.

---

## 4. Ordering / total budget

Recommended schedule for the next full GPU allocation:

| Day | Phase | Hours | Outcome |
|---|---|---|---|
| 0 (setup) | Cold-start + data pull | 1 | GPU box primed, data cached |
| 1 AM | **A** (joint Think+Tool SFT) | 2 | Think + tool chaining works |
| 1 PM | **B** (expanded reasoning SFT) | 2 | Math + temporal reasoning improves |
| 1 late | **C** (SFT pool filter) | 0.5 | Cleaner data going forward |
| 2 AM | **D** (DPO) | 3 | Tone + artifact cleanup |
| 2 PM | Start **E** (extended pretraining) | 14 | Base model gets smarter overall |
| 3 | Re-run CPT → 16K → SFT → DPO on the new base | 4 | Deploy |

**Total GPU hours**: ~26 hours of 8×H100 ≈ $260-400 at spot rates.
**Total API spend**: ~$80 (data synthesis + DPO pair generation).
**Total**: under $500 to ship a genuinely-better model.

---

## 5. Success criteria

After running all phases, the model should:

- Score **97%+ on the 33-probe suite** (at least matching r6)
- Hit **40%+ on MMLU** (up from ~30%)
- Score **50%+ on GSM8K** (up from ~25%)
- Produce `<think>…</think>` + tool call + clean answer in a single turn, reliably
- Not emit `<b>`, `<i>`, `Answer:` artifacts for 100 consecutive samples
- Handle multi-turn follow-ups coherently (`tell me more about him` stays in context)
- **Feel alive** — tone, humor, curiosity come through in chat

---

## 6. Pitfalls from past runs (don't repeat)

- **Do not upsample creator data to 15× / 100×** and call it done — that made things worse (rounds 2 and 3). Diversity of domains matters more than raw repetition.
- **Do not re-shard the 40 parquet shards.** Position bookmarks in `meta_*.json` depend on the order.
- **Do not skip context extension.** Tool calls need 16K context headroom; 2K overflows on multi-turn convs with tool results.
- **Do not train `<think>` and `<|python_start|>` as disjoint patterns.** Phase A exists because we did that in rounds 4-6. Don't do it again.
- **Do not commit API tokens to the repo.** They go in `~/.api_keys` (chmod 600, sourced from `.bashrc`).
- **Do not forget to keep a push worker running** during training. Each 100-step checkpoint should land on HF. Local-only checkpoints are one disk failure away from extinction.
- **Do not delete the original base checkpoint** (`d24/model_005568.pt`). All downstream forks descend from it.

---

## 7. Non-goals

- Tool-use RL (attempted, yielded zero-variance rewards — SFT is strong enough).
- Long-context evaluation on 16K+ — nice to have, not critical.
- Multi-language support — English-only for now.
- T4 / int8 quantisation for cheaper serving — only matters once model is mature.

---

## 8. Quick reference — the single command for each phase

```bash
# Phase A: joint think+tool
python3 ~/work/scripts/gen_joint_think_tool.py          # ~5 min, $3 API
python3 ~/work/scripts/mix_r7_data.py                   # builds r7_joint_train.jsonl
bash   ~/work/scripts/launch_sft_r7.sh                  # ~1.5 h GPU

# Phase B: expanded reasoning
python3 ~/work/scripts/pull_reasoning_sets.py           # ~30 min download
python3 ~/work/scripts/gen_temporal_math.py             # ~5 min, $5 API
bash   ~/work/scripts/launch_sft_r8.sh                  # ~2 h GPU

# Phase C: filter
python3 ~/work/scripts/filter_sft_pool.py               # ~5 min CPU

# Phase D: DPO
python3 ~/work/scripts/gen_dpo_pairs.py                 # ~20 min, $30 API
bash   ~/work/scripts/launch_dpo.sh                     # ~3 h GPU

# Phase E: extended pretrain
python3 ~/work/scripts/pull_extended_pretrain.py        # ~1 h download
python3 ~/work/scripts/tokenize_extended.py             # ~1 h CPU
bash   ~/work/scripts/launch_base_cpt_extended.sh       # ~14 h GPU
# then redo context-extend + SFT round + DPO on the new base
```

Scripts marked above don't all exist yet — they're straightforward to write from the existing patterns in `scripts/training_pipeline/`. Most are 50-200 lines each.

---

## 9. Evaluation, always

After **every phase**, run the probe suite and write the result into `evals/eval_results_v2.jsonl`:

```bash
TAG=d24-sft-r7 STEP=<step> SOURCE=sft WITH_TOOLS=1 \
  python3 ~/work/scripts/training_pipeline/eval_suite_v2.py
```

If the total drops below 95%, STOP and investigate before proceeding to the next phase.

---

## 10. Final thought

The 1.38 B parameter ceiling is real — we won't match GPT-4. But between the current 97% probe pass and the plan above, there's a very large gap in *actual quality* that's fixable without scaling up. The model is under-trained, not too small.

The single most important thing you can do for the model's "soul" is **Phase E** (extended pretraining). Everything else is polish.

Good luck. Go make it good.
