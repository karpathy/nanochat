# NanoChat Chinese Training Experiment Summary

Date: 2026-06-20

## Goal

Evaluate whether the existing NanoChat d6 model can acquire Chinese behavior
without replacing its tokenizer, then distinguish three separate capabilities:

1. Producing valid Chinese text.
2. Modeling natural Chinese text.
3. Answering Chinese instructions with useful and coherent content.

The experiments were designed for a local Apple M4 machine with 16 GB memory.
Existing checkpoints were preserved under separate model tags.

## Baseline

Model checkpoints:

```text
base/d6/step5000
chatsft/d6/step1500
```

Model configuration:

```text
layers: 6
hidden size: 384
heads: 6
context: 512
vocabulary: 32768
parameters: 73,531,646
```

The tokenizer can encode arbitrary UTF-8 text, but it is strongly optimized for
English. Examples:

| Text | Characters | Tokens |
|---|---:|---:|
| `机器学习` | 4 | 11 |
| `请用中文简单介绍北京。` | 11 | 26 |
| `Machine learning` | 16 | 2 |
| `Beijing` | 7 | 2 |

The original English SFT model answered only 4 of 100 Chinese prompts mainly in
Chinese. Those four responses were not useful answers; they were mostly repeated
Chinese characters. Most Chinese prompts were routed into familiar English
SpellingBee or generic reasoning templates.

## Code Changes

The experiment adds optional behavior while preserving existing defaults:

- Incremental UTF-8 decoding for the CLI.
- Optional custom Parquet directory for base training and BPB evaluation.
- Weight-only base checkpoint initialization with a fresh optimizer and data
  position.
- Optional custom SFT train/validation JSONL data, token-ratio sampling,
  independent output tags, and intermediate checkpoints.
- Reproducible Chinese data preparation, language evaluation, and run scripts.

Compatibility principles:

- Existing commands do not need new arguments.
- `--resume-from-step` keeps its original optimizer and dataloader resume
  semantics.
- `--init-from-model-tag` is a separate weight-only initialization path.
- Default data discovery remains unchanged when `--data-dir` is omitted.
- Chinese experiment checkpoints use independent tags and do not overwrite d6.

## Prepared Data

### Chinese SFT

Source:

```text
shibing624/alpaca-zh
```

Prepared data:

```text
train rows: 45,518
validation rows: 1,000
rendered training tokens at the 512-token cap: 17,237,842
assistant characters: 10,275,133
CJK assistant characters: 7,782,773
```

Chinese data was sampled to approximately 30% of SFT rendered tokens. Existing
English tasks remained in the training mixture.

### Continued Pretraining

Sources:

```text
FineWeb2 Simplified Chinese: cmn_Hani
existing ClimbMix English data
```

Prepared mixture:

```text
target ratio: 70% Chinese / 30% English by token
training documents: 15,131
training tokens: 19,031,250
validation documents: 796
validation tokens: 970,532
```

The final Parquet shard is reserved for validation. The manifest records dataset
sources, tokenizer identity, seed, and sample statistics.

## Experiment 1: Chinese SFT Only

Training:

```text
source: base/d6/step5000
output: chatsft/d6-zh-sft-demo
steps: 600
checkpoints: 200, 400, 600
Chinese SFT token share: approximately 30%
```

Validation BPB:

| Step | BPB |
|---|---:|
| 200 | 0.8692 |
| 400 | 0.8382 |
| 600 | 0.7944 |

Deterministic language evaluation at step 600:

```text
Chinese response rate: 100%
English response rate: 98%
responses containing replacement characters: 0
```

Conclusion:

- SFT successfully teaches language selection and valid Chinese generation.
- It does not create strong Chinese knowledge or semantic coherence.
- Responses remain repetitive, generic, and frequently copy prompt wording.

## Experiment 2: Chinese Continued Pretraining

Training:

```text
source: base/d6/step5000
output: base/d6-zh-cpt
steps: 1200
tokens: approximately 19.66 million
learning rates: 10% of the original pretraining rates
fresh optimizer and dataloader
```

Pure Chinese validation BPB:

| Model | BPB |
|---|---:|
| original d6/5000 | 1.6263 |
| CPT step 200 | 1.3846 |
| CPT step 400 | 1.2755 |
| CPT step 800 | 1.1688 |
| CPT step 1200 | 1.1463 |

Chinese BPB improved by approximately 29.5%, demonstrating a real improvement in
Chinese language modeling.

Mixed Chinese/English validation BPB:

```text
original d6: 1.1650
CPT step 400: 1.1601
CPT step 1200: 1.1685
```

The mixed metric improved early and then slightly regressed while pure Chinese
BPB continued to improve. This suggests a small English tradeoff during later
CPT, not failure to learn Chinese.

## Experiment 3: CPT Followed by SFT

### Low-LR Run

The first run used:

```text
output: chatsft/d6-zh-cpt-sft
```

`chat_sft.py` inherited the reduced CPT learning rates, making its SFT learning
rates 10 times lower than the SFT-only experiment. At step 600:

```text
validation BPB: 0.8332
Chinese response rate: 100%
English response rate: 100%
```

It showed more repetition and prompt copying. This run is retained as a useful
low-learning-rate control, but it is not a fair CPT versus non-CPT comparison.

### Learning-Rate-Corrected Run

Training:

```text
source: base/d6-zh-cpt/step1200
output: chatsft/d6-zh-cpt-sft-lrfix
steps: 600
embedding LR: 0.3
unembedding LR: 0.008
matrix LR: 0.02
```

Validation BPB:

| Step | BPB |
|---|---:|
| 200 | 0.8724 |
| 400 | 0.8399 |
| 600 | 0.7952 |

Final deterministic language evaluation:

```text
Chinese response rate: 100%
English response rate: 99%
responses containing replacement characters: 0
```

The corrected run recovered the SFT-only optimization level:

```text
SFT-only BPB: 0.7944
CPT+SFT corrected BPB: 0.7952
```

However, qualitative Chinese answer quality remained close to SFT-only. CPT
improved Chinese next-token modeling but did not clearly improve factuality,
instruction reasoning, or resistance to repetitive loops at this scale.

## ChatCORE Comparison

All evaluations used 128 examples per task and deterministic decoding.

| Task | Original SFT | Chinese SFT | Chinese CPT+SFT |
|---|---:|---:|---:|
| ARC-Easy | 22.66% | 23.44% | 25.78% |
| ARC-Challenge | 24.22% | 25.78% | 26.56% |
| MMLU | 21.88% | 24.22% | 26.56% |
| GSM8K | 0% | 0% | 0% |
| HumanEval | 0% | 0% | 0% |
| SpellingBee | 91.41% | 65.62% | 75.78% |
| ChatCORE | 0.1385 | 0.1059 | 0.1350 |

Interpretation:

- Direct Chinese SFT caused substantial forgetting of the heavily trained
  SpellingBee behavior.
- Mixed CPT before SFT recovered part of that loss and restored aggregate
  ChatCORE close to the original model.
- ARC and MMLU remain near their 25% random baselines. Small differences at 128
  examples are not strong evidence of improved reasoning.
- GSM8K and HumanEval remain at zero, showing that Chinese training does not
  address the base model's math and code limitations.
- Aggregate ChatCORE is dominated by SpellingBee and must not be interpreted as
  broad general capability.

## Final Conclusions

Confirmed:

1. The existing tokenizer can represent Chinese, but inefficiently.
2. Incremental decoding removes display corruption from partial UTF-8 tokens.
3. Chinese SFT reliably teaches the model to answer in Chinese.
4. Chinese CPT significantly improves Chinese language-model BPB.
5. Explicit SFT learning rates are required for a fair post-CPT comparison.
6. Mixed CPT reduces the English forgetting caused by direct Chinese SFT.

Not confirmed:

1. CPT did not clearly improve Chinese assistant semantics or factuality.
2. The model still lacks reliable math, code, and broad reasoning abilities.
3. The small ARC/MMLU changes are within the range where evaluation noise is
   significant.

The main bottlenecks are the small six-layer Transformer core, limited
pretraining data, 512-token context, English-centric tokenizer, and an SFT
mixture heavily biased toward SpellingBee token patterns.

## Recommended Next Work

Before further Chinese-only tuning:

1. Strengthen the English/base model with substantially more pretraining tokens.
2. Evaluate base BPB, CORE, fixed factual prompts, and repetition at regular
   checkpoints.
3. Re-run the same SFT recipe from the improved base to isolate base-model gains.
4. Rebalance SFT by assistant-token mass, reducing SpellingBee dominance and
   adding legitimate math and code instruction data.
5. Only after establishing a stronger base, train a bilingual tokenizer and a
   new bilingual model from scratch.

The current experiment is successful as a mechanism study: it establishes a
reproducible Chinese SFT/CPT pipeline and clearly separates language selection,
language modeling, and assistant-quality outcomes.
