# NanoChat Chinese Capability Experiment

Date: 2026-06-20

## Goal

Compare three checkpoints without changing the existing tokenizer or overwriting
the current `d6` baseline:

```text
chatsft/d6:1500                 existing baseline
chatsft/d6-zh-sft-demo:600     Chinese SFT only
chatsft/d6-zh-cpt-sft:600      Chinese continued pretraining, then the same SFT
```

The experiment measures whether SFT can switch the response language and whether
continued pretraining improves Chinese language modeling.

## Important Separation

Both branches start from the same original base checkpoint:

```text
base/d6:5000
├── Chinese SFT
└── Chinese/English continued pretraining -> Chinese SFT
```

Do not continue pretraining from an SFT checkpoint. Continued pretraining can
erase chat formatting and would make the comparison difficult to interpret.

## Environment

```bash
cd /Users/tw/Documents/NanoChat/nanochat
source .venv/bin/activate
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
```

On Apple Silicon the run script defaults to MPS. Override when needed:

```bash
DEVICE_TYPE=cpu bash runs/runzh.sh baseline
```

## 1. Prepare Data

```bash
bash runs/runzh.sh prepare
```

Outputs:

```text
$NANOCHAT_BASE_DIR/zh_experiment/
  manifest.json
  sft/train.jsonl
  sft/val.jsonl
  pretrain/shard_00000.parquet
  pretrain/...
  pretrain/shard_99999.parquet
  pretrain_zh_eval/shard_00000.parquet
  pretrain_zh_eval/shard_99999.parquet
```

The preparation script:

- Converts public
  [shibing624/alpaca-zh](https://huggingface.co/datasets/shibing624/alpaca-zh)
  instruction rows into NanoChat conversation JSONL.
- Keeps 1,000 deterministic validation conversations.
- Streams Simplified Chinese
  [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) text.
  The FineWeb2 configuration is `cmn_Hani` (Mandarin Chinese, Han script).
- Mixes Chinese FineWeb2 and local English ClimbMix at an approximate 70:30
  token ratio.
- Builds about 20 million tokens using the existing tokenizer.
- Writes dataset revisions, actual token counts, file hashes, and tokenizer SHA
  to `manifest.json`.

The SFT source should be treated as research/non-commercial data unless its
upstream terms explicitly permit broader use.

## 2. Baseline

```bash
bash runs/runzh.sh baseline
```

This evaluates 100 fixed Chinese prompts and 100 fixed English prompts at
temperature zero. It records:

- Chinese-dominant response rate.
- English-dominant response rate.
- Mean Chinese-character ratio.
- Responses containing the Unicode replacement character.
- Every prompt and complete response for manual inspection.

## 3. SFT-Only Branch

```bash
bash runs/runzh.sh sft
```

Configuration:

```text
source: base/d6:5000
output: chatsft/d6-zh-sft-demo
steps: 600
checkpoints: 200, 400, 600
fresh optimizer
custom Chinese task weighted to an estimated 30% of rendered SFT tokens
```

Evaluate individual checkpoints:

```bash
python -m scripts.zh_eval -i sft -g d6-zh-sft-demo -s 200 --device-type=mps
python -m scripts.zh_eval -i sft -g d6-zh-sft-demo -s 400 --device-type=mps
python -m scripts.zh_eval -i sft -g d6-zh-sft-demo -s 600 --device-type=mps
```

## 4. Continued-Pretraining Branch

```bash
bash runs/runzh.sh cpt
```

Configuration:

```text
source weights: base/d6:5000
output: base/d6-zh-cpt
steps: 1,200
tokens: 19,660,800
learning rates: 10% of the original base-training CLI values
fresh optimizer and dataloader state
checkpoints: every 200 steps
```

Then run the same SFT setup:

```bash
bash runs/runzh.sh cpt-sft
```

## 5. Compare

```bash
bash runs/runzh.sh eval-all
```

This runs:

- Language-response evaluation for all three SFT models.
- BPB on the same held-out Chinese Parquet data before and after continued
  pretraining.
- The existing ChatCORE task set on both Chinese SFT outputs.

Primary acceptance targets:

```text
Chinese prompts answered mainly in Chinese: >= 80%
Responses containing replacement characters: 0
English prompts answered mainly in English: >= 80%
SpellingBee after training: >= 80%
ChatCORE drop versus comparable baseline: <= 0.05
Chinese validation BPB after CPT: lower than base/d6
```

## Known Limits

- The tokenizer remains English-centric; most Chinese characters take two or
  three tokens.
- The d6 model is intentionally small and undertrained.
- Passing the language-response test demonstrates output-language control, not
  broad Chinese knowledge or reasoning.
- Training on M4 16 GB is expected to take hours. Run stages separately and keep
  the machine powered.
