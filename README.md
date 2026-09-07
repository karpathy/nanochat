# nanochat

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat is a minimal, hackable harness for training and evaluating small language models end to end on one GPU node: data, tokenizer, pretraining, evaluation, inference, and finetuning into a chat model you can talk to. It is written to be read in one sitting and forked, not configured. There are no config files or model factories: **the code is the config**, and one integer, `--depth`, sets every other hyperparameter (width, heads, learning rates, batch size, training horizon, weight decay) so that each model comes out compute-optimal. Sweeping the depth traces a *miniseries* of models across two orders of magnitude of compute, and that cost-performance curve is the product of an experiment: the quality, speed and cost of every model, in one file, comparable across experiments by compute multiplier. A GPT-2 grade model (depth 24) is one point on the curve and trains in under two hours on an 8XH100 node, roughly $50 at current prices.

For questions about the repo, use [DeepWiki](https://deepwiki.com/karpathy/nanochat), the [Discussions tab](https://github.com/karpathy/nanochat/discussions), or the [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) channel on Discord.

## Quick start

nanochat uses [uv](https://docs.astral.sh/uv/) for dependencies. On an 8XH100 node (e.g. from [Lambda](https://lambda.ai/service/gpu-cloud)):

```bash
uv sync --extra gpu          # --extra cpu for CPU/MPS; add --group dev for pytest, matplotlib, ...
source .venv/bin/activate
bash run.sh my_experiment
```

That one command runs one named *experiment* end to end. It prepares the data (downloads ~300 shards of [ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle), trains the tokenizer, packs the tokens once), then for every depth in the ladder `12 14 16 20 24 28` it pretrains a model, scores it on CORE, benchmarks its inference, finetunes it into a chat model, and evaluates that. Everything lands in `~/.cache/nanochat/experiments/my_experiment/`: checkpoints, one log per stage, and the aggregated `curve.log`. The full ladder takes about 13 hours. Every stage is idempotent: if the run crashes, re-run the same command and completed stages are skipped.

The ladder and stages are environment knobs:

```bash
DEPTHS="12 14 16" bash run.sh quick                    # a <1h mini-ladder for a quick scaling check
STAGES="base_train base_eval base_inference" bash run.sh pretrain_only   # a pretraining researcher: skip the chat stages
DEPTHS="24" BASE_TRAIN_FLAGS="--target-param-data-ratio=8 --fp8" bash run.sh fast_d24   # one depth, fewer tokens, fp8 matmuls
```

A running ladder can be steered by touching files in the experiment directory (`skip_d24`, `skip_chat_train`, `stop`); see the comments at the top of [run.sh](run.sh).

Once a depth has finished its SFT stage you can talk to it:

```bash
NANOCHAT_EXPERIMENT=my_experiment python -m scripts.chat_inference -p "Why is the sky blue?"
python -m scripts.chat_inference   # interactive; picks the largest chat model in the experiment
```

The models are small (a d24 is a 4e19 FLOPs model), so it is a bit like talking to a kindergartener. An example conversation with a d24:

> **User:** Hello whats up?
>
> **Assistant:** Hello! How can I help you today?
>
> **User:** Why is the sky blue?
>
> **Assistant:** The sky is blue due to an optical illusion called the Rayleigh Scattering of Light, which occurs when sunlight passes through tiny particles in the air. These particles are made up of tiny blue and violet particles that cause the light to bend in a particular way.
>
> **User:** That's cool. Write a poem about it.
>
> **Assistant:** In the twilight's glow, the sky is blue,<br>
> A canvas of shimmering blue, where skies unfold,<br>
> A world of blue, where blue and more blue,<br>
> Are woven together in a tapestry so fine,<br>
> Where every hue seems to sing a story.<br>
> ...

A few notes:

- The code also runs on an Ampere 8XA100 node, a bit slower and without `--fp8`.
- A single GPU works too (omit `torchrun` or set `NPROC_PER_NODE=1`): gradient accumulation makes the results ~identical, 8 times slower.
- With less than 80GB per GPU, pass a smaller `--device-batch-size` to `scripts/base_train.py` (it defaults to 32, 16 or 8 by depth) until it fits; the total batch size is held fixed by gradient accumulation.
- Most of the code is plain PyTorch, so xpu/mps/cpu should work but are less exercised. A laptop will not train a good model, but the code paths run at toy settings (`prepare -n 8`, a small `--depth`, `--max-seq-len=512`, a small `--device-batch-size`).

## What an experiment is

An experiment is `(name, git commit, depth ladder, dataset)`. `meta.json` records the commit (and the diff, if the working tree was dirty, so even dirty runs are reproducible) and the dataset name. To test an idea: edit the code, name an experiment, run the ladder, compare curves. The directory of an experiment looks like:

```
~/.cache/nanochat/experiments/<name>/
  meta.json                 commit, dirty diff, dataset, ladder
  d12/
    base/                   pretraining checkpoint
    chat/                   chat checkpoint
    base_train.log, base_eval.log, base_inference.log, chat_train.log, chat_eval.log
  d16/ ...
  curve.log                 one `model` record per depth: the product
```

The stage scripts print to stdout and `run.sh` tees each into its log. Lines of the form `summary key=value ...` are the machine-readable contract (the top of `harness/experiment.py`); everything else is prose. `python -m harness.experiment curve` joins the summaries into `curve.log` and prints the headline table, and `python -m harness.experiment compare baseline variant` reads compute multipliers off two experiments: at each of the variant's final models, how much compute the baseline needs to reach the same validation loss. A multiplier above 1 is a win. For calibration, the original GPT-2 (1.6B, 2019) scores 0.2565 on CORE; a d24 passes it.

## Data

`python -m scripts.base_prepare` does everything that touches text, once per dataset, into `~/.cache/nanochat/datasets/<name>/`: it downloads the raw parquet shards, trains a 32K-vocab BPE tokenizer (rustbpe to train, tiktoken to run), and packs every shard into rows of 2048 tokens where each row starts at a document start. `python -m scripts.chat_prepare` renders the finetuning conversations into the same format, and `run.sh` only runs it when a chat stage is planned. The packed shards are the only thing the training scripts read: a 256-byte header followed by two `uint16` sections, inputs and targets, readable from anywhere.

```python
header = np.fromfile(path, dtype=np.int32, count=64)              # magic, version, row_len, num_rows, vocab_size, tokenizer_id
inputs = np.memmap(path, dtype=np.uint16, mode="r", offset=256, shape=(num_rows, row_len))
```

Targets are stored explicitly rather than derived by shifting, so a position the model must not learn from (user turns and padding in the finetuning rows) is just a target value of 65535. Shards are what let this scale: download and pack a few for small models, more later for a bigger ladder, or multi-epoch over a handful with `--num-shards`. To bring your own data, drop parquet shards with a `text` column into `~/.cache/nanochat/datasets/<name>/` (the last shard is the validation split) and run with `NANOCHAT_DATASET=<name>`; everything downstream stays fixed.

## Research workflow

For quick iteration my favorite scale is a 12-layer model, about five minutes of pretraining on 8XH100 (prepare the data once first):

```bash
NANOCHAT_EXPERIMENT=my_experiment OMP_NUM_THREADS=1 \
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --save-every=-1
```

This uses wandb (run name "d12") and saves no intermediate checkpoints. I change something, re-run a d12 (or d16), and watch the wandb plots for `val_bpb` (validation loss in vocab-size-invariant bits per byte) against `step`, `total_training_time` and `total_training_flops`, and `train/mfu` and `train/tok_per_sec`; then `python -m scripts.base_eval -g d12` for the CORE score (the DCLM ensemble of 22 evals, a few minutes on the node). Then the change has to hold on the ladder: `bash run.sh` at a few depths and `python -m harness.experiment compare baseline variant`.

The important rule is that nanochat has one dial. `--depth` determines every other hyperparameter, so any candidate change has to be principled enough to work at every depth, and its value is read off the curve, not off one model. [dev/scaling_laws.sh](dev/scaling_laws.sh) sweeps a grid of (FLOPs budget, depth) inside one experiment to re-derive the compute-optimal frontier. The model itself (`nanochat/gpt.py`) is ~540 lines in a functional style, with rotary embeddings, QK norm, ReLU² MLPs, sliding-window attention, value embeddings with gates, and per-layer residual scalars; the optimizer (`nanochat/optim.py`) is Muon for the matrices and AdamW for the rest, sharded ZeRO-style across ranks without DDP.

The chat stages turn any base model into something you can talk to. SFT trains one pass over general conversations (SmolTalk) plus a multiple-choice format primer (MMLU's auxiliary train set), and the chat eval scores ARC-Easy, ARC-Challenge and MMLU by the logit of the answer letter, reported as ChatCORE (mean accuracy centered on chance). Models this small do not do multiple choice without the primer, which is why it is there.

## Precision

nanochat does not use `torch.amp.autocast`. Parameters and optimizer state are fp32; every matmul casts its operands to a single global `COMPUTE_DTYPE` at the matmul boundary (`bf16_matmul` in `nanochat/gpt.py`), and `--fp8` swaps in fp8 matmuls on Hopper. `COMPUTE_DTYPE` is auto-detected and can be overridden with `NANOCHAT_DTYPE`:

| Hardware | Default | Notes |
|----------|---------|-------|
| CUDA SM 80+ (A100, H100, ...) | `bfloat16` | native bf16 tensor cores |
| CUDA SM < 80 (V100, T4, ...) | `float32` | no bf16 tensor cores; fp16 is not supported |
| CPU / MPS | `float32` | recent macOS also runs `NANOCHAT_DTYPE=bfloat16` fine |

## Guides

Most recent first:

- [Feb 1 2026: Beating GPT-2 for <<$100: the nanochat journey](https://github.com/karpathy/nanochat/discussions/481)
- [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) documents the first nanochat miniseries of models.
- [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164) (the tool-use it describes has since been removed, but the method carries over).
- [Oct 13 2025: original nanochat post](https://github.com/karpathy/nanochat/discussions/1) introducing nanochat; parts are now deprecated and the model is a lot older than current master.

## File structure

`nanochat/` is the load-bearing code, what a model is and how it trains, and it never imports the rest (a test enforces this). `harness/` is the experiment around it (including the raw datasets, pretraining text and conversations), `evals/` measures models, and `scripts/` are the stages you run.

```
.
├── LICENSE
├── README.md
├── run.sh                          # The master script: one experiment end to end
├── experiment_refactor.md          # Design doc of the Sep 2026 refactor (retires at merge)
├── dev                             # Not the product: the research record and optional tooling
│   ├── LOG.md                      # The experiment log: what was tried, what happened, verdicts
│   ├── scaling_laws.sh             # (FLOPs x depth) grid inside one experiment
│   ├── scaling_analysis.ipynb      # Fits the compute-optimal frontier from that grid
│   ├── estimate_gpt3_core.ipynb    # Estimating GPT-3's CORE scores from the paper
│   ├── repackage_data_reference.py # How the canonical dataset's parquet shards were made
│   ├── nanochat.png
│   └── scaling_laws_jan26.png
├── nanochat                        # The meat: what a model is and how it trains
│   ├── gpt.py                      # The GPT Transformer, functional style
│   ├── optim.py                    # Muon + AdamW, single GPU and distributed
│   ├── dataloader.py               # The shard format, the packer, the memmap loader
│   ├── tokenizer.py                # BPE tokenizer, conversation rendering
│   ├── engine.py                   # Inference with a KV cache
│   ├── flash_attention.py          # FA3 on Hopper, SDPA fallback elsewhere
│   ├── fp8.py                      # FP8 matmul for --fp8
│   └── common.py                   # Compute dtype, rank info, print0, where things live on disk
├── harness                         # The experiment around it
│   ├── experiment.py               # The log grammar; init/curve/compare: meta.json, curve.log, compute multipliers
│   ├── checkpoint.py               # Save/load model checkpoints
│   ├── dataset.py                  # Raw pretraining text: named parquet datasets, download/verify/read
│   ├── tasks.py                    # Raw conversations: SmolTalk, MMLU, ARC from the hub
│   └── runtime.py                  # Distributed init, device autodetect, logging, hardware tables
├── evals
│   ├── core.py                     # The CORE metric (DCLM's 22-task ensemble)
│   └── bpb.py                      # Bits per byte
├── pyproject.toml
├── scripts
│   ├── base_prepare.py             # Base data: download, tokenizer, packed shards
│   ├── base_train.py               # Base model: train
│   ├── base_eval.py                # Base model: CORE, samples
│   ├── base_inference.py           # Base model: latency/throughput/VRAM bench
│   ├── chat_prepare.py             # Chat data: the SFT rows
│   ├── chat_train.py               # Chat model: SFT
│   ├── chat_eval.py                # Chat model: multiple-choice evals, ChatCORE
│   └── chat_inference.py           # Chat model: talk to it
├── tests
│   ├── test_dataloader.py          # Shard format, packer, loader
│   ├── test_engine.py              # KV-cache decode matches the full forward; sampling
│   ├── test_flash_attention.py     # The SDPA fallback matches FA3 (needs a Hopper GPU)
│   ├── test_layering.py            # nanochat/ never imports the harness
│   ├── test_experiment.py          # The log grammar
│   ├── test_optim.py               # MuonAdamW optimizer (needs GPU)
│   ├── test_tasks.py               # Hub dataset wrapper, multiple choice prompt
│   └── test_tokenizer.py           # BPE round-trips, chat rendering
└── uv.lock
```

Tests: `python -m pytest tests/ -v` (one GPU is enough).

## Contributing

nanochat wants to be the simplest solid baseline for end-to-end work on small language models: a reference implementation people fork, not a framework people configure. Accessibility is about cost, but also about cognitive complexity, so there are no giant configuration objects, model factories, or if-then-else monsters, and every addition is weighed against what it costs a reader. Improvements are judged on the curve: a change must be principled enough to hold across the depth ladder, measured against a baseline experiment with `python -m harness.experiment compare`, and a gnarly or esoteric change can lose to a simpler one that gains less.

AI policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for smoltalk, and to NVIDIA for ClimbMix.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Thank you to the repo czar Sofie [@svlandeg](https://github.com/svlandeg) for help with managing issues, pull requests and discussions of nanochat.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
