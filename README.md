# rustAgent

rustAgent is a Rust implementation of nanochat's tokenizer, model runtime,
training, evaluation, and serving workflows. It reads existing nanochat
checkpoints and tokenizer files, so existing cached models remain usable.

## Build

Install the Rust stable toolchain. Build the portable CPU binary with:

```sh
cargo build --release
```

For NVIDIA CUDA, build with `--features cuda` (FlashAttention 2 enabled) or
`--features cuda-base` (portable attention fallback). On Linux,
`--features cuda-nccl` enables NCCL collectives when libnccl is installed.

The Rust executable does not require a system Python interpreter or the Python
W&B package. For W&B training logs, set `WANDB_API_KEY`; Rust locates or fetches
the W&B service binary itself. Set `WANDB_CORE_PATH` to select a specific
binary.

## Use an existing nanochat cache

By default, checkpoints and tokenizer assets are read from
`~/.cache/nanochat/`. Set `NANOCHAT_BASE_DIR` to use another cache root. Rust
reads `tokenizer.pkl` directly; convert it to native JSON if desired:

```sh
cargo run --release -- convert-tokenizer \
  --input "$HOME/.cache/nanochat/tokenizer/tokenizer.pkl" \
  --output "$HOME/.cache/nanochat/tokenizer/tokenizer.json"

cargo run --release --features cuda -- chat \
  --source sft --model-tag d12 --step 100 --device auto

cargo run --release --features cuda -- serve \
  --source sft --model-tag d12 --step 100 --host 127.0.0.1 --port 8000
```

The API implements `GET /v1/models` and `POST /v1/chat/completions`, including
streaming responses. Set `NANOCHAT_API_KEY` to require a bearer token.

## Tokenizer and training

The tokenizer commands use local inputs or datasets in the nanochat cache.
Download the data before training when it is not already cached:

```sh
cargo run --release -- download-dataset --dataset climbmix --num-files 4
cargo run --release -- train-tokenizer --dataset climbmix --first-dataset none
```

Start a small base-model run with:

```sh
cargo run --release --features cuda -- train \
  --depth 4 --max-seq-len 512 --device-batch-size 1 --num-iterations 20 \
  --eval-every 0 --core-metric-every 0 --sample-every 0
```

For a fresh ClimbMix run, CrowdMind 2048 is prepended by default; use
`--first-dataset none` to skip it. Use `--model-tag` to name a run and
`--resume-from-step` to resume a numbered base checkpoint. `--run NAME` enables
W&B logging. Run `cargo run --release -- --help` or add `--help` after any
subcommand for the full options.

## Evaluation and inference

The CLI includes `base-eval`, `eval-core`, `eval-chat`, `eval-tokenizer`,
`eval-bench`, and `chat-rl`. Evaluation commands download supported benchmark
data when needed. `generate` accepts a model config and checkpoint directly:

```sh
cargo run --release --features cuda -- generate \
  --config model-config.json --checkpoint model.pt \
  --tokenizer "$HOME/.cache/nanochat/tokenizer/tokenizer.json" \
  --prompt "Hello" --num-samples 4 --device cuda
```

For local multi-process training, the Rust executable launches local rank
processes directly, so a Python launcher is not required:

```sh
cargo run --release --features cuda,cuda-nccl -- launch --nproc-per-node 2 train \
  --depth 12 --model-tag d12
```

## Development

Run the Rust test suite from WSL/Linux:

```sh
cargo test --release --locked
```

The Rust sources are under `src/`; Rust integration tests are under `tests/`.
`RUST_REWRITE.md` tracks remaining cutover work.

## Background

rustAgent continues the nanochat research project and preserves its model and
tokenizer formats. See the [nanochat guides](https://github.com/karpathy/nanochat/discussions)
and [leaderboard notes](dev/LEADERBOARD.md) for background on the original
training recipes. For citation guidance and acknowledgements, see the original
[nanochat repository](https://github.com/karpathy/nanochat).

MIT License.
