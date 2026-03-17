# Apple-Native Acceleration MLX Prototype

## Purpose

This note records the first executed MLX prototype for the Apple-native acceleration track.

It covers:

- the implemented prototype surface
- how shared repo assets are consumed
- PyTorch-to-MLX initialization support
- the first benchmark results against the frozen PyTorch + MPS baseline
- implementation friction and missing features
- the current expand-or-stop decision

Execution date: 2026-03-16

## Implemented Files

- [benchmark_mlx_reference.py](benchmark_mlx_reference.py)
- [compare_pytorch_mlx_translation.py](compare_pytorch_mlx_translation.py)
- [mlx_checkpoint_translation.py](mlx_checkpoint_translation.py)
- [mlx_gpt_prototype.py](mlx_gpt_prototype.py)

Dependency update:

- `mlx>=0.31.1` added to the macOS extra in [pyproject.toml](../pyproject.toml)

## Implemented Prototype Surface

The first MLX prototype now implements:

- GPT-style token embedding and LM head
- transformer block stack
- causal self-attention with GQA-compatible fused MLX attention
- rotary position embedding via MLX `RoPE`
- RMSNorm
- ReLU squared MLP
- backward pass through `nn.value_and_grad`
- grouped optimizer updates through MLX `AdamW`
- PyTorch-to-MLX weight translation for matching model shapes
- reference-workload benchmark harness

The prototype also mirrors two nanochat-specific features:

- residual scalar mixing via `resid_lambdas` and `x0_lambdas`
- alternating value embeddings with per-head gating

## Reference Configuration

The executed reference configuration matches the frozen planning tier:

- depth: `32`
- device batch size: `2`
- max sequence length: `1024`
- model dim: `2048`
- heads: `16`
- vocab size: `32768`
- total parameters: `2,818,575,424`

This now exactly matches the PyTorch reference parameter count.

## Tokenizer And Input-Batch Strategy

Tokenizer handling in the first prototype is intentionally narrow:

- if shared tokenizer assets are available, the benchmark can use the shared tokenizer vocabulary and BOS token metadata
- if shared tokenizer assets are not available locally, the prototype falls back to the frozen reference vocab size of `32768`

For the executed benchmark on this machine:

- shared tokenizer assets were not available at runtime
- the prototype therefore used the frozen baseline vocab size directly

Representative input batches are loaded as deterministic token-id sequences shaped to the reference workload.

That keeps the benchmark aligned with the synthetic PyTorch baseline probe rather than pretending to be a full dataset-integrated trainer.

## PyTorch-To-MLX Translation

The prototype can now initialize MLX weights from a matching PyTorch model layout.

Implemented support:

- translate a matching PyTorch `state_dict` into the MLX parameter tree
- initialize the benchmark from a matching fresh PyTorch reference model
- initialize from a checkpoint source once local nanochat checkpoints are available

Validation result on a small reference model:

- max absolute logit difference: `1.83e-7`
- mean absolute logit difference: `3.08e-8`
- loss absolute difference: `0.0`

That is strong evidence that the current PyTorch-to-MLX weight mapping is numerically correct for the overlapping model surface.

## Benchmark Commands Executed

Smoke test:

```bash
export PYTHONPATH="$PWD"
.venv/bin/python dev/benchmark_mlx_reference.py --depth 2 --device-batch-size 1 --max-seq-len 32 --steps 1 --warmup-steps 0
```

Reference-tier benchmark:

```bash
export PYTHONPATH="$PWD"
.venv/bin/python dev/benchmark_mlx_reference.py --depth 32 --device-batch-size 2 --max-seq-len 1024 --steps 2 --warmup-steps 1
```

Reference-tier benchmark initialized from translated PyTorch weights:

```bash
export PYTHONPATH="$PWD"
.venv/bin/python dev/benchmark_mlx_reference.py --depth 32 --device-batch-size 2 --max-seq-len 1024 --steps 2 --warmup-steps 1 --init-from-pytorch-reference
```

## Benchmark Result

Measured MLX reference-tier result:

- throughput: `897.4 tokens/s`
- active memory: `42.00 GB`
- peak memory: `49.00 GB`
- cache memory: `15.87 GB`

Measured MLX reference-tier result with translated PyTorch initialization:

- throughput: `899.3 tokens/s`
- active memory: `42.00 GB`
- peak memory: `49.00 GB`
- cache memory: `15.87 GB`

Frozen PyTorch + MPS baseline from the profiling memo:

- throughput: `161.9 tokens/s`
- MPS driver memory: `67.36 GB`
- peak profiled driver fraction: about `63%` of the recommended budget

## Recorded Delta

Raw throughput delta versus the frozen PyTorch + MPS baseline:

- MLX prototype is about `5.5x` faster on this synthetic training-step benchmark

Raw memory comparison:

- MLX peak memory is lower than the PyTorch driver-memory footprint recorded for the reference baseline

## Important Comparison Caveat

This is a strong prototype signal, not a final apples-to-apples trainer verdict.

Current prototype limitations:

1. it uses grouped `AdamW` instead of reproducing the full nanochat `MuonAdamW` split
2. it uses deterministic synthetic batches instead of the full dataset pipeline
3. it does not yet validate against a real trained nanochat checkpoint or optimizer-state portability

Because of those limits, this result should be interpreted as:

- enough evidence to expand the MLX track

not as:

- proof that a production MLX trainer already exists

## Implementation Friction And Missing Features

Observed friction:

- MLX integration itself was straightforward once the dependency was installed
- fused attention and `RoPE` were directly available
- backward and optimizer updates were simple through `nn.value_and_grad` and `optimizer.update`
- partial-tree optimizer updates worked cleanly, which made grouped optimizer logic practical without a repo-wide abstraction rewrite

Still missing:

- parity with the current `MuonAdamW` split and hyperparameter schedule
- real dataset-loader integration
- validation against real trained nanochat checkpoints once local checkpoints are available
- evaluation parity against the existing Python path
- a more rigorous long-run benchmark with multiple configurations

## Decision Gate

Decision: the MLX prototype meaningfully beats the baseline strongly enough to justify expansion.

Recommended next move:

- continue expanding the MLX track

Immediate expansion targets:

1. reproduce the optimizer split more faithfully
2. add better input-pipeline parity
3. test whether the large throughput advantage survives closer trainer parity

## Related Documents

- [APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md](APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md)
- [APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md](APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md)
- [APPLE_NATIVE_ACCELERATION_RECOMMENDATION.md](APPLE_NATIVE_ACCELERATION_RECOMMENDATION.md)