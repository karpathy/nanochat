# Apple-Native Acceleration Handoff

## Purpose

This document is a fresh-context handoff for the current Apple Silicon native acceleration phase.

It is written for either:

- a new agent starting with no prior chat history
- a human who needs a concise technical state-of-play

The focus of this phase is narrow:

- use Apple-native frameworks and execution paths effectively on Apple Silicon
- validate whether MLX is a practical native training path for nanochat
- avoid drifting into broader architecture work unless it directly supports that goal

## Executive State

The Apple-native MLX track is already past feasibility.

That decision is no longer the question.

The current state is:

- MLX forward, backward, and optimizer-step execution are working
- the MLX reference prototype beats the frozen PyTorch + MPS baseline materially on the synthetic reference workload
- PyTorch-to-MLX weight translation is implemented and numerically validated
- a short-run MLX training sanity check is implemented and passing
- a longer reference-tier MLX training session has been executed successfully on the stable eager MLX path
- an experimental Muon-style matrix optimizer exists, but it is much slower than the current grouped-AdamW matrix path
- dataset-backed MLX input mode exists in code, but could not be exercised because local parquet shards are not available on this machine
- real-checkpoint MLX validation exists in code, but could not be exercised because local trained checkpoints are not available on this machine

The current best practical Apple-native training path is:

- MLX model
- MLX fused attention and RoPE
- MLX eager execution
- grouped AdamW-style optimizer path for stability and speed
- repeated synthetic batch input
- optional initialization from translated PyTorch reference weights

## Files To Read First

Read these first, in this order:

1. [APPLE_NATIVE_ACCELERATION_RECOMMENDATION.md](APPLE_NATIVE_ACCELERATION_RECOMMENDATION.md)
2. [APPLE_NATIVE_ACCELERATION_TASKLIST.md](APPLE_NATIVE_ACCELERATION_TASKLIST.md)
3. [APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md](APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md)
4. [APPLE_NATIVE_ACCELERATION_MLX_TRAINING_CHECK.md](APPLE_NATIVE_ACCELERATION_MLX_TRAINING_CHECK.md)
5. [mlx_gpt_prototype.py](mlx_gpt_prototype.py)
6. [benchmark_mlx_reference.py](benchmark_mlx_reference.py)
7. [mlx_training_check.py](mlx_training_check.py)
8. [mlx_training_session.py](mlx_training_session.py)
9. [mlx_checkpoint_translation.py](mlx_checkpoint_translation.py)
10. [mlx_input_batches.py](mlx_input_batches.py)

## What Has Been Implemented

### Core MLX Prototype

Implemented in [mlx_gpt_prototype.py](mlx_gpt_prototype.py):

- GPT-style token embedding and LM head
- transformer block stack
- fused causal attention using MLX scaled dot-product attention
- RoPE using MLX native implementation
- stateless RMSNorm helper
- ReLU squared MLP
- residual scalar controls via `resid_lambdas` and `x0_lambdas`
- alternating value embeddings with gating

### Optimizer Paths

Implemented optimizer paths:

- grouped AdamW-style optimizer path for the main MLX prototype
- experimental Muon-style matrix optimizer for closer algorithmic resemblance to nanochat

Important current judgment:

- grouped AdamW path is the practical path
- Muon-style path is exploratory only at this point

### Translation And Initialization

Implemented in [mlx_checkpoint_translation.py](mlx_checkpoint_translation.py):

- PyTorch config to MLX config mapping
- PyTorch `state_dict` to MLX parameter-tree translation
- initialization from a matching fresh PyTorch reference model
- optional initialization from checkpoint source when local checkpoints exist

Validation script:

- [compare_pytorch_mlx_translation.py](compare_pytorch_mlx_translation.py)

### Benchmarking And Checks

Implemented execution tooling:

- [benchmark_mlx_reference.py](benchmark_mlx_reference.py)
- [mlx_training_check.py](mlx_training_check.py)
- [mlx_training_session.py](mlx_training_session.py)

Coverage:

- short synthetic benchmark runs
- short-run training health checks with explicit pass/fail criteria
- longer training session execution with periodic progress output

### Input Modes

Implemented in [mlx_input_batches.py](mlx_input_batches.py):

- repeated synthetic reference batch mode
- dataset-backed batch mode using nanochat tokenizer and parquet text iteration

Status:

- repeated mode is exercised and working
- dataset-backed mode is implemented but not yet validated on this machine because local data shards are missing

## Verified Results

### Frozen Baseline Reference

Frozen PyTorch + MPS baseline from the project docs:

- throughput: about `161.9 tok/s`
- MPS driver memory: about `67.36 GB`

### MLX Reference Benchmark

Reference-tier MLX benchmark on the practical grouped-AdamW matrix path:

- configuration: `depth=32`, `batch=2`, `seq=1024`
- params: `2,818,575,424`
- throughput: about `897` to `905 tok/s`
- peak memory: about `49.0 GB`

Interpretation:

- MLX is about `5.5x` faster than the frozen PyTorch + MPS baseline on this synthetic reference benchmark
- MLX peak memory is lower than the baseline memory footprint

### PyTorch-To-MLX Translation Validation

Small-model translation check:

- max absolute logit diff: about `1.83e-7`
- mean absolute logit diff: about `3.08e-8`
- loss absolute diff: `0.0`

Interpretation:

- translation is numerically correct for the overlapping model surface already implemented

### Short-Run Training Check

Reference-tier short-run MLX training check on grouped-AdamW matrix path:

- status: `PASS`
- initial loss: about `9.349`
- final loss: about `2.092`
- mean throughput: about `803 tok/s`
- peak memory: about `59.5 GB`

Interpretation:

- the MLX path is not just runnable, it behaves like a healthy training loop under repeated steps

### Longer Training Session

Reference-tier longer MLX training session on grouped-AdamW matrix path:

- configuration: `depth=32`, `batch=2`, `seq=1024`, `steps=32`, `warmup=2`
- execution mode: eager MLX
- initialization: translated PyTorch reference weights
- initial measured loss: about `2.092`
- final loss: about `0.540`
- minimum loss: about `0.501`
- loss drop: about `74.2%`
- mean throughput: about `905 tok/s`
- peak memory: about `49.0 GB`
- wall time: about `72.4 s`

Interpretation:

- the current stable Apple-native MLX training path holds up over a materially longer session, not just a few steps

### Experimental Muon Path

Reference-tier Muon-style matrix optimizer runs:

- short-run health check: `PASS`
- mean throughput during health check: about `258 tok/s`
- raw benchmark throughput: about `269 tok/s`

Interpretation:

- numerically healthy
- not performance-competitive with the grouped-AdamW matrix path
- currently not the recommended path for this phase

## What Is Working Really Well

This section is intentionally human-readable rather than process-heavy.

### MLX Is a Real Win on This Machine

The core conclusion is strong and already supported by execution evidence.

MLX is not just theoretically attractive here. It is already materially better than the current PyTorch + MPS baseline for the specific reference workload this phase uses.

That matters because it means the Apple-native direction is justified by measurements, not preference.

### The Stable Path Is Actually Stable

The grouped-AdamW MLX path is doing the important things correctly:

- it runs reliably
- it trains without non-finite values
- it reduces loss meaningfully over both short and longer runs
- it keeps memory under control
- it sustains strong throughput over the reference session

That is enough to treat it as a legitimate prototype training path, not a toy demo.

### Translation Is Better Than Expected

The PyTorch-to-MLX translation layer is in very good shape.

The numerical agreement on the validation run is extremely tight, which removes a major source of ambiguity. That means future comparisons can start from comparable initialized weights rather than from unrelated random seeds.

### Apple-Native Primitives Are Already Paying Off

The parts that should matter most for an Apple-native path are already contributing:

- MLX fused attention
- MLX RoPE
- MLX tensor execution on Apple Silicon
- MLX memory behavior

There is no sign that the project is blocked on some missing Apple-native primitive for the current prototype scope.

## What Needs Improvement

### Optimizer Parity Is Still the Main Technical Gap

The fastest stable MLX path is not yet the closest path to nanochat’s current optimizer design.

Right now the practical winner is grouped AdamW, while the more Muon-like path is much slower. So the open problem is not “can we make Muon-like math run,” but rather “can we make a closer optimizer path run without giving away the Apple-native performance advantage.”

That is the single biggest technical gap still open in this phase.

### Dataset Parity Is Still Incomplete

The training runs that matter most so far still use repeated synthetic batches.

That was acceptable for the feasibility and health-check phase, but it is not enough for stronger claims about trainer parity. The dataset-backed mode exists in code, but it has not been validated because the required parquet shards are missing on this machine.

So the code path exists, but the evidence does not yet.

### Real Checkpoint Validation Is Still Missing

The translation path can initialize from real checkpoints, but there are no local trained checkpoints available right now.

That means the infrastructure is there, but the strongest form of practical validation has not been performed yet.

### Compiled MLX Training-Step Reuse Is Not Ready

An important attempted improvement was to push the longer training loop through `mx.compile` for a more purely Apple-native compiled execution path.

Current conclusion:

- compiled train-step execution appears fine for a first update
- repeated stateful optimizer updates through the compiled path did not continue advancing the training state across iterations in the way needed here
- the current stable long-run session therefore uses eager MLX execution, not compiled multi-step optimizer execution

This is important because it means “Apple-native” is already true, but “compiled stateful training loop” is not yet a reliable building block for this repo.

## Recommended Default Path Right Now

If a fresh agent needs to continue this phase without overthinking it, default to this:

- use MLX, not PyTorch + MPS, for native prototype work
- use [mlx_gpt_prototype.py](mlx_gpt_prototype.py) as the core model surface
- use grouped-AdamW matrix path, not the Muon-style experimental path, unless the task is explicitly optimizer-parity research
- use translated PyTorch reference initialization when comparing runs
- use [mlx_training_check.py](mlx_training_check.py) for health checks
- use [mlx_training_session.py](mlx_training_session.py) for longer runs
- treat dataset-backed and real-checkpoint validation as the next evidence-building tasks once assets are available

## Recommended Next Actions

If continuing this phase, do these in order:

1. validate dataset-backed MLX runs once local parquet shards are available
2. validate MLX initialization from a real trained nanochat checkpoint once one exists locally
3. investigate whether a closer optimizer path can preserve most of the grouped-AdamW performance advantage
4. only revisit compiled multi-step training if there is a clear MLX-supported pattern for stateful optimizer updates across repeated calls

## Explicit Non-Goals For The Next Agent

Do not drift into these unless the task explicitly changes:

- general cross-platform abstraction work
- hybrid Python plus native hotspot replacement work
- inference-only packaging work
- Azure, deployment, UI, or unrelated repo cleanup
- speculative large refactors before dataset-backed and checkpoint-backed validation are attempted

## Useful Commands

Reference benchmark:

```bash
export PYTHONPATH="$PWD"
.venv/bin/python dev/benchmark_mlx_reference.py --depth 32 --device-batch-size 2 --max-seq-len 1024 --steps 2 --warmup-steps 1 --init-from-pytorch-reference
```

Reference training health check:

```bash
export PYTHONPATH="$PWD"
.venv/bin/python dev/mlx_training_check.py --depth 32 --device-batch-size 2 --max-seq-len 1024 --steps 6 --warmup-steps 1 --init-from-pytorch-reference --matrix-optimizer adamw --progress
```

Longer stable training session:

```bash
export PYTHONPATH="$PWD"
.venv/bin/python dev/mlx_training_session.py --depth 32 --device-batch-size 2 --max-seq-len 1024 --steps 32 --warmup-steps 2 --init-from-pytorch-reference --matrix-optimizer adamw --progress-interval 4
```

Translation validation:

```bash
export PYTHONPATH="$PWD"
.venv/bin/python dev/compare_pytorch_mlx_translation.py
```

## Final Takeaway

The Apple Silicon native phase is already successful enough to justify itself.

The practical Apple-native MLX training path is real, measurably better than the frozen PyTorch + MPS baseline on this machine, and stable enough to run longer sessions.

The remaining work is no longer basic feasibility.

The remaining work is evidence and refinement:

- dataset-backed validation
- real-checkpoint validation
- deciding how much optimizer parity is worth if it costs too much performance