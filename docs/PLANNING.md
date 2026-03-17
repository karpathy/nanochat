# MLX Apple-Native Acceleration: Next Phase Plan

## Context

The MLX feasibility phase is complete. The key outcome:

- MLX training is **5.5x faster** than PyTorch + MPS on the reference workload
- MLX peak memory is **lower** than the PyTorch baseline
- The prototype is stable over longer sessions (32+ steps, 72s wall time)
- Weight translation from PyTorch is numerically validated

This plan covers the next phase: turning the validated prototype into a credible training path by closing the remaining evidence and parity gaps.

See [APPLE_NATIVE_ACCELERATION_HANDOFF.md](APPLE_NATIVE_ACCELERATION_HANDOFF.md) for full current state.

## Phase Goal

Produce enough evidence to decide whether the MLX path should become a permanent second backend for nanochat on Apple Silicon.

The decision requires:

1. training on real data, not just synthetic batches
2. initialization from real checkpoints, not just fresh weights
3. a clearer picture of how much optimizer parity is achievable without giving back the performance win

## Work Items

### 1. Dataset-Backed MLX Training Validation

**Priority: highest**

**Status:** code exists in [mlx_input_batches.py](../dev/mlx_input_batches.py) but has not been exercised because local parquet shards are missing.

**Steps:**

1. Download parquet shards locally (use the existing nanochat bootstrap: `python -m nanochat.dataset -n 32` for a minimal set)
2. Train the tokenizer if missing (`python -m scripts.tok_train --max-chars=500000000`)
3. Run the MLX training check with dataset-backed input:
   ```bash
   export PYTHONPATH="$PWD"
   .venv/bin/python dev/mlx_training_check.py \
     --depth 32 --device-batch-size 2 --max-seq-len 1024 \
     --steps 6 --warmup-steps 1 \
     --init-from-pytorch-reference \
     --input-mode dataset --progress
   ```
4. Run a longer dataset-backed training session and compare loss trajectory against the synthetic-batch baseline
5. Confirm throughput and memory are comparable to synthetic-batch runs

**Success criteria:**

- Training check passes with real data
- Throughput does not regress more than ~10% versus synthetic batches
- Loss trajectory is reasonable (decreasing, finite, stable)

**Evidence produced:** first credible MLX training run on real nanochat data

### 2. Real-Checkpoint Initialization Validation

**Priority: high**

**Status:** checkpoint translation infrastructure exists in [mlx_checkpoint_translation.py](../dev/mlx_checkpoint_translation.py) but no local trained checkpoints are available.

**Steps:**

1. Obtain or produce a trained nanochat checkpoint (run a short PyTorch training session if needed)
2. Translate the checkpoint to MLX format using the existing translation path
3. Validate that the translated model produces comparable logits and loss to the PyTorch original
4. Run a short MLX training session starting from the translated checkpoint
5. Verify that the MLX path continues training coherently from the translated state

**Success criteria:**

- Logit agreement between PyTorch and translated MLX model is tight (similar to the fresh-weight validation: max diff ~1e-7)
- MLX training from the translated checkpoint reduces loss further
- No non-finite values or training instability

**Evidence produced:** proof that existing trained models can seed MLX training

### 3. Optimizer Parity Investigation

**Priority: medium**

**Status:** grouped AdamW is the practical winner (~900 tok/s). Muon-style path works but is ~3.5x slower (~258 tok/s).

This is the main open technical question: nanochat uses a MuonAdamW split optimizer. The MLX prototype uses grouped AdamW because the Muon-like path is too slow. Understanding where the Muon overhead comes from and whether it can be reduced determines how closely the MLX trainer can match nanochat's optimizer behavior.

**Steps:**

1. Profile the Muon-style MLX path to identify the dominant overhead source
   - Is it the matrix orthogonalization (Newton-Schulz / SVD)?
   - Is it the separate parameter-group iteration?
   - Is it framework overhead from the extra operations?
2. Test focused improvements:
   - Reduce Newton-Schulz iterations or precision
   - Explore whether MLX has efficient SVD or matrix operations that could replace the current path
   - Test partial Muon (apply only to the largest weight matrices) to find the cost/benefit frontier
3. Benchmark partial-Muon paths against both pure AdamW and pure Muon

**Success criteria:**

- Identify the specific bottleneck in the Muon path
- Find a partial-Muon configuration that is no more than ~50% slower than AdamW but closer to nanochat's optimizer behavior
- OR determine conclusively that the performance gap is fundamental and AdamW is the right MLX choice

**Evidence produced:** a clear optimizer recommendation for the MLX backend

### 4. Compiled MLX Training Exploration

**Priority: low**

**Status:** compiled single-step works but stateful optimizer updates across repeated calls do not advance correctly.

This is worth revisiting only if MLX surfaces a supported pattern for it. Do not invest substantial time here until items 1-3 are resolved.

**Steps:**

1. Monitor MLX releases for compiled stateful training support
2. If a supported pattern emerges, test it on the reference workload
3. Compare eager vs compiled throughput

**Success criteria:**

- Compiled multi-step training matches eager loss trajectory
- Measurable throughput improvement from compilation

**Evidence produced:** whether compilation is a practical training accelerator for this codebase

## Decision Gate

After items 1-3, the project should be able to answer:

> Should the MLX path become a permanent, maintained second backend for nanochat on Apple Silicon?

**Criteria for "yes, make it permanent":**

- Dataset-backed training works and produces reasonable results
- Checkpoint initialization from real models works
- The throughput advantage survives real-data training (remains >3x vs PyTorch + MPS)
- Either AdamW is sufficient for Apple Silicon use, OR a partial-Muon path closes the optimizer gap acceptably

**Criteria for "keep as experimental only":**

- Real-data training exposes problems not visible with synthetic batches
- Checkpoint translation has unresolvable numerical issues
- Optimizer parity gap is too large to produce comparable model quality

**Criteria for "stop the MLX track":**

- Fundamental blockers emerge that make MLX training impractical for the actual model
- The performance advantage disappears with real data and real optimization

## Explicit Non-Goals

Do not pursue these during this phase:

- General cross-platform backend abstraction
- Hybrid Python + native hotspot replacement
- Inference-only packaging (Core ML, ONNX)
- Deployment, serving, or UI work
- Large refactors to the existing PyTorch path
- Swift-native Metal rewrites

## Timeline Dependencies

- Item 1 (dataset validation) is blocked on local parquet data availability
- Item 2 (checkpoint validation) is blocked on having a trained checkpoint
- Item 3 (optimizer parity) can start independently
- Item 4 (compiled training) depends on upstream MLX progress

Recommended execution order: **1 → 2 → 3**, with 3 starting as soon as 1 is underway.
