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

The leading hypothesis is that the Polar Express / Newton-Schulz orthogonalization loop (5 rounds of GEMMs dispatched as separate MLX kernels) is the primary bottleneck — not the algorithmic GEMM count itself, but the repeated kernel launch and intermediate buffer allocation between iterations. This is the most targeted intervention available: a fused Metal compute shader or `MPSGraph` subgraph for the full Polar Express loop would keep intermediate values in GPU SRAM across iterations and allow the GPU compiler to see the full dependency chain.

**Steps:**

1. Profile the Muon-style MLX path to identify the dominant overhead source
   - Use Metal GPU Frame Capture to distinguish dispatch overhead from compute time
   - Is it the matrix orthogonalization (Polar Express / Newton-Schulz) kernel dispatch cost?
   - Is it the separate parameter-group iteration?
   - Is it framework overhead from the extra operations?
2. If dispatch overhead dominates: implement a fused Metal compute shader (or `MPSGraph` subgraph) for the Polar Express orthogonalization loop
   - Target: eliminate 5 rounds of separate kernel dispatch; keep intermediates in SRAM
   - This is the single most targeted intervention for closing the 3.5x gap
3. If the gap is algorithmic rather than dispatch: test focused Python-level improvements
   - Reduce Newton-Schulz iterations or precision
   - Test partial Muon (apply only to the largest weight matrices) to find the cost/benefit frontier
4. Benchmark partial-Muon paths against both pure AdamW and pure Muon

**Success criteria:**

- Identify via profiling whether the bottleneck is dispatch overhead or algorithmic compute
- If dispatch: fused Metal kernel reduces Muon path to within ~50% of AdamW throughput
- If algorithmic: find a partial-Muon configuration no more than ~50% slower than AdamW but closer to nanochat's optimizer behavior
- OR determine conclusively that the performance gap is fundamental and AdamW is the right MLX choice

**Evidence produced:** a clear optimizer recommendation for the MLX backend, with profiling data to support it

### 4. Incremental Swift Integration

**Priority: medium** (after items 1–2 are validated)

**Status:** not started. Analysis of the execution model shows that Python overhead is only meaningfully visible in specific hot paths; the training loop itself is GPU-bound and Swift translation there would add marginal value.

MLX's lazy-and-fused execution model means Python builds the computation DAG and `mx.eval()` triggers asynchronous GPU execution. Between evals, the actual tensor math runs in MLX's C++ runtime. A naive Python→Swift translation of the training loop buys little. Swift helps only where the Python runtime is actually visible.

Three components have genuine Python-visibility exposure:

**4a. Swift inference engine** (highest user-visible impact)

The per-token generation loop in `engine.py` calls the model once per token and then runs a Python state machine (`RowState`, `forced_tokens`, calculator parsing). At 2.8B params the GPU work per token is O(10–30ms); Python per-token overhead is O(1–5ms), serialised through the GIL even under batch generation. Replacing `engine.py` with a Swift binary using `mlx-swift` eliminates per-token Python overhead and GIL serialisation. The tokenizer encode/decode boundary is already C-backed (tiktoken), making the interface clean.

**4b. Data pipeline prefetch** (prerequisite for real-data training throughput)

The `build_dataset_backed_batch()` function does parquet read → BPE encode → BOS-pack in a single Python thread. At ~900 tok/s, the data pipeline has a narrow window to stay ahead of the GPU. A Swift worker using Foundation's structured concurrency and true parallelism (no GIL) could prefetch the next batch while the current one is on the GPU. The tokenizer encode calls are already multi-threaded in C; the Python glue in `_fill_row_from_docs` and the iteration loop is the bottleneck.

**4c. Training session orchestration** (low priority — monitor MLX-Swift compiled training support)

The outer training loop (`mlx_training_session.py`) is already GPU-bound. Swift translation here adds marginal value until MLX resolves compiled stateful training. If upstream MLX adds compiled stateful optimizer updates, `mlx-swift` would get that capability and rewriting the training session in Swift would become worthwhile.

**Steps (in order):**

1. Validate item 1 (dataset-backed training) before starting 4b
2. Build Swift inference engine using `mlx-swift`, with the checkpoint `.safetensors` file as the boundary with the Python training path
3. Build Swift data prefetch worker; connect to the MLX training loop via a shared memory or pipe boundary
4. Monitor MLX releases for compiled stateful training support before investing in 4c

**Explicit non-patterns to avoid:**

- Do not attempt a hybrid graph where Swift owns part of the MLX computation and Python owns the rest — integration overhead exceeds the gain
- Do not port the model definition to Swift prematurely — graph construction is fast relative to evaluation
- Do not abstract a general Swift backend — the seam is the `.safetensors` checkpoint file

**Success criteria:**

- Swift inference engine reduces per-token latency measurably versus the Python loop
- Swift data pipeline prefetch keeps GPU utilisation at ≥95% on real-data training runs
- No regressions on training throughput or loss trajectory

**Evidence produced:** user-visible latency improvement in inference; data pipeline no longer the bottleneck for real-data training

### 5. Compiled MLX Training Exploration

**Priority: low**

**Status:** compiled single-step works but stateful optimizer updates across repeated calls do not advance correctly.

This is worth revisiting only if MLX surfaces a supported pattern for it. Do not invest substantial time here until items 1-3 are resolved. Note: if the Swift training path (item 4c) is pursued, it would benefit from this resolution automatically — monitor both together.

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
- Hybrid Python + MLX graph co-ownership (integration overhead exceeds any gain)
- Inference-only packaging (Core ML, ONNX)
- Deployment, serving, or UI work
- Large refactors to the existing PyTorch path
- Premature full Swift rewrite of the training loop (GPU-bound; translation adds marginal value until compiled stateful training is available upstream)
- Blanket avoidance of Swift — targeted Swift components (inference engine, data pipeline) are in-scope once items 1-2 are validated

## Timeline Dependencies

- Item 1 (dataset validation) is blocked on local parquet data availability
- Item 2 (checkpoint validation) is blocked on having a trained checkpoint
- Item 3 (optimizer parity) can start independently; Metal kernel work within item 3 can start as soon as profiling identifies the bottleneck
- Item 4 (Swift integration) depends on item 1 being validated before component 4b; 4a (inference engine) can start independently once a checkpoint exists
- Item 5 (compiled training) depends on upstream MLX progress; monitor alongside item 4c

Recommended execution order: **1 → 2 → 3**, with 3 starting as soon as 1 is underway, and 4a starting in parallel once item 2 produces a checkpoint.
