# MLX Apple-Native Acceleration: Next Phase Plan

## Context

The MLX feasibility phase is complete. The key outcome:

- MLX training is **5.5x faster** than PyTorch + MPS on the reference workload
- MLX peak memory is **lower** than the PyTorch baseline
- The prototype is stable over longer sessions (32+ steps, 72s wall time)
- Weight translation from PyTorch is numerically validated

This plan covers the next phase: turning the validated prototype into a credible training path by closing the remaining evidence and parity gaps.

See [APPLE_NATIVE_ACCELERATION_HANDOFF.md](APPLE_NATIVE_ACCELERATION_HANDOFF.md) for full current state.

## Goal

Produce enough evidence to decide whether the MLX path should become the permanent second backend for nanochat on Apple Silicon. The decision needs three things:

1. Training on real data, not just synthetic batches
2. Initialization from real checkpoints, not just fresh weights
3. A clear optimizer recommendation that does not give back the performance win

---

## Phase 1 — Real-Data Training Validation

**Priority: highest.** Code exists ([mlx_input_batches.py](../dev/mlx_input_batches.py)) but has not been exercised because local parquet shards are missing.

### Story 1.1 — Bootstrap data and tokenizer

- [X] Download a minimal set of parquet shards: `python -m nanochat.dataset -n 32`
- [X] Train the tokenizer if missing: `python -m scripts.tok_train --max-chars=500000000`

### Story 1.2 — Run the dataset-backed training check

- [X] Run the MLX training check with `--input-mode dataset`:
  ```bash
  export PYTHONPATH="$PWD"
  .venv/bin/python dev/mlx_training_check.py \
    --depth 32 --device-batch-size 2 --max-seq-len 1024 \
    --steps 6 --warmup-steps 1 \
    --init-from-pytorch-reference \
    --input-mode dataset --progress
  ```
- [X] Confirm the check passes: no crashes, finite loss at every step

### Story 1.3 — Validate throughput and loss trajectory

- [X] Run a longer dataset-backed session (64+ steps)
- [X] Compare loss trajectory against the synthetic-batch baseline
- [X] Confirm throughput does not regress more than ~10% versus synthetic batches

**Done when:** training check passes with real data; throughput regression ≤10%; loss is decreasing, finite, and stable.

---

## Phase 2 — Checkpoint Initialization Validation

**Priority: high.** Translation infrastructure exists ([mlx_checkpoint_translation.py](../dev/mlx_checkpoint_translation.py)) but no trained checkpoint is available locally.

### Story 2.1 — Obtain a trained checkpoint

- [ ] Run a short PyTorch training session to produce a checkpoint, or locate an existing one

### Story 2.2 — Translate and validate

- [ ] Translate the checkpoint to MLX format using the existing translation path
- [ ] Validate logit agreement between the PyTorch original and the translated MLX model (target: max diff ~1e-7)

### Story 2.3 — Continue training from the translated checkpoint

- [ ] Run a short MLX training session starting from the translated weights
- [ ] Confirm loss continues to decrease; verify no non-finite values or instability

**Done when:** logit agreement is tight (~1e-7); MLX training from the checkpoint reduces loss; no instability.

---

## Phase 3 — Optimizer Parity Investigation

**Priority: medium. Can start independently of Phases 1–2.**

Grouped AdamW is the current practical winner (~900 tok/s). The Muon-style path works but is ~3.5x slower (~258 tok/s). The leading hypothesis is that repeated kernel dispatch in the Polar Express / Newton-Schulz loop (5 rounds of GEMMs dispatched as separate MLX kernels) is the bottleneck — not the GEMM count itself, but the launch overhead and intermediate buffer allocation between iterations.

### Story 3.1 — Profile to identify the bottleneck

- [ ] Use Metal GPU Frame Capture to profile the Muon-style MLX path
- [ ] Determine the dominant source: Polar Express dispatch overhead, parameter-group iteration, or framework overhead

### Story 3.2a — If dispatch overhead dominates: fuse the Polar Express loop

- [ ] Implement a fused Metal compute shader (or `MPSGraph` subgraph) for the full Newton-Schulz orthogonalization loop
- [ ] Target: eliminate 5 separate kernel dispatches and keep intermediates in GPU SRAM
- [ ] Benchmark fused vs. unfused Muon and vs. AdamW

### Story 3.2b — If the bottleneck is algorithmic: tune at the Python level

- [ ] Test reduced Newton-Schulz iterations or lower precision
- [ ] Test partial Muon (apply only to the largest weight matrices) to map the cost/benefit frontier
- [ ] Benchmark partial-Muon configurations against AdamW and full Muon

**Decision point:** if neither 3.2a nor 3.2b closes the gap to within ~50% of AdamW throughput, document the gap and recommend AdamW as the right MLX optimizer choice.

**Done when:** profiling identifies the dominant bottleneck; a configuration is found that is ≤50% slower than AdamW and closer to nanochat's optimizer behavior, OR AdamW is confirmed as the correct choice with data to support it.

---

## Phase 4 — Targeted Swift Integration

**Priority: medium. Start after Phases 1–2 are validated.**

MLX's lazy-and-fused execution model means Python builds the computation DAG and `mx.eval()` triggers asynchronous GPU execution — so a naive Python→Swift rewrite of the training loop buys little. Swift helps only where the Python runtime is actually visible in the hot path. Three components qualify.

> **Dependency note:** Story 4b requires Phase 1 to be done. Story 4a can start as soon as Phase 2 produces a checkpoint. Story 4c is low priority and depends on upstream MLX progress.

### Story 4a — Swift inference engine *(can start after Phase 2)*

The per-token loop in `engine.py` calls the model once per token and then runs a Python state machine (`RowState`, `forced_tokens`, calculator parsing). At 2.8B params, GPU work per token is O(10–30ms); Python per-token overhead is O(1–5ms), serialised through the GIL under batch generation.

- [ ] Build a Swift inference binary using `mlx-swift`; use the `.safetensors` checkpoint file as the boundary with the Python training path
- [ ] Replace the Python per-token loop in `engine.py` with the Swift binary
- [ ] Measure per-token latency vs. the Python baseline

### Story 4b — Swift data prefetch worker *(requires Phase 1)*

`build_dataset_backed_batch()` does parquet read → BPE encode → BOS-pack in a single Python thread. At ~900 tok/s the data pipeline has a narrow window to stay ahead of the GPU. The Python glue in `_fill_row_from_docs` and the iteration loop is the bottleneck (the tokenizer encode calls are already multi-threaded in C).

- [ ] Build a Swift worker using Foundation structured concurrency (no GIL)
- [ ] Replace the Python glue layer in `_fill_row_from_docs` / `build_dataset_backed_batch()`
- [ ] Connect to the MLX training loop via shared memory or pipe boundary
- [ ] Verify GPU utilisation stays ≥95% on real-data training runs

### Story 4c — Training session orchestration *(low priority — monitor upstream)*

The outer training loop (`mlx_training_session.py`) is GPU-bound. Swift translation adds marginal value until MLX supports compiled stateful optimizer updates.

- [ ] Monitor MLX releases for compiled stateful training support
- [ ] If a supported pattern emerges, evaluate rewriting the training session in `mlx-swift`

**Non-patterns to avoid:**
- No hybrid Python+Swift graph ownership — integration overhead exceeds the gain
- No premature model-definition port to Swift — graph construction is fast relative to evaluation
- No general multi-backend Swift abstraction — the seam is the `.safetensors` file

**Done when:** Swift inference engine shows measurable per-token latency reduction; Swift data pipeline keeps GPU utilisation ≥95% on real-data runs; no regressions on throughput or loss.

---

## Phase 5 — Compiled MLX Training

**Priority: low. Do not invest time here until Phases 1–3 are done.**

Compiled single-step training works today, but stateful optimizer updates across repeated calls do not advance correctly. This is worth revisiting only if MLX surfaces a supported pattern. If the Swift training path (Story 4c) is pursued, it benefits from this resolution automatically — monitor both together.

- [ ] Monitor MLX releases for compiled stateful training support
- [ ] If a supported pattern emerges, test it on the reference workload and compare eager vs. compiled throughput

---

## Decision Gate

After Phases 1–3, answer:

> **Should the MLX path become the permanent, maintained second backend for nanochat on Apple Silicon?**

| Decision | Criteria |
|---|---|
| **Yes — make it permanent** | Real-data training works; checkpoint init works; >3x throughput advantage survives real data; optimizer gap is acceptable |
| **Keep as experimental** | Real-data issues exist but are not fundamental blockers; optimizer gap is manageable |
| **Stop the MLX track** | Fundamental blockers emerge; performance advantage disappears with real data and real optimization |

---

## Execution Order

```
Phase 1 (real-data training) ──────────────────────────────► Story 4b (Swift data pipeline)
         │
         └──► Phase 2 (checkpoint validation) ─────────────► Story 4a (Swift inference)
                       │
                       └──► Phase 3 (optimizer parity) ─────► [can start independently;
                                                                Metal kernel work starts
                                                                after profiling]

Phase 5 (compiled training) ── monitor upstream; low investment until Phases 1–3 done
```

Recommended order: **Phase 1 → Phase 2 → Phase 3**, with Phase 3 starting as soon as Phase 1 is underway, and Story 4a starting in parallel once Phase 2 produces a checkpoint.

---

## Explicit Non-Goals

- General cross-platform backend abstraction
- Hybrid Python + MLX graph co-ownership
- Inference-only packaging (Core ML, ONNX)
- Deployment, serving, or UI work
- Large refactors to the existing PyTorch path
- Premature full Swift rewrite of the training loop (GPU-bound; marginal value until compiled stateful training is available upstream)
- Blanket avoidance of Swift — targeted components (Stories 4a, 4b) are in-scope once Phases 1–2 are validated
