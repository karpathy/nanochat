# Apple-Native Acceleration Task List

## How To Use This Document

This file converts the Apple-native acceleration planning documents into a checklist that can be marked off over time.

It is organized in two layers:

- planning-phase tasks needed to answer the architecture question
- implementation-phase tasks that should only begin after a go decision

The planning phase should be completed first.

## Phase 0: Baseline Freeze

### Reference Environment

- [x] Record the exact hardware profile for the reference M2 Ultra machine. See [APPLE_NATIVE_ACCELERATION_MACHINE_PROFILE.md](APPLE_NATIVE_ACCELERATION_MACHINE_PROFILE.md).
- [x] Record the exact macOS, Python, PyTorch, and dependency versions used for baseline comparison. See [APPLE_NATIVE_ACCELERATION_MACHINE_PROFILE.md](APPLE_NATIVE_ACCELERATION_MACHINE_PROFILE.md).
- [x] Record the current reference branch and baseline commit for comparison. See [APPLE_NATIVE_ACCELERATION_MACHINE_PROFILE.md](APPLE_NATIVE_ACCELERATION_MACHINE_PROFILE.md).

### Reference Workload

- [x] Freeze the default reference benchmark configuration: `depth=32`, `device-batch-size=2`, `max-seq-len=1024`. See [APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md](APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md).
- [x] Define the primary comparison metrics: tokens per second, driver memory, recommended memory fraction, implementation complexity. See [APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md](APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md).
- [x] Define which workloads are in scope for this phase: training, inference, or both. See [APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md](APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md).

### Baseline Artifacts

- [x] Create a baseline benchmark note for the current PyTorch + MPS path. See [APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md](APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md).
- [x] Create a baseline machine-profile note. See [APPLE_NATIVE_ACCELERATION_MACHINE_PROFILE.md](APPLE_NATIVE_ACCELERATION_MACHINE_PROFILE.md).
- [x] Confirm the baseline commands are repeatable on the reference machine. See [APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md](APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md).

## Phase 1: Profiling

### Throughput Profiling

- [x] Profile the current PyTorch + MPS training step on the reference workload. See [APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md](APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md).
- [x] Measure how much time is spent in forward pass, backward pass, and optimizer step. See [APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md](APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md).
- [x] Identify whether attention, optimizer, or runtime overhead is the dominant throughput bottleneck. See [APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md](APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md).

### Memory Profiling

- [x] Measure current MPS driver memory and recommended memory fraction on the reference workload. See [APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md](APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md).
- [x] Identify the largest contributors to memory pressure. See [APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md](APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md).
- [x] Determine whether the main limit is activations, optimizer state, parameters, or framework behavior. See [APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md](APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md).

### Profiling Deliverables

- [x] Write the profiling memo. See [APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md](APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md).
- [x] Produce a ranked bottleneck list. See [APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md](APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md).
- [x] Mark which bottlenecks are likely framework-specific versus model-specific. See [APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md](APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md).

## Phase 2: Framework Evaluation

### MLX Evaluation

- [x] Document MLX support for transformer-style training workloads relevant to nanochat. See [APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md](APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md).
- [x] Document MLX support for autodiff, optimizer behavior, and graph optimization. See [APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md](APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md).
- [x] Document expected integration burden for an MLX prototype. See [APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md](APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md).

### Core ML Evaluation

- [x] Document Core ML relevance for inference and deployment. See [APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md](APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md).
- [x] Document Core ML limitations for replacing nanochat training workflows. See [APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md](APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md).
- [x] Decide whether Core ML remains in scope only as an inference path. See [APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md](APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md).

### Swift-native Metal and MPSGraph Evaluation

- [x] Document what a Swift-native Metal or MPSGraph path would buy technically. See [APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md](APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md).
- [x] Document the likely rewrite surface required for a training path. See [APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md](APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md).
- [x] Decide whether this path should remain deferred unless earlier experiments fail. See [APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md](APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md).

### Comparison Deliverables

- [x] Write the framework comparison memo. See [APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md](APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md).
- [x] Create the backend decision matrix. See [APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md](APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md).
- [x] Rank the candidate paths. See [APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md](APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md).
- [x] Choose the first prototype candidate for planning purposes. See [APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md](APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md).

## Phase 3: Prototype Planning

### Prototype Scope

- [x] Define the minimal model surface required for the first prototype. See [APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md](APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md).
- [x] Decide whether the prototype targets training, inference, or both. See [APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md](APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md).
- [x] Define what is explicitly out of scope for the first prototype. See [APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md](APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md).

### Benchmark Definition

- [x] Define the exact prototype benchmark workload. See [APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md](APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md).
- [x] Define the acceptance thresholds for performance, memory, and complexity. See [APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md](APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md).
- [x] Define explicit kill criteria if the prototype underperforms. See [APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md](APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md).

### Prototype Planning Deliverables

- [x] Write the prototype scope memo. See [APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md](APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md).
- [x] Write the benchmark acceptance criteria. See [APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md](APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md).
- [x] Record the selected prototype backend. See [APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md](APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md).

## Phase 4: Integration Planning

### Shared Components

- [x] Decide whether tokenization remains shared with the existing Python path. See [APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md](APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md).
- [x] Decide whether dataset preparation remains shared. See [APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md](APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md).
- [x] Decide whether evaluation remains Python-first. See [APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md](APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md).

### Compatibility Planning

- [x] Assess checkpoint compatibility expectations. See [APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md](APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md).
- [x] Decide whether checkpoint translation is required or optional. See [APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md](APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md).
- [x] Identify backend-specific components that would need to exist separately. See [APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md](APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md).

### Integration Deliverables

- [x] Write the integration strategy note. See [APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md](APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md).
- [x] Write the compatibility and migration risk note. See [APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md](APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md).

## Phase 5: Decision

### Recommendation

- [x] Synthesize profiling, research, and prototype-planning outputs into one recommendation. See [APPLE_NATIVE_ACCELERATION_RECOMMENDATION.md](APPLE_NATIVE_ACCELERATION_RECOMMENDATION.md).
- [x] Decide whether the project should proceed with an Apple-native prototype. See [APPLE_NATIVE_ACCELERATION_RECOMMENDATION.md](APPLE_NATIVE_ACCELERATION_RECOMMENDATION.md).
- [x] Decide whether the recommended first implementation track is MLX, hybrid acceleration, inference-only native work, or no new backend. See [APPLE_NATIVE_ACCELERATION_RECOMMENDATION.md](APPLE_NATIVE_ACCELERATION_RECOMMENDATION.md).

### Decision Deliverables

- [x] Write the final recommendation memo. See [APPLE_NATIVE_ACCELERATION_RECOMMENDATION.md](APPLE_NATIVE_ACCELERATION_RECOMMENDATION.md).
- [x] Record the go or no-go decision. See [APPLE_NATIVE_ACCELERATION_RECOMMENDATION.md](APPLE_NATIVE_ACCELERATION_RECOMMENDATION.md).
- [ ] If no-go, define the PyTorch + MPS continuation plan.
- [x] If go, approve the implementation-phase task list below. See [APPLE_NATIVE_ACCELERATION_RECOMMENDATION.md](APPLE_NATIVE_ACCELERATION_RECOMMENDATION.md).

## Implementation Phase: MLX Prototype

Only start this section after a go decision.

### Setup

- [x] Define the MLX reference model configuration equivalent to the PyTorch baseline tier. See [APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md](APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md).
- [x] Define how tokenizer assets are consumed by the MLX prototype. See [APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md](APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md).
- [x] Define how the prototype will load representative input batches. See [APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md](APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md).

### Model Prototype

- [x] Implement the GPT forward pass in MLX. See [APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md](APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md).
- [x] Implement the backward pass in MLX. See [APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md](APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md).
- [x] Implement one optimizer step in MLX. See [APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md](APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md).

### Benchmarking

- [x] Benchmark the MLX prototype against the PyTorch + MPS baseline. See [APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md](APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md).
- [x] Record throughput delta. See [APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md](APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md).
- [x] Record memory delta. See [APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md](APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md).
- [x] Record implementation friction and missing features. See [APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md](APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md).

### Decision Gate

- [x] Decide whether the MLX prototype meaningfully beats the baseline. See [APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md](APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md).
- [x] Decide whether to expand the MLX track or stop it. See [APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md](APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md).

## Implementation Phase: Hybrid Hot-Path Prototype

Only start this section after a go decision and only if hybrid acceleration is selected.

### Hotspot Selection

- [ ] Select the single hotspot to replace first.
- [ ] Define the interface boundary between Python control flow and Apple-native execution.
- [ ] Define the benchmark used to justify the hotspot replacement.

### Prototype

- [ ] Implement the isolated accelerated hotspot.
- [ ] Benchmark the hotspot against the PyTorch baseline.
- [ ] Record maintenance and debugging burden.

### Decision Gate

- [ ] Decide whether the hotspot replacement justifies broader hybrid work.

## Implementation Phase: Inference-Only Native Path

Only start this section after a go decision and only if inference-native work is selected.

### Scope

- [ ] Define the target runtime scenario for native inference.
- [ ] Define how checkpoints or exported weights would reach the native runtime.
- [ ] Define latency and memory targets.

### Prototype

- [ ] Implement or prepare a native inference prototype.
- [ ] Benchmark latency and memory versus the Python inference path.
- [ ] Record deployment and packaging implications.

### Decision Gate

- [ ] Decide whether a native inference path should become a supported track.

## Completion Criteria

- [x] Baseline is frozen.
- [x] Profiling is complete.
- [x] Framework comparison is complete.
- [x] Prototype path is selected.
- [x] Integration planning is complete.
- [x] Final recommendation is documented.

## Related Documents

- [APPLE_NATIVE_ACCELERATION_PHASE_SPEC.md](APPLE_NATIVE_ACCELERATION_PHASE_SPEC.md)
- [APPLE_NATIVE_ACCELERATION_ARCHITECTURE_SPEC.md](APPLE_NATIVE_ACCELERATION_ARCHITECTURE_SPEC.md)
- [APPLE_NATIVE_ACCELERATION_EXECUTION_PLAN.md](APPLE_NATIVE_ACCELERATION_EXECUTION_PLAN.md)
- [APPLE_NATIVE_ACCELERATION_MACHINE_PROFILE.md](APPLE_NATIVE_ACCELERATION_MACHINE_PROFILE.md)
- [APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md](APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md)
- [APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md](APPLE_NATIVE_ACCELERATION_PROFILING_MEMO.md)
- [APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md](APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md)
- [APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md](APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md)
- [APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md](APPLE_NATIVE_ACCELERATION_INTEGRATION_STRATEGY.md)
- [APPLE_NATIVE_ACCELERATION_RECOMMENDATION.md](APPLE_NATIVE_ACCELERATION_RECOMMENDATION.md)
- [APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md](APPLE_NATIVE_ACCELERATION_MLX_PROTOTYPE.md)