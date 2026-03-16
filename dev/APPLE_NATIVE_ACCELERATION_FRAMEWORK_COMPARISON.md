# Apple-Native Acceleration Framework Comparison

## Purpose

This memo compares the main Apple-native paths against nanochat's actual needs and ranks the candidates for prototype planning.

## Evaluation Criteria

The comparison is scored qualitatively against the requirements already established in the planning set:

1. transformer-training fit
2. inference and deployment relevance
3. autodiff and optimizer support
4. rewrite surface and implementation effort
5. integration burden with the existing Python repo
6. likely long-term maintainability

## Backend Decision Matrix

| Candidate | Training Fit | Inference Fit | Optimizer / Autodiff Fit | Implementation Effort | Integration Burden | Maintainability | Planning Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch + MPS only | Medium | Medium | High in current repo | Low | Low | High | Keep as baseline and fallback |
| MLX backend | High | Medium to High | High | Medium | Medium | Medium to High | Rank 1, first prototype candidate |
| Hybrid Apple-native hot path | Medium | Medium | Targeted, depends on hotspot | Medium | Medium to High | Medium | Rank 2, profiling-dependent fallback |
| Core ML path | Low for training | High | Low for trainer replacement | Medium for inference-only work | Medium | Medium | Rank 3, inference-only scope |
| Swift-native Metal / MPSGraph rewrite | Potentially High | High | High in theory | Very High | Very High | Low to Medium | Rank 4, defer |

## Candidate Notes

### 1. MLX

Why it ranks first:

- purpose-built for Apple Silicon
- supports autodiff, optimizers, and graph optimization
- materially better fit for transformer training experiments than Core ML
- much lower rewrite burden than a full Swift-native training stack

Main costs:

- introduces a second training stack
- requires deliberate choices on checkpoint compatibility and evaluation parity
- still requires model and optimizer re-expression outside PyTorch

Planning decision:

- MLX is the first prototype backend for Apple-native training evaluation

### 2. Hybrid hot-path acceleration

Why it remains viable:

- preserves the current Python and PyTorch control plane
- offers a path to accelerate one dominant hotspot without committing to a full backend split
- may be the best option if profiling isolates a narrow bottleneck

Why it is not rank 1:

- the current planning set does not yet isolate a single dominant hotspot strongly enough
- interop and debugging burden can become disproportionate if the performance win is small

Planning decision:

- keep as the second option if profiling points to a specific high-value hotspot

### 3. Core ML

Why it stays in scope:

- strong relevance for inference, deployment, and native-runtime packaging
- plausible export target if the repo later prioritizes on-device serving

Why it is not a trainer replacement candidate:

- weak fit for replacing the current training workflow
- not the right first move for answering the training-throughput question on this machine

Planning decision:

- keep Core ML in scope only as an inference or deployment path

### 4. Swift-native Metal or MPSGraph

Why it is technically interesting:

- highest theoretical control over kernel behavior and runtime design
- strongest long-term ceiling if Apple-native execution became the repo's center of gravity

Why it is deferred:

- highest rewrite surface by far
- longest time to first useful benchmark
- most severe tooling and maintenance split from the current repo

Planning decision:

- defer unless MLX and hybrid approaches fail to justify themselves

## Ranked Outcome

1. MLX-based training backend
2. Hybrid PyTorch + Apple-native hotspot acceleration
3. Core ML inference-only path
4. Full Swift-native Metal or MPSGraph rewrite

## First Prototype Candidate

The selected first prototype candidate for planning purposes is:

- MLX-based training backend

Rationale:

- best match for the actual architecture question
- best training-oriented Apple-native candidate
- materially lower rewrite cost than a pure Swift-native track
- preserves the option to stop early if the upside is not real

## Explicit Non-Recommendations For This Phase

- do not replace the current PyTorch + MPS path yet
- do not start with Core ML as a training answer
- do not begin a full Swift-native rewrite

## Related Documents

- [APPLE_NATIVE_ACCELERATION_ARCHITECTURE_SPEC.md](APPLE_NATIVE_ACCELERATION_ARCHITECTURE_SPEC.md)
- [APPLE_NATIVE_ACCELERATION_EXECUTION_PLAN.md](APPLE_NATIVE_ACCELERATION_EXECUTION_PLAN.md)
- [APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md](APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md)