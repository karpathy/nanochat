# Apple-Native Acceleration Execution Plan

## Document Purpose

This document turns the Apple-native acceleration architecture decision into an execution-phase plan without authorizing implementation.

It is designed to answer two questions:

1. What would we do, in what order, to answer the architecture question with evidence?
2. What tasks would define the first implementation phase if the architecture decision is approved?

## Scope Of This Plan

This plan covers:

- research work
- profiling work
- prototype planning
- benchmark planning
- integration planning
- decision checkpoints

This plan does not include actual implementation.

Checklist version of this plan:

- [APPLE_NATIVE_ACCELERATION_TASKLIST.md](APPLE_NATIVE_ACCELERATION_TASKLIST.md)

## Planning Assumptions

- The current PyTorch + MPS branch remains the reference baseline.
- The target machine is the current M2 Ultra Mac Studio.
- The first benchmark tier remains the current default local reference:
  - `depth=32`
  - `device-batch-size=2`
  - `max-seq-len=1024`
- MLX is the leading candidate for first prototype exploration.
- Core ML is evaluated mainly for inference and deployment relevance.
- Swift-native Metal or MPSGraph work is deferred unless a prior checkpoint justifies it.

## Workstreams

### Workstream 1: Baseline freeze and profiling

Objective:

- define the exact PyTorch MPS baseline and profile it cleanly

Questions to answer:

- where does time go in the representative run?
- where does memory go?
- what are the dominant blockers to further scale?
- how much overhead is framework-specific versus model-specific?

Outputs:

- baseline benchmark definition
- profiling memo
- prioritized bottleneck list

### Workstream 2: Apple-native framework evaluation

Objective:

- evaluate MLX, Core ML, and Swift-native paths against the actual repo requirements

Questions to answer:

- which stack best supports transformer training rather than just inference?
- which stack provides enough control over attention and optimizer behavior?
- which stack imposes the least destructive rewrite burden?

Outputs:

- framework comparison memo
- backend decision matrix

### Workstream 3: Prototype planning

Objective:

- define the minimum viable prototype that can answer the architecture question

Prototype scope should be limited to:

- GPT block stack
- forward pass
- backward pass
- one optimizer step
- one benchmarkable configuration

Outputs:

- prototype scope definition
- benchmark acceptance criteria

### Workstream 4: Integration feasibility

Objective:

- determine how a second backend would coexist with the current repo

Questions to answer:

- can tokenization stay shared?
- can datasets stay shared?
- can checkpoints be translated or must they diverge?
- should evaluation remain in Python even if training changes?

Outputs:

- integration strategy note
- compatibility risk list

## Milestones

### Milestone 0: Baseline freeze

Definition:

- declare the current PyTorch MPS branch as the comparison baseline

Exit criteria:

- benchmark commands and reference configs are fixed
- reference machine profile is documented

### Milestone 1: Framework research complete

Definition:

- document the strengths and weaknesses of MLX, Core ML, and Swift-native approaches

Exit criteria:

- one ranked shortlist
- one explicit recommendation for first prototype path

### Milestone 2: Prototype path selected

Definition:

- choose one backend for experimental implementation planning

Exit criteria:

- a single selected prototype direction
- all other directions explicitly deferred or rejected

### Milestone 3: Prototype benchmark plan approved

Definition:

- define exactly what the prototype must implement and how it will be measured

Exit criteria:

- scope, benchmark config, and acceptance thresholds are documented

### Milestone 4: Final architecture recommendation

Definition:

- finalize the go or no-go recommendation for implementation

Exit criteria:

- proceed to implementation plan
or
- stop the native backend track and continue PyTorch + MPS optimization only

## Task Breakdown For This Planning Phase

### Phase 0: Baseline freeze

Tasks:

- lock the reference benchmark workload for PyTorch MPS
- document the exact hardware and software environment
- define primary metrics:
  - tokens per second
  - driver memory
  - recommended memory fraction
  - implementation complexity

Deliverables:

- baseline benchmark note
- metrics definition note

### Phase 1: Profiling

Tasks:

- profile the current default MPS training tier
- identify dominant time hotspots
- identify dominant memory consumers
- classify bottlenecks as:
  - attention-related
  - optimizer-related
  - framework runtime-related
  - Python overhead-related

Deliverables:

- profiling memo
- ranked bottleneck list

### Phase 2: Framework evaluation

Tasks:

- map nanochat requirements against MLX capabilities
- map nanochat requirements against Core ML capabilities
- map nanochat requirements against Swift-native Metal or MPSGraph capabilities
- score each candidate on:
  - training fit
  - inference fit
  - implementation effort
  - integration burden
  - long-term maintainability

Deliverables:

- framework comparison memo
- backend decision matrix

### Phase 3: Prototype planning

Tasks:

- choose the first prototype backend
- define what the prototype must and must not include
- document explicit kill criteria if the prototype underperforms

Deliverables:

- prototype selection record
- prototype scope memo

### Phase 4: Integration planning

Tasks:

- decide what existing components remain shared
- decide what new backend-specific components would be required
- evaluate checkpoint translation feasibility
- determine whether evaluation should remain Python-first

Deliverables:

- integration strategy note
- compatibility and migration risk note

### Phase 5: Final recommendation

Tasks:

- synthesize the findings into a go or no-go recommendation
- if go, define the implementation-phase roadmap
- if no-go, define the PyTorch MPS continuation plan

Deliverables:

- final recommendation memo
- implementation roadmap or stop memo

## Implementation-Phase Task Breakdown

This section defines the next-phase tasks that would be used only if the architecture decision approves implementation.

### Implementation Track A: MLX prototype

Tasks:

- define MLX model configuration equivalent to the reference nanochat tier
- implement tokenizer and input pipeline reuse strategy
- implement GPT forward pass in MLX
- implement backward pass and optimizer step in MLX
- benchmark a single representative training step against PyTorch MPS baseline
- record throughput, memory, and engineering friction

Definition of done:

- one benchmarkable MLX prototype exists
- results are directly comparable to baseline

### Implementation Track B: Hybrid hot-path prototype

Tasks:

- identify one hotspot worth replacing
- define boundary between Python control flow and Apple-native execution
- implement one isolated accelerated path
- benchmark it against baseline
- evaluate integration and maintenance burden

Definition of done:

- one isolated hotspot prototype exists
- the benefit and complexity are quantified

### Implementation Track C: Inference-only native path

Tasks:

- define the target runtime shape for local inference
- evaluate export or translation path from current checkpoints
- benchmark latency and memory versus Python inference path
- determine how a native runtime would fit the repo and tooling

Definition of done:

- one clear decision on whether inference-only native work is justified

## Deliverables Matrix

### Research deliverables

- baseline benchmark note
- profiling memo
- framework comparison memo
- backend decision matrix

### Planning deliverables

- prototype scope memo
- integration strategy note
- compatibility risk note
- implementation roadmap draft

### Decision deliverables

- final recommendation memo
- go or no-go decision record

## Dependencies

- stable PyTorch MPS baseline measurements
- repeatable benchmark commands on the M2 Ultra
- enough documentation from Apple-native frameworks to avoid speculative design
- agreement that this phase remains planning and evaluation only

## Risks And Mitigations

### Risk 1: Rewrite trap

The project could drift into a full backend rewrite before proving upside.

Mitigation:

- require prototype benchmark evidence before any expanded scope

### Risk 2: Tooling fragmentation

Two backends may create too much maintenance overhead.

Mitigation:

- keep tokenizer, dataset, reporting, and evaluation shared whenever possible

### Risk 3: Apple-native upside may be smaller than expected

Even if MLX or Metal is cleaner, the absolute performance gain may not justify the effort.

Mitigation:

- make performance and memory deltas explicit gate criteria

### Risk 4: Inference and training needs may diverge

The best inference stack may not be the best training stack.

Mitigation:

- allow the outcome to recommend different stacks for training and inference

### Risk 5: Benchmark mismatch

A prototype may look good on synthetic workloads but fail on realistic training flows.

Mitigation:

- require at least one benchmark close to the current training path

## Exit Criteria For The Planning Phase

The planning phase is complete when:

1. The baseline is frozen.
2. A ranked framework comparison exists.
3. One prototype path is selected.
4. The prototype benchmark scope is explicitly defined.
5. A final go or no-go decision framework is documented.

## Recommended Immediate Next Step

If this planning set is accepted, the next non-coding activity should be:

1. Create a benchmark baseline note for the current `depth=32, batch=2, seq=1024` PyTorch MPS tier.
2. Create the framework comparison memo with MLX as the primary candidate under evaluation.
3. Use those two artifacts to decide whether implementation planning should target MLX first or a smaller hybrid hotspot experiment.

## Related Documents

- [APPLE_NATIVE_ACCELERATION_PHASE_SPEC.md](APPLE_NATIVE_ACCELERATION_PHASE_SPEC.md)
- [APPLE_NATIVE_ACCELERATION_ARCHITECTURE_SPEC.md](APPLE_NATIVE_ACCELERATION_ARCHITECTURE_SPEC.md)
- [APPLE_NATIVE_ACCELERATION_TASKLIST.md](APPLE_NATIVE_ACCELERATION_TASKLIST.md)
- [APPLE_SILICON_PORTABILITY.md](APPLE_SILICON_PORTABILITY.md)
- [M2_ULTRA_SCALING.md](M2_ULTRA_SCALING.md)