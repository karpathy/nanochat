# Apple-Native Acceleration Phase

## Purpose

This file is now the index for the Apple-native acceleration planning set.

The original combined phase document has been split into:

- [APPLE_NATIVE_ACCELERATION_ARCHITECTURE_SPEC.md](APPLE_NATIVE_ACCELERATION_ARCHITECTURE_SPEC.md)
- [APPLE_NATIVE_ACCELERATION_EXECUTION_PLAN.md](APPLE_NATIVE_ACCELERATION_EXECUTION_PLAN.md)
- [APPLE_NATIVE_ACCELERATION_TASKLIST.md](APPLE_NATIVE_ACCELERATION_TASKLIST.md)

## What Each Document Does

### Architecture spec

The architecture spec defines:

- the problem statement
- the current baseline
- goals and non-goals
- the option space
- research-informed tradeoffs
- the recommended technical direction
- success metrics and decision rules

### Execution plan

The execution plan defines:

- workstreams
- milestones
- phase-by-phase task breakdown
- deliverables
- dependencies
- risks and mitigations
- implementation-phase tasks without authorizing implementation

### Task list

The task list defines:

- checklist-style tasks that can be marked complete
- planning-phase tasks first
- implementation-phase tasks gated behind a go decision

## Current Recommendation

The current planning recommendation remains:

1. Keep PyTorch + MPS as the working baseline.
2. Evaluate MLX first as the primary Apple-native training candidate.
3. Treat Core ML as primarily an inference and deployment path.
4. Defer any full Swift + Metal or MPSGraph rewrite unless prototype evidence justifies it.

## Related Documents

- [APPLE_SILICON_PORTABILITY.md](APPLE_SILICON_PORTABILITY.md)
- [M2_ULTRA_SCALING.md](M2_ULTRA_SCALING.md)