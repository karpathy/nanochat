# Apple-Native Acceleration Integration Strategy

## Purpose

This note defines how an Apple-native prototype should coexist with the current Python repo during planning and early experimentation.

## Shared Components That Should Remain Shared

The prototype should reuse the existing Python path wherever that does not distort the backend evaluation.

Shared by default:

- tokenizer assets and tokenization workflow
- dataset preparation and local data bootstrap process
- evaluation entrypoints and reporting workflows
- benchmark definitions and comparison metrics

Planning decision:

- evaluation remains Python-first even if a second training backend is prototyped

## Backend-Specific Components

The following components are expected to become backend-specific in an MLX prototype:

- model definition and layer implementation
- autodiff and optimizer execution
- runtime-specific memory and performance instrumentation
- any backend-native benchmark harness needed to run the prototype cleanly

## Checkpoint Compatibility Expectation

Checkpoint compatibility should be treated as desirable but optional for the first prototype.

Recommended planning stance:

- weight translation is useful if it can be done with modest effort
- optimizer-state portability is optional for the first benchmark gate
- full parity with existing checkpoint metadata should not block the initial benchmark prototype

## Translation Strategy

If checkpoint reuse is needed, prefer a narrow adapter layer instead of forcing identical internal representations.

Recommended order:

1. start with random-init or minimally bootstrapped prototype validation
2. add weight translation only if it improves comparability enough to matter
3. defer optimizer-state translation unless the prototype survives the first benchmark gate

## Compatibility And Migration Risks

Primary risks:

1. tensor naming, layout, or dtype mismatches across backends
2. optimizer-state incompatibility between PyTorch and MLX
3. attention-kernel or numerical-behavior differences that weaken apples-to-apples comparisons
4. duplicated maintenance burden if backend boundaries are not kept narrow

Mitigations:

1. keep tokenizer, dataset, eval, and reporting shared
2. avoid promising full checkpoint parity in the first prototype phase
3. require benchmark evidence before expanding backend scope
4. keep the PyTorch + MPS path fully intact as the reference and fallback

## Integration Rule For This Phase

The Apple-native track is a contained experiment.

That means:

- no repo-wide forced abstraction layer before evidence exists
- no destabilization of existing PyTorch training scripts
- no assumption that the prototype becomes a permanent second backend unless it clears the benchmark gate

## Related Documents

- [APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md](APPLE_NATIVE_ACCELERATION_FRAMEWORK_COMPARISON.md)
- [APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md](APPLE_NATIVE_ACCELERATION_PROTOTYPE_SCOPE.md)
- [APPLE_NATIVE_ACCELERATION_ARCHITECTURE_SPEC.md](APPLE_NATIVE_ACCELERATION_ARCHITECTURE_SPEC.md)