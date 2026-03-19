# Code Review Recommendations

Date: 2026-03-19

## Scope

This review covered the core `nanochat/` package, selected entry scripts, the Apple-native / Swift / MLX workstream, and the current documentation set. The goal was not to redesign the repo, but to identify the highest-value changes for code quality, consistency, clean practices, and documentation accuracy.

Validation used during the review:

- `uv run python -m pytest -q` -> `24 passed, 10 skipped`
- `uv run pytest -q` -> import-path failures during collection in this environment
- `uv run python - <<'PY' ... flash_attn.flash_attn_func(..., causal=False) ... PY` -> confirmed SDPA fallback behaves causally
- `cd swift/NanochatMLXStub && xcodebuild -scheme NanochatMLXStub -destination 'platform=macOS' -derivedDataPath .derived build` -> fails on `Memory.resetPeakMemory()`

## Priority Findings

### High

1. `nanochat/flash_attention.py` does not preserve API parity on the SDPA fallback path.
   - Evidence: `_sdpa_attention` always applies causal-style masking and the fallback wrappers do not thread the public `causal` argument through the implementation.
   - References: `nanochat/flash_attention.py:69`, `nanochat/flash_attention.py:107`, `nanochat/flash_attention.py:131`
   - Impact: the module claims FA3-compatible behavior, but `causal=False` produces incorrect results outside Hopper/FA3 environments.
   - Recommendation: make `causal` an explicit input to `_sdpa_attention`, add non-causal parity tests for both `flash_attn_func` and `flash_attn_with_kvcache`, and treat FA3/SDPA parity as a contract.

2. `nanochat/engine.py` hardcodes KV-cache dtype from device type instead of actual compute dtype.
   - Evidence: cache dtype is set to `bfloat16` for CUDA and `float32` otherwise, even though `NANOCHAT_DTYPE` is configurable at the repo level.
   - References: `nanochat/engine.py:173`, `nanochat/common.py:13`
   - Impact: CUDA inference will drift once non-bf16 compute modes are used intentionally, and current tests will not catch it because they use CPU-only mocks.
   - Recommendation: derive cache dtype from model-owned state or a single authoritative config field, then add tests for `float32` and `float16` compute modes.

3. The Swift rebuild path is currently broken, but the Python wrappers assume it is a supported workflow. UPDATE: the rebuild path is now fixed, but a smoke check is still needed to prevent future regressions.
   - Evidence: a clean `xcodebuild` currently fails on `Memory.resetPeakMemory()`, while both Swift Python entrypoints call `xcodebuild` as the rebuild path.
   - References: `swift/NanochatMLXStub/Sources/NanochatMLXStub/main.swift:538`, `nanochat/swift_stub_engine.py:257`, `scripts/mlx_swift_stub.py:89`
   - Impact: the repo can appear healthy if a stale binary already exists, while clean rebuilds fail.
   - Recommendation: fix the Swift API call for the current `mlx-swift` version and add one smoke validation that exercises the rebuild path on a clean derived-data directory.

4. The documented test workflow is not reliable as written.
   - Evidence: `uv run python -m pytest -q` passes, but `uv run pytest -q` fails to import `nanochat` and `dev` in this environment.
   - References: `tests/test_engine.py:4`, `tests/test_attention_fallback.py:4`, `docs/APPLE_SILICON_PORTABILITY.md:94`
   - Impact: contributors and future agents can follow the documented command and get a broken result unrelated to the code under test.
   - Recommendation: standardize on one canonical command, preferably `uv run python -m pytest`, and add a CI smoke step that validates the documented invocation.

### Medium

5. Importing core modules has process-wide side effects.
   - Evidence: `nanochat.common` freezes `COMPUTE_DTYPE` at import time and configures root logging immediately; `checkpoint_manager` repeats the logging setup during import.
   - References: `nanochat/common.py:31`, `nanochat/common.py:62`, `nanochat/common.py:70`, `nanochat/checkpoint_manager.py:14`, `nanochat/checkpoint_manager.py:17`
   - Impact: behavior depends on import order, host logging can be overridden, and tests or tools cannot easily control initialization explicitly.
   - Recommendation: move dtype/logging setup behind explicit initialization helpers and keep imports side-effect free.

6. `scripts/chat_cli.py` assumes generation always returns at least one token.
   - Evidence: the code accesses `response_tokens[-1]` unconditionally after generation.
   - Reference: `scripts/chat_cli.py:137`
   - Impact: an empty generation result can crash the CLI with `IndexError` instead of failing gracefully.
   - Recommendation: guard the empty-response case explicitly and decide on a stable fallback behavior for both PyTorch and Swift backends.

7. Swift helper logic is duplicated across the codebase.
   - Evidence: repo-root resolution, build output paths, `xcodebuild`, output parsing, and token parsing appear in both `nanochat/swift_stub_engine.py` and `scripts/mlx_swift_stub.py`, with another partial variant in `dev/benchmark_swift_vs_python.py`.
   - References: `nanochat/swift_stub_engine.py:18`, `scripts/mlx_swift_stub.py:62`, `dev/benchmark_swift_vs_python.py:14`, `dev/benchmark_swift_vs_python.py:111`
   - Impact: behavior and documentation will continue to drift because the same integration rules are maintained in multiple places.
   - Recommendation: extract a shared Swift helper module and make the CLI, engine wrapper, and benchmarks consume it.

8. The Apple-native benchmark tooling contains machine-specific portability hacks.
   - Evidence: `dev/benchmark_swift_vs_python.py` falls back to `/Users/peternicholls/Dev/nanochatter` if repo-root detection fails.
   - Reference: `dev/benchmark_swift_vs_python.py:14`
   - Impact: the script is not portable to another machine or workspace checkout.
   - Recommendation: replace the fallback with the same repo-root helper used elsewhere and remove absolute workstation paths from committed code.

9. The execution sandbox documentation overstates what is enforced on macOS.
   - Evidence: the module-level docs say memory limits are enforced by default, but RLIMIT setup is skipped on Darwin.
   - References: `nanochat/execution.py:6`, `nanochat/execution.py:136`
   - Impact: developers may assume stronger isolation than they actually have on Apple hardware.
   - Recommendation: document platform capability differences explicitly and add tests around timeout and failure reporting for supported platforms.

10. Optional dependency boundaries are too broad in the tokenizer path.
    - Evidence: `rustbpe` is imported at module import time even though runtime tokenizer loading only uses the serialized tokenizer.
    - References: `nanochat/tokenizer.py:158`, `nanochat/tokenizer.py:390`
    - Impact: unrelated runtime codepaths inherit training-time dependency requirements and tests already work around this with import stubbing.
    - Recommendation: lazy-import training-only dependencies and separate tokenizer training concerns from inference/runtime loading.

11. Several direct runtime imports are not declared explicitly in `pyproject.toml`.
    - Evidence: the repo imports `filelock`, `jinja2`, `requests`, `pyarrow`, and `yaml`, but those packages are not listed in the project dependency set.
    - References: `pyproject.toml:7`, `nanochat/common.py:11`, `nanochat/core_eval.py:10`, `nanochat/dataset.py:13`, `nanochat/dataloader.py:20`, `scripts/base_eval.py:26`
    - Impact: fresh environments can work only by accident through transitive dependencies.
    - Recommendation: audit all direct imports, declare them explicitly, and add a fresh-environment smoke check for the canonical test command.

12. Some entrypoints still do CLI parsing and runtime initialization at import time.
    - Evidence: `base_train.py` prints banners, parses args, and initializes runtime state at module scope; `chat_web.py` parses args, configures logging, and initializes compute/device state at import time.
    - References: `scripts/base_train.py:37`, `scripts/base_train.py:41`, `scripts/base_train.py:80`, `scripts/chat_web.py:62`, `scripts/chat_web.py:73`, `scripts/chat_web.py:83`
    - Impact: reuse and testing are harder than necessary, and importing these modules has non-local side effects.
    - Recommendation: normalize entrypoints around `build_parser()` plus `main()` and keep import-time behavior minimal.

13. The web chat request validator is internally inconsistent.
    - Evidence: `chat_web.py` rejects any role other than `user` or `assistant`, but the raised error message says `system` is also valid.
    - Reference: `scripts/chat_web.py:186`
    - Impact: clients and future contributors cannot tell whether `system` messages are unsupported or just not implemented yet.
    - Recommendation: decide the intended API contract, align the validator and error text, and add request-validation tests.

14. `nanochat/report.py` hides operational failures with broad exception swallowing.
    - Evidence: helper functions fall back silently on command failures and timestamp parsing failures.
    - References: `nanochat/report.py:15`, `nanochat/report.py:25`
    - Impact: report generation can degrade silently, making environment problems harder to diagnose.
    - Recommendation: catch specific exceptions, surface warning details, and keep "best effort" behavior explicit instead of silent.

15. Checkpoint compatibility handling is functional but ad hoc.
    - Evidence: migration helpers mutate loaded structures in place, and checkpoint loading calls `init_weights()` just to rebuild buffers before assigning real state.
    - References: `nanochat/checkpoint_manager.py:23`, `nanochat/checkpoint_manager.py:30`, `nanochat/checkpoint_manager.py:104`
    - Impact: backward-compatibility logic is harder to reason about, test, and extend safely.
    - Recommendation: separate buffer initialization from parameter initialization, make migrations pure/versioned, and add explicit backward-compatibility tests.

16. Test coverage is concentrated in the newer inference / Apple-native work, not the broader core surface.
    - Evidence: there are targeted tests for engine, flash attention fallback, Swift stub routing, and MLX batching, but no direct unit coverage for `checkpoint_manager`, `execution`, `report`, `dataset`, or tokenizer conversation rendering.
    - References: `tests/`, compared against `nanochat/checkpoint_manager.py`, `nanochat/execution.py`, `nanochat/report.py`, `nanochat/dataset.py`, `nanochat/tokenizer.py`
    - Impact: fragile infrastructure and documentation-facing modules can regress without signal.
    - Recommendation: add small deterministic tests for checkpoint migration, execution timeout/error reporting, report generation, tokenizer rendering masks, and dataset discovery/error handling.

### Low

17. Runtime validation still relies heavily on `assert` in user-facing paths.
    - Evidence: external/input validation in engine, checkpoint loading, and tokenizer rendering uses `assert`.
    - References: `nanochat/engine.py:172`, `nanochat/checkpoint_manager.py:85`, `nanochat/tokenizer.py:287`, `nanochat/tokenizer.py:307`
    - Impact: those checks disappear under optimized Python.
    - Recommendation: reserve `assert` for internal invariants and use explicit exceptions for user/data validation.

18. Documentation indexes and file-structure snapshots are stale.
    - Evidence: the top-level README still shows `tests/test_engine.py` as the only test and omits newer scripts/docs; `docs/README.md` omits the March 2026 Apple acceleration research and plan documents.
    - References: `README.md:183`, `docs/README.md:7`
    - Impact: future planning agents and contributors will miss active material or form an outdated picture of the repo shape.
    - Recommendation: update the docs indexes after structural changes and prefer smaller maintained indexes over large static tree snapshots.

19. Some Apple-native docs now contradict the current repo state.
    - Evidence: the Swift CLI help still says `swift run` and claims GPU is not wired, while the README points users to `xcodebuild` plus `DYLD_FRAMEWORK_PATH`; the handoff docs also contain inconsistent statements about dataset-backed validation state.
    - References: `swift/NanochatMLXStub/Sources/NanochatMLXStub/main.swift:120`, `swift/NanochatMLXStub/README.md:12`, `docs/APPLE_NATIVE_ACCELERATION_HANDOFF.md:24`, `docs/APPLE_NATIVE_ACCELERATION_HANDOFF.md:129`
    - Impact: the Apple-native workstream is becoming harder to trust as planning input.
    - Recommendation: do a consistency pass that marks one current truth per topic and moves superseded notes into clearly historical sections.

## Recommended Workstreams

### Workstream 1: Correctness and API Parity

- Fix `flash_attention` SDPA parity with FA3, including `causal=False`.
- Give `Engine` a single authoritative source of compute dtype.
- Harden `chat_cli` against empty generation results.

### Workstream 2: Build and Integration Reliability

- Repair the Swift rebuild path against the current `mlx-swift` API.
- Add one reproducible rebuild smoke check.
- Consolidate Swift helper code into one shared module.
- Remove workstation-specific paths from committed scripts.

### Workstream 3: Initialization and Dependency Hygiene

- Remove import-time logging and dtype side effects.
- Split runtime tokenizer loading from tokenizer training dependencies.
- Replace broad `except:` and user-facing `assert` usage where they hide failures.

### Workstream 4: Documentation and Planning Hygiene

- Standardize one test command and document it consistently.
- Refresh README/docs indexes so they match the current tree.
- Reconcile Apple-native docs so they can serve as reliable planning inputs.

### Workstream 5: Test Expansion

- Add targeted tests for:
  - non-causal flash attention parity
  - configurable compute dtype in inference
  - Swift rebuild / clean-build health
  - checkpoint compatibility migrations
  - execution timeout and failure reporting
  - tokenizer conversation rendering and masks
  - report generation behavior under partial failure

## Suggested Planning Order

1. Fix correctness defects first: flash attention parity, engine dtype ownership, empty-generation handling.
2. Fix build and workflow trust next: Swift rebuild health and canonical test invocation.
3. Remove structural sources of drift: duplicated Swift helpers, import-time side effects, broad dependency loading.
4. Finish with documentation reconciliation and targeted test expansion.

## Overall Assessment

The repo is still relatively readable and the newer test work around inference / Apple-native paths is useful, but consistency is beginning to degrade at the boundaries: fallback behavior, build workflows, initialization side effects, and documentation now disagree in a few important places. The next planning pass should focus on restoring one clear source of truth per behavior and tightening the interfaces around the newer Swift/MLX integration work.
