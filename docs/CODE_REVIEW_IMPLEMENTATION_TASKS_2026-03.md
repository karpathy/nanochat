# Code Review Implementation Tasks (March 2026)

## Purpose

Convert [CODE_REVIEW_RECOMMENDATIONS_2026-03.md](CODE_REVIEW_RECOMMENDATIONS_2026-03.md) into a definitive, phase-based execution plan with sprint-sized, checkable tasks.

## Planning Priorities

1. Fix correctness and workflow trust before cleanup/refactor work.
2. Stabilize the Python/Swift/MLX seams before widening scope.
3. Prefer changes that reduce drift between runtime behavior, tests, and docs.
4. End each sprint in a shippable, reviewable state.

## Phase Structure

- Phase 1: Correctness and workflow trust
- Phase 2: Swift/build/integration hardening
- Phase 3: Initialization and dependency hygiene
- Phase 4: Documentation and regression protection

## Sprint Structure

- Sprint 1: Fallback correctness
- Sprint 2: Test workflow and runtime safety
- Sprint 3: Swift rebuild and portability
- Sprint 4: Shared integration surfaces and entrypoint cleanup
- Sprint 5: Dependency, initialization, and compatibility hygiene
- Sprint 6: Docs reconciliation and regression backfill

## Global Dependency Order

- Sprint 1 blocks the rest of the plan.
- Sprint 2 should complete before broader refactors land.
- Sprint 3 should complete before further Swift routing/productization work.
- Sprint 5 should complete before declaring the cleanup phase done.
- Sprint 6 closes the plan.

---

## Phase 1: Correctness and Workflow Trust

### Sprint 1: Fallback Correctness

**Goal:** fix the highest-risk behavioral issues in inference and attention fallback logic.

- [ ] Fix `nanochat/flash_attention.py` so the SDPA fallback honors `causal=False`.
- [ ] Thread `causal` explicitly through the internal SDPA helper and both public wrappers.
- [ ] Add parity tests for non-causal `flash_attn_func`.
- [ ] Add parity tests for non-causal `flash_attn_with_kvcache`.
- [ ] Replace the KV-cache dtype hack in `nanochat/engine.py` with one authoritative dtype source.
- [ ] Add coverage for non-default compute dtype behavior.
- [ ] Harden `scripts/chat_cli.py` against empty generation results.

**Acceptance criteria**

- [ ] FA3/SDPA parity tests cover both causal and non-causal paths.
- [ ] `Engine` no longer infers cache dtype ad hoc from `device.type`.
- [ ] `chat_cli` handles empty output safely.
- [ ] `uv run python -m pytest -q` stays green.

### Sprint 2: Test Workflow and Runtime Safety

**Goal:** make one canonical contributor/test workflow reliable from repo root.

- [ ] Choose the canonical test command.
- [ ] Make the repo importable under that command without `PYTHONPATH` workarounds.
- [ ] Add a smoke check for the canonical invocation.
- [ ] Standardize test-command references in docs and comments.
- [ ] Align `chat_web` request validation with the intended API contract for `system` messages.
- [ ] Add request-validation tests for `chat_web`.

**Acceptance criteria**

- [ ] One documented test command works from a clean shell in repo root.
- [ ] Test docs use one consistent invocation.
- [ ] `chat_web` validator behavior and error text agree.
- [ ] Validation tests cover the supported request contract.

### Phase 1 Exit Gate

- [ ] Fallback correctness is fixed.
- [ ] Canonical test workflow is trustworthy.
- [ ] Basic API/runtime safety issues are covered by tests.

---

## Phase 2: Swift / Build / Integration Hardening

### Sprint 3: Swift Rebuild and Portability

**Goal:** make the Swift seam rebuildable and portable without depending on local machine state.

- [ ] Add a clean Swift rebuild smoke path that does not rely on an existing binary.
- [ ] Verify `xcodebuild` assumptions in both Python wrapper entrypoints.
- [ ] Remove machine-specific repo-root fallbacks from committed code.
- [ ] Remove workstation-specific absolute paths from Swift benchmark/integration code.
- [ ] Align Swift CLI usage/help text with the actual supported invocation path.

**Acceptance criteria**

- [ ] Clean rebuild is exercised and trusted on the supported machine.
- [ ] No committed runtime code depends on `/Users/...` fallback paths.
- [ ] Swift usage/help text matches the supported workflow.

### Sprint 4: Shared Integration Surfaces and Entrypoint Cleanup

**Goal:** reduce integration drift and import-time side effects.

- [ ] Extract shared Swift helper logic from:
  - [ ] `nanochat/swift_stub_engine.py`
  - [ ] `scripts/mlx_swift_stub.py`
  - [ ] `dev/benchmark_swift_vs_python.py`
- [ ] Normalize older entrypoints around `build_parser()` and `main()`.
- [ ] Move module-scope CLI parsing out of `scripts/base_train.py`.
- [ ] Move module-scope CLI parsing and runtime initialization out of `scripts/chat_web.py`.
- [ ] Isolate smoke-test-only imports under `if __name__ == "__main__":` where appropriate.

**Acceptance criteria**

- [ ] Swift wrapper logic exists behind one shared helper surface.
- [ ] Script imports do not parse CLI args or initialize hardware.
- [ ] Entry-point behavior is consistent across old and new scripts.

### Phase 2 Exit Gate

- [ ] Swift rebuild path is trusted.
- [ ] Swift integration code no longer drifts across duplicated helpers.
- [ ] Entrypoints are safer to import, test, and reuse.

### Parallelization Notes

- Sprint 3 rebuild/portability work can run in parallel with helper extraction prep.
- Helper extraction should merge before entrypoint cleanup finishes, so all call sites converge on the same API.

---

## Phase 3: Initialization and Dependency Hygiene

### Sprint 5: Dependency, Initialization, and Compatibility Cleanup

**Goal:** make runtime requirements explicit and reduce import-time/global-state surprises.

- [ ] Audit all direct runtime imports against `pyproject.toml`.
- [ ] Add missing direct dependencies explicitly.
- [ ] Separate tokenizer runtime loading from tokenizer training-only dependency imports.
- [ ] Reduce import-time logging and compute-dtype side effects in core modules.
- [ ] Replace broad `except:` handling in reporting/utilities with explicit failure paths.
- [ ] Convert user-facing `assert` validation to explicit exceptions where appropriate.
- [ ] Refactor checkpoint compatibility helpers toward cleaner, testable migrations.

**Acceptance criteria**

- [ ] `pyproject.toml` matches direct runtime imports.
- [ ] Runtime tokenizer use no longer depends on training-only packages being present.
- [ ] Logging/dtype setup is explicit instead of import-order sensitive.
- [ ] Report/utilities surface actionable failures.
- [ ] Compatibility logic is cleaner and testable.

### Phase 3 Exit Gate

- [ ] Runtime dependencies are explicit.
- [ ] Core imports are materially less surprising.
- [ ] Compatibility and failure handling are easier to reason about.

### Parallelization Notes

- Dependency audit and tokenizer/runtime boundary cleanup can run in parallel.
- Checkpoint compatibility cleanup should wait until import-side effects are reduced, to keep refactor scope contained.

---

## Phase 4: Documentation and Regression Protection

### Sprint 6: Docs Reconciliation and Coverage Backfill

**Goal:** make docs trustworthy again and backstop the cleanup with targeted regression tests.

- [ ] Refresh `README.md` to reflect the current repo shape.
- [ ] Replace or trim stale file-tree snapshots.
- [ ] Update `docs/README.md` to include current March 2026 planning/research docs.
- [ ] Reconcile Apple-native docs so each topic has one current truth.
- [ ] Remove machine-local absolute paths from docs where repo-relative paths are possible.
- [ ] Add checkpoint migration/backward-compatibility tests.
- [ ] Add tokenizer conversation rendering and mask tests.
- [ ] Add execution timeout/error-reporting tests.
- [ ] Add report generation tests for partial-failure behavior.
- [ ] Add a Swift rebuild or wrapper smoke test where feasible.
- [ ] Add environment/import smoke checks for the canonical contributor workflow.

**Acceptance criteria**

- [ ] README and docs index reflect the current codebase.
- [ ] Apple-native docs are internally consistent enough to act as planning inputs.
- [ ] Previously uncovered core seams have direct regression coverage.
- [ ] Rebuild/import/workflow regressions are caught automatically.

### Phase 4 Exit Gate

- [ ] Documentation is aligned with code and workflow reality.
- [ ] Core cleanup work is protected by targeted regression tests.
- [ ] The code review plan can be considered complete.

### Parallelization Notes

- README/docs index cleanup can run in parallel with test backfill.
- Apple-native doc reconciliation should merge after any workflow-command changes, so docs capture the final canonical process.

---

## Explicit Deferrals

- [ ] Defer speculative architecture rewrites not required by the review findings.
- [ ] Defer broad MPS/MLX strategy work unrelated to the concrete code review items.
- [ ] Defer custom low-level performance work unless it is needed to complete a sprint acceptance criterion.
- [ ] Defer additional Swift feature work until rebuild trust and shared helpers are in place.

## Recommended Execution Order

- [ ] Sprint 1
- [ ] Sprint 2
- [ ] Sprint 3
- [ ] Sprint 4
- [ ] Sprint 5
- [ ] Sprint 6

## Suggested Cadence

- Sprint 1: 1 week
- Sprint 2: 0.5 to 1 week
- Sprint 3: 1 week
- Sprint 4: 1 week
- Sprint 5: 1 week
- Sprint 6: 1 to 1.5 weeks

## Definition of Done

- [ ] Fallback correctness is fixed and protected by tests.
- [ ] One canonical test workflow works from repo root.
- [ ] Swift rebuild/integration path is rebuildable and portable.
- [ ] Import-time side effects are materially reduced.
- [ ] Direct runtime dependencies are explicit.
- [ ] Docs and runtime/API behavior are aligned.
- [ ] Regression coverage exists for the previously weak seams.
