# Apple Silicon Portability

## Status

- Branch: `feature/apple-silicon-native-mps`
- Goal: make nanochat run more cleanly on Apple Silicon / MPS without regressing existing CUDA behavior.
- Current state: targeted code changes are in place and validated with focused tests plus a minimal MPS forward-pass smoke test.

## Changes Made

### Runtime helpers

- Added `is_mps_available()` in `nanochat/common.py` to centralize MPS detection.
- Added `should_torch_compile()` and `maybe_torch_compile()` in `nanochat/common.py`.
- Default behavior on MPS is eager mode, not `torch.compile`.
- `NANOCHAT_COMPILE=1` can be used to force `torch.compile` attempts on MPS.

### Training entrypoints

- `scripts/base_train.py` now uses `maybe_torch_compile(...)` instead of unconditional `torch.compile(...)`.
- `scripts/chat_sft.py` now uses `maybe_torch_compile(...)` instead of unconditional `torch.compile(...)`.

### Engine and reporting

- Fixed CUDA-only synchronization in `nanochat/engine.py` self-test code.
- Updated `nanochat/report.py` to report MPS / Apple Silicon accelerators sensibly instead of only reporting CUDA or no GPU.

### Packaging and setup

- Added `macos` optional dependency extra in `pyproject.toml`.
- Updated `runs/runcpu.sh` to use `uv sync --extra macos` on Darwin arm64.
- Updated `README.md` with macOS install guidance and compile behavior notes.

### Tests

- Updated `tests/test_attention_fallback.py` so the portable SDPA path chooses MPS when available instead of falling back directly from CUDA to CPU.

## Validation Completed

### Unit / integration tests

Run from repo root:

```bash
export PYTHONPATH="$PWD"
uv run pytest tests/test_attention_fallback.py -q
```

Observed result:

- `5 passed, 10 skipped`

### Environment resolution

Run from repo root:

```bash
uv sync --extra macos
```

Observed result:

- dependency resolution completed successfully on Apple Silicon

### MPS runtime smoke test

Run from repo root:

```bash
export PYTHONPATH="$PWD"
uv run python -c "import torch; from nanochat.gpt import GPT, GPTConfig; from nanochat.common import maybe_torch_compile; device=torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'); model=GPT(GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=64, window_pattern='L')); model.to(device); x=torch.randint(0, 128, (1, 16), device=device); y=model(x); compiled=maybe_torch_compile(model, 'mps', dynamic=False) is model if device.type == 'mps' else None; print({'device': str(device), 'output_shape': tuple(y.shape), 'compile_skipped': compiled})"
```

Observed result:

- device detected as `mps`
- output shape `(1, 16, 128)`
- compile helper skipped compilation on MPS by default

### Optimizer-step smoke test

Run from repo root:

```bash
export PYTHONPATH="$PWD"
uv run python -c "import torch; from nanochat.gpt import GPT, GPTConfig; device=torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'); model=GPT(GPTConfig(sequence_len=16, vocab_size=128, n_layer=2, n_head=2, n_kv_head=2, n_embd=64, window_pattern='L')).to(device); opt=model.setup_optimizer(unembedding_lr=0.001, embedding_lr=0.001, matrix_lr=0.001, weight_decay=0.0); x=torch.randint(0,128,(1,16),device=device); y=x.clone(); loss=model(x,y); loss.backward(); opt.step(); model.zero_grad(set_to_none=True); print({'device': str(device), 'loss': float(loss.item())})"
```

Observed result:

- single optimizer step completed on `mps`
- no portability-specific optimizer failure surfaced

### Full test suite

Run from repo root:

```bash
export PYTHONPATH="$PWD"
uv run pytest -q
```

Observed result:

- `13 passed, 10 skipped`

## Files Changed

- `README.md`
- `nanochat/common.py`
- `nanochat/engine.py`
- `nanochat/report.py`
- `pyproject.toml`
- `runs/runcpu.sh`
- `scripts/base_train.py`
- `scripts/chat_sft.py`
- `tests/test_attention_fallback.py`
- `uv.lock`

## Notes On `uv.lock`

- `uv sync --extra macos` updated `uv.lock` to include the new `macos` extra.
- The lockfile diff is broader than the source diff because `uv` rewrites environment markers and optional dependency combinations.
- Before merging, it is worth a quick review to confirm the lockfile shape is acceptable for the repo's dependency management expectations.

## Known Unrelated Working Tree Item

- `.vscode/settings.json` is present in the working tree and is unrelated to this portability work.
- It was not modified as part of this task.

## Suggested Next Steps

1. Decide whether to keep the current `uv.lock` rewrite or regenerate it in the preferred repo workflow.
2. Run a tiny end-to-end training smoke test on MPS using a very small config.
3. Commit the branch changes once the lockfile decision is settled.
4. Open a PR from `feature/apple-silicon-native-mps`.