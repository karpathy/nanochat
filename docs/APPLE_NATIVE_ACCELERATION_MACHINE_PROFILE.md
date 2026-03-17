# Apple-Native Acceleration Machine Profile

## Purpose

This note freezes the reference machine and software environment for Apple-native acceleration planning.

Capture date: 2026-03-16

## Reference Hardware

- Model: Mac Studio
- Model identifier: `Mac14,14`
- Chip: Apple M2 Ultra
- CPU cores: 24 total (`16` performance + `8` efficiency)
- Memory: `128 GB` unified memory
- Firmware version: `13822.81.10`
- OS loader version: `13822.81.10`

Additional repo context from the current scaling note:

- PyTorch exposes this machine as a single `mps` accelerator.
- The relevant scaling problem is single-device unified-memory optimization, not multi-GPU distribution.

## Reference Software

- macOS: `26.3.1 (25D2128)`
- Darwin kernel: `25.3.0`
- Repo branch: `feature/apple-silicon-native-mps`
- Baseline commit: `7815bc27538f0494c03ee1e6f8dba0005ad8ca91`

## Reference Python Runtime

The planning baseline should use the repo-local virtual environment, not the system interpreter.

- Python: `3.10.17`
- Python implementation: `CPython`
- PyTorch: `2.9.1`

Installed baseline package versions observed in `.venv`:

- `datasets==4.0.0`
- `fastapi==0.117.1`
- `matplotlib==3.10.8`
- `psutil==7.1.0`
- `python-dotenv==1.2.1`
- `regex==2025.9.1`
- `rustbpe==0.1.0`
- `scipy==1.15.3`
- `tabulate==0.9.0`
- `tiktoken==0.11.0`
- `tokenizers==0.22.0`
- `torch==2.9.1`
- `transformers==4.57.3`
- `uvicorn==0.36.0`
- `wandb==0.21.3`
- `zstandard==0.25.0`

## Evidence Collection Commands

Hardware and macOS:

```bash
system_profiler SPHardwareDataType SPSoftwareDataType
```

Branch and commit:

```bash
git branch --show-current
git rev-parse HEAD
```

Repo-local runtime:

```bash
.venv/bin/python -V
.venv/bin/python -c "import platform, torch; print(torch.__version__); print(platform.python_implementation())"
.venv/bin/python -c "import importlib.metadata as m; ..."
```

## Baseline Usage Rule

Unless a later planning note explicitly supersedes this file, this machine profile is the default environment for:

- PyTorch + MPS baseline measurements
- framework comparison planning
- prototype acceptance criteria

## Related Documents

- [APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md](APPLE_NATIVE_ACCELERATION_BASELINE_BENCHMARK.md)
- [M2_ULTRA_SCALING.md](M2_ULTRA_SCALING.md)
- [APPLE_SILICON_PORTABILITY.md](APPLE_SILICON_PORTABILITY.md)