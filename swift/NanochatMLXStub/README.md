# Nanochat MLX Swift Stub

This is the first minimal `mlx-swift` inference stub for the Apple-native runtime seam.

Current scope:

- loads the exported nanochat MLX sidecar manifest
- loads the paired `.safetensors` checkpoint directly with `mlx-swift`
- validates the expected tensor surface for the current MLX prototype
- runs a full forward pass from token ids to logits
- runs a greedy generation loop with KV-cache incremental decoding
- supports both CPU and GPU execution via `Device.withDefaultDevice`
- prints the generated token ids and per-token timing so a Python-side wrapper can decode and benchmark
- supports a persistent JSON-lines stdin/stdout worker mode for Python-side reuse
- validates the first exported checkpoint boundary on the Swift side without introducing tokenizer or engine-state work yet

Current non-goals:

- tokenizer parity in Swift
- tool-state logic from `nanochat/engine.py`
- production packaging or app integration

Why `xcodebuild` instead of `swift run`:

`mlx-swift` command-line tools need the `Cmlx` bundle on the runtime framework path so Metal shaders can be found. Plain `swift run` builds the code, but for shell execution the reliable path is to build with `xcodebuild` and then export `DYLD_FRAMEWORK_PATH` to the built products directory.

Build:

```bash
cd swift/NanochatMLXStub
xcodebuild -scheme NanochatMLXStub -destination 'platform=macOS' -derivedDataPath .derived build
```

Run from the repo root:

```bash
export DYLD_FRAMEWORK_PATH="$PWD/swift/Build/Products/Debug"
$PWD/swift/Build/Products/Debug/nanochat-mlx-stub \
  --manifest runs/mlx_exports/phase2_d4_l_mps_step20.json \
  --prompt-tokens 32759,464,1223 \
  --max-new-tokens 8 \
  --device gpu
```

The stub intentionally takes token ids instead of raw text so the checkpoint boundary can be exercised before committing to a Swift tokenizer path.

Persistent worker mode:

```bash
export DYLD_FRAMEWORK_PATH="$PWD/swift/Build/Products/Debug"
$PWD/swift/Build/Products/Debug/nanochat-mlx-stub \
  --manifest runs/mlx_exports/mlx_reference_d32.json \
  --device gpu \
  --serve-stdin \
  --prompt-tokens 0 \
  --max-new-tokens 1
```

After the initial ready line, send one JSON request per line on stdin:

```json
{"prompt_tokens":[32759,483,2027],"max_new_tokens":8,"stop_token_ids":[32759]}
```

Current validation status:

- build verified with `swift build`
- runtime verified with `xcodebuild` output plus `DYLD_FRAMEWORK_PATH`
- end-to-end CPU and GPU forward pass verified against `runs/mlx_exports/phase2_d4_l_mps_step20.safetensors`
- KV-cache produces identical token sequences to full-prefix recompute
- timing: d4 model on M2 Ultra — GPU 3.94ms/token, CPU 8.91ms/token (KV-cache decode)

Python bridge:

If you want to stay Python-side for prompt handling, use the existing tokenizer and let a small wrapper invoke the Swift binary for you:

```bash
python -m scripts.mlx_swift_stub \
  --prompt "The chemical formula of water is" \
  --max-new-tokens 8 \
  --print-token-ids
```

Add `--rebuild` to force a fresh `xcodebuild` pass before invoking the stub.