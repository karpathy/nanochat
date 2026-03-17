# Nanochat MLX Swift Stub

This is the first minimal `mlx-swift` inference stub for the Apple-native runtime seam.

Current scope:

- loads the exported nanochat MLX sidecar manifest
- loads the paired `.safetensors` checkpoint directly with `mlx-swift`
- validates the expected tensor surface for the current MLX prototype
- runs a full forward pass from token ids to logits
- runs a simple greedy generation loop by recomputing over the growing prefix
- prints the generated token ids so a Python-side wrapper can decode them back to text
- validates the first exported checkpoint boundary on the Swift side without introducing tokenizer or engine-state work yet

Current non-goals:

- tokenizer parity in Swift
- KV-cache decoding loop
- tool-state logic from `nanochat/engine.py`
- production packaging or app integration
- GPU execution wiring for this CLI shell path
- efficient incremental decode; the current loop recomputes the full prefix each step

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
  --device cpu
```

The stub intentionally takes token ids instead of raw text so the checkpoint boundary can be exercised before committing to a Swift tokenizer path.

Current validation status:

- build verified with `swift build`
- runtime verified with `xcodebuild` output plus `DYLD_FRAMEWORK_PATH`
- end-to-end CPU forward pass verified against `runs/mlx_exports/phase2_d4_l_mps_step20.safetensors`
- GPU execution is deferred for a later pass

Python bridge:

If you want to stay Python-side for prompt handling, use the existing tokenizer and let a small wrapper invoke the Swift binary for you:

```bash
python -m scripts.mlx_swift_stub \
  --prompt "The chemical formula of water is" \
  --max-new-tokens 8 \
  --print-token-ids
```

Add `--rebuild` to force a fresh `xcodebuild` pass before invoking the stub.