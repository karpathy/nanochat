# torch.compile Validation Report

## Status: READY FOR MANUAL TESTING

This document tracks the validation status of the `torch.compile` implementation for the chat SFT training script.

---

## Prerequisite Tasks Assessment

### Task 42: Fixed Padding Implementation
**Status**: ❌ NOT IMPLEMENTED

**Current State**:
- The `collate_and_yield` function (lines 89-109 in `scripts/chat_sft.py`) uses dynamic padding:
  ```python
  ncols = max(len(ids) for ids, mask in batch) - 1  # Line 94
  ```
- No `max_seq_len` constant is defined (unlike `base_train.py` and `mid_train.py`)

**Required for Task 43**: Fixed padding with constant `max_seq_len=2048` must be implemented before `torch.compile` with `dynamic=False` can work effectively.

---

### Task 43: torch.compile with dynamic=False
**Status**: ❌ NOT ENABLED

**Current State**:
- Line 72 in `scripts/chat_sft.py`:
  ```python
  # model = torch.compile(model, dynamic=True)  # doesn't work super well because of variable lengths of inputs
  ```
- The torch.compile call is commented out
- Uses `dynamic=True` (should be `dynamic=False`)

**Required Change**:
```python
model = torch.compile(model, dynamic=False)
```

---

### Task 44: Use orig_model for Evaluation and Checkpointing
**Status**: ⚠️ PARTIALLY IMPLEMENTED

**Current State**:
- ✅ Line 71: `orig_model = model` - Variable is created
- ❌ Line 173: Uses `model` for validation (should be OK for gradient computation)
- ❌ Line 192: `run_chat_eval("MMLU", model, ...)` - Should use `orig_model`
- ❌ Line 251: `model.state_dict()` - Should use `orig_model.state_dict()`

**Required Changes**:
1. Update evaluation calls to use `orig_model`:
   ```python
   metrics["mmlu_acc"] = run_chat_eval("MMLU", orig_model, tokenizer, engine, ...)
   metrics["arc_easy_acc"] = run_chat_eval("ARC-Easy", orig_model, tokenizer, engine, ...)
   ```

2. Update checkpoint saving to use `orig_model`:
   ```python
   save_checkpoint(
       checkpoint_dir,
       step,
       orig_model.state_dict(),  # Changed from model.state_dict()
       None,
       {...}
   )
   ```

---

## Validation Instrumentation Added

The following temporary logging has been added to `scripts/chat_sft.py` to facilitate validation:

### 1. Compilation Status Detection (Line ~76)
```python
if hasattr(model, '_orig_mod'):
    print0("[VALIDATION] ✓ Model is compiled (torch.compile detected)")
else:
    print0("[VALIDATION] ✗ Model is NOT compiled (running in eager mode)")
```

**Purpose**: Confirms whether torch.compile is active at startup

---

### 2. Batch Shape Logging (Line ~211)
```python
if step < 3 and micro_step == 0:
    print0(f"[VALIDATION] Step {step} | Batch shape: {train_inputs.shape}")
```

**Purpose**: Verifies fixed padding by checking if all batches have constant shape `(4, 2048)`

**Expected Output** (with fixed padding):
```
[VALIDATION] Step 0 | Batch shape: torch.Size([4, 2048])
[VALIDATION] Step 1 | Batch shape: torch.Size([4, 2048])
[VALIDATION] Step 2 | Batch shape: torch.Size([4, 2048])
```

---

### 3. Performance Metrics (Line ~236)
```python
# Tracks step_time and calculates tokens/sec every 10 steps
# Excludes first 5 warmup iterations
```

**Purpose**: Measures performance improvement from torch.compile

**Expected Output**:
```
[VALIDATION] Avg time/step: 2.450s | Tokens/sec: 3265.3
[VALIDATION] Avg time/step: 2.380s | Tokens/sec: 3358.0
```

**Key Metrics**:
- Baseline (without compile): Record tokens/sec
- With compile: Should show 1.3-1.5x improvement (30-50% faster)

---

## Test Execution Plan

Once prerequisites (Tasks 42, 43, 44) are completed, run the following tests:

### Test 1: Baseline (No Compilation)
```bash
# Comment out line 72 (torch.compile)
torchrun --standalone --nproc_per_node=1 \
  -m scripts.chat_sft -- \
  --max_iterations=100 \
  --model_source=base \
  --model_tag=d20 \
  --step=0
```

**Record**:
- [ ] All batch shapes are `(4, 2048)`
- [ ] Tokens/sec: _______
- [ ] Avg time/step: _______
- [ ] Final loss: _______

---

### Test 2: With Compilation
```bash
# Uncomment line 72 and set dynamic=False
torchrun --standalone --nproc_per_node=1 \
  -m scripts.chat_sft -- \
  --max_iterations=100 \
  --model_source=base \
  --model_tag=d20 \
  --step=0
```

**Verify**:
- [ ] Compilation message appears: `[VALIDATION] ✓ Model is compiled`
- [ ] No recompilation messages after initial compilation
- [ ] Tokens/sec improvement: _______ (target: ≥1.3x baseline)
- [ ] Loss trajectory matches Test 1 (within ±5%)

---

### Test 3: Multi-GPU (4 GPUs)
```bash
torchrun --standalone --nproc_per_node=4 \
  -m scripts.chat_sft -- \
  --max_iterations=100 \
  --model_source=base \
  --model_tag=d20 \
  --step=0
```

**Verify**:
- [ ] All 4 ranks initialize successfully
- [ ] No DDP synchronization errors
- [ ] Performance improvement similar to single-GPU test

---

## Success Criteria

### Functional Requirements
- [ ] Constant batch shapes throughout training (verified by logging)
- [ ] Successful compilation without errors
- [ ] Zero recompilations during training
- [ ] Zero recompilations during evaluation (using orig_model)
- [ ] Checkpoints save and load correctly
- [ ] Works in both single-GPU and multi-GPU configurations

### Performance Requirements
- [ ] 30-50% speed improvement (tokens/sec ratio ≥ 1.3x)
- [ ] Initial compilation time ≤ 60 seconds
- [ ] GPU memory usage within 10% of baseline

### Accuracy Requirements
- [ ] Loss convergence matches baseline (within ±5% at iteration 100)
- [ ] Evaluation metrics match historical baselines

---

## Current Blockers

1. **Task 42 (Fixed Padding)**: Must be implemented to enable `dynamic=False` compilation
2. **Task 43 (Enable Compilation)**: Line 72 must be uncommented and changed to `dynamic=False`
3. **Task 44 (Use orig_model)**: Evaluation and checkpointing must use uncompiled model

**Recommendation**: Complete prerequisite tasks before proceeding with validation tests.

---

## Rollback Procedure

If validation fails, disable compilation by commenting out line 72:
```python
# model = torch.compile(model, dynamic=False)
```

To remove validation logging after successful testing:
1. Remove lines ~159-161 (performance tracking variables)
2. Remove line ~163 (step_start_time)
3. Remove lines ~211-213 (batch shape logging)
4. Remove lines ~233-245 (performance metrics calculation)
5. Remove lines ~76-80 (compilation status logging)

---

## Notes

- **Validation logging is temporary** and should be removed after testing
- Performance measurements should exclude the first 5 warmup iterations
- Expected net time savings: 15-20 minutes per full SFT training run
- PyTorch version must be ≥ 2.0 for torch.compile support
