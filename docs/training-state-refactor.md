---
title: "TrainingState Refactor Plan"
summary: "Plan to extract mutable training loop variables into a TrainingState dataclass, eliminating the train_base_model closure."
read_when:
  - Implementing or reviewing the TrainingState refactor in base_train.py
  - Understanding why train_base_model was converted from a closure to a plain function
status: draft
last_updated: "2025-07-14"
---

# TrainingState Refactor Plan

Goal: eliminate the `train_base_model()` closure in `base_train.py` by bundling all mutable
loop state into a `TrainingState` dataclass. The closure exists solely to avoid passing ~20
captured variables as arguments — removing it makes state explicit, resumable, and testable.

---

## Step 1 — Create `training/train_state.py`

Define `TrainingState` with all mutable loop variables:

```python
@dataclass
class TrainingState:
    step: int
    val_bpb: float | None
    min_val_bpb: float
    smooth_train_loss: float
    total_training_time: float
    dataloader_state_dict: dict | None
```

Add two constructors:

- `TrainingState.fresh()` — zero-initialised state for a new run
- `TrainingState.from_checkpoint(meta_data)` — restores state from checkpoint `meta_data` dict

`from_checkpoint` reads:
- `meta_data["step"]`
- `meta_data["val_bpb"]`
- `meta_data["loop_state"]["min_val_bpb"]`
- `meta_data["loop_state"]["smooth_train_loss"]`
- `meta_data["loop_state"]["total_training_time"]`
- `meta_data["dataloader_state_dict"]`

---

## Step 2 — Update `base_train.py`

### 2a. Import and construct state

Replace the `if resuming / else` state init block:

```python
# before
if not resuming:
    step = 0
    val_bpb = None
    ...
else:
    step = meta_data["step"]
    loop_state = cast(...)
    ...

# after
state = TrainingState.fresh() if not resuming else TrainingState.from_checkpoint(meta_data)
```

### 2b. Replace bare variable access with `state.*`

All reads/writes of `step`, `val_bpb`, `min_val_bpb`, `smooth_train_loss`,
`total_training_time`, `dataloader_state_dict` become `state.*` throughout the loop.

### 2c. Replace `loop_state` dict in `save_checkpoint`

```python
# before
"loop_state": {
    "min_val_bpb": min_val_bpb,
    "smooth_train_loss": smooth_train_loss,
    "total_training_time": total_training_time,
},

# after
"loop_state": {
    "min_val_bpb": state.min_val_bpb,
    "smooth_train_loss": state.smooth_train_loss,
    "total_training_time": state.total_training_time,
},
```

### 2d. Promote `train_base_model` to module-level function

Remove the closure. Make it a plain `def train_base_model(state, ...)` with explicit
parameters for everything it previously captured:

```python
def train_base_model(
    state: TrainingState,
    config: Config,
    model,
    orig_model,
    optimizer,
    scaler,
    train_loader,
    build_val_loader,
    token_bytes,
    tokenizer,
    device,
    device_type,
    ...
) -> None:
```

Call site in `base_train` becomes:

```python
train_base_model(state=state, config=config, model=model, ...)
```

---

## Step 3 — Verify checkpoint round-trip

Confirm `TrainingState.from_checkpoint` reads exactly the keys that `save_checkpoint`
writes. The keys to verify:

| `from_checkpoint` reads | `save_checkpoint` writes |
|---|---|
| `meta_data["step"]` | `"step": step` |
| `meta_data["val_bpb"]` | `"val_bpb": val_bpb` |
| `meta_data["loop_state"]["min_val_bpb"]` | `"loop_state": {"min_val_bpb": ...}` |
| `meta_data["loop_state"]["smooth_train_loss"]` | `"loop_state": {"smooth_train_loss": ...}` |
| `meta_data["loop_state"]["total_training_time"]` | `"loop_state": {"total_training_time": ...}` |
| `meta_data["dataloader_state_dict"]` | `"dataloader_state_dict": ...` |

---

## Files touched

| File | Change |
|---|---|
| `src/nanochat/training/train_state.py` | New file — `TrainingState` dataclass |
| `src/nanochat/training/base_train.py` | Import `TrainingState`, replace init block, replace bare vars, promote closure |
