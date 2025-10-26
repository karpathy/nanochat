# Multilingual Support Proposal for nanochat

## Original Issue

### Summary
Add first-class multilingual support across tokenizer, data, training, and evaluation so models trained with nanochat learn robust semantic embeddings for multiple languages, not just English.

### Motivation
Current setup is English-centric (dataset and SFT tasks), which limits performance and semantic fidelity for non‑English scripts. Many users want to analyze or serve content in languages such as Spanish, French, German, Arabic, Hindi, Chinese, Japanese, Korean, etc.

### Current Behavior

**Tokenizer**: Unicode‑aware ByteLevel BPE with GPT‑style regex, but tuned for English; works on any script, though segmentation is suboptimal for CJK/Thai/etc. See `nanochat/tokenizer.py:30` and usage at `nanochat/tokenizer.py:75`.

**Data**: Pretraining pulls from FineWeb‑Edu shards, predominantly English. See `nanochat/dataset.py:19`.

**Training**: No multilingual‑specific settings, losses, or evaluations; SFT/mid tasks are English.

### Alternatives Considered
- Relying on Byte‑fallback only: functional but yields poor segmentation for CJK/Thai and weaker downstream performance.
- Training English tokenizer on multilingual data: still biases merges toward English and inflates non‑English token lengths.

---

## Proposed Changes

### Option 1: Multilingual Data Task (Minimal)

Add a generic task class that allows loading any HuggingFace dataset for multilingual training.

**Changes Required:**

1. **Create new file**: `tasks/multilingual.py` (~50 lines)
   - Generic wrapper around `load_dataset()` 
   - Supports any HF dataset with conversation format
   - Follows existing `Task` interface

2. **Documentation**: Update `README.md` with usage example

**Benefits:**
- Minimal code changes (~50 lines)
- Flexible: works with any HF dataset
- No breaking changes
- Users can add any language/corpus

**Limitations:**
- Doesn't address tokenizer segmentation quality
- No built-in evaluation for multilingual tasks

---

## Pull Request Checklist

### Core Implementation

- [ ] Create `tasks/multilingual.py` with `MultilingualTask` class
  - Inherit from `Task` base class
  - Implement `num_examples()` method
  - Implement `get_example()` method
  - Handle dataset loading via `load_dataset()`
  
- [ ] Add error handling for dataset format validation
  - Check for required "messages" field
  - Validate message structure (roles, content)
  
- [ ] Test with sample dataset
  - Load a simple HF dataset successfully
  - Verify task integration with `TaskMixture`

### Documentation

- [ ] Add multilingual example to `README.md`
  - Show how to add `MultilingualTask` to training pipeline
  - Include example HF dataset reference
  - Add brief explanation of use case

- [ ] Add docstring to `MultilingualTask` class
  - Document constructor parameters
  - Explain expected dataset format
  - Provide usage example

### Testing

- [ ] Test with at least one multilingual dataset
  - Suggested: `HuggingFaceTB/smol-talk-lt` (Lithuanian)
  - Verify data loads correctly
  - Check conversation format compatibility
  
- [ ] Verify backward compatibility
  - Existing training scripts still work
  - No regressions in existing tasks

### Code Quality

- [ ] Follow existing code style
  - Match formatting in `tasks/` directory
  - Use similar naming conventions
  
- [ ] Add type hints where appropriate
  - Use `typing` module for return types
  - Document parameter types

- [ ] Handle edge cases
  - Empty datasets
  - Missing fields in data
  - Invalid dataset names

---

## Minimal Documentation Changes

**Add to `README.md`:**

```markdown
## Multilingual Support (Experimental)

Add multilingual data to training using any HuggingFace dataset:

```python
from tasks.multilingual import MultilingualTask

train_ds = TaskMixture([
    # ... existing tasks ...
    MultilingualTask("HuggingFaceTB/smol-talk-lt", split="train"),  # Lithuanian
    MultilingualTask("tatsu-lab/alpaca", split="train"),  # Example
])
```

See `docs/multilingual_proposal.md` for full details.
```

---

## Backward Compatibility

✅ All changes are **opt-in**. Default behavior remains English-only.

- Default `vocab_size=65536` unchanged
- New `MultilingualTask` is additive
- `get_embeddings()` method is new, doesn't affect existing code
- No changes to default training pipeline

---

## Implementation Summary

**Effort**: ~1 day, ~80 lines of code

- **New file**: `tasks/multilingual.py` (~50 lines)
- **Documentation**: ~30 lines in README
- **Testing**: 1 multilingual dataset

**Files Changed:**
1. `tasks/multilingual.py` (new)
2. `README.md` (add section)

---

## Future Enhancements (Out of Scope)

- SentencePiece tokenizer option for better CJK segmentation
- Configurable tokenizer vocab size (128k-200k)
- Sentence embedding API
- Custom per-language regex patterns
- Contrastive training script
- Multilingual eval bundle
- Per-language weighting in TaskMixture
