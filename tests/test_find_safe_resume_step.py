"""Tests for find_safe_resume_step — all 5 test-plan scenarios plus multi-rank edge cases.
Run: python -m pytest tests/test_find_safe_resume_step.py -v
"""
import os
import torch
from nanochat.checkpoint_manager import find_safe_resume_step, save_checkpoint

# Minimal data blobs reused across all tests
_MODEL = {"w": torch.zeros(1)}
_OPTIM = {"state": {}}
_META = {"step": 0}


def _full_checkpoint(d, step, world_size=1):
    """Write a complete checkpoint for all ranks using the production save_checkpoint API."""
    # rank 0 saves model + meta + its own optim shard
    save_checkpoint(d, step, _MODEL, _OPTIM, _META, rank=0)
    # remaining ranks save only their optim shard
    for r in range(1, world_size):
        save_checkpoint(d, step, None, _OPTIM, None, rank=r)


# ── scenario 3: empty-dir / no-dir ───────────────────────────────────────────

def test_no_directory_returns_minus_one(tmp_path):
    """Fresh start — checkpoint dir does not exist yet → -1, no crash."""
    assert find_safe_resume_step(str(tmp_path / "nonexistent")) == -1


def test_empty_directory_returns_minus_one(tmp_path):
    """mkdir base_checkpoints/d6 then run — empty dir returns -1, no crash."""
    d = tmp_path / "d6"
    d.mkdir()
    assert find_safe_resume_step(str(d)) == -1


# ── scenario 5: partial-write protection ─────────────────────────────────────

def test_model_only_no_meta_returns_minus_one(tmp_path):
    """Stray model_000500.pt with no meta/optim → find_safe_resume_step returns -1."""
    d = str(tmp_path)
    torch.save(_MODEL, os.path.join(d, "model_000500.pt"))
    assert find_safe_resume_step(d) == -1


def test_model_meta_no_optim_returns_minus_one(tmp_path):
    """model + meta present but optim shard missing → still -1 (incomplete write)."""
    d = str(tmp_path)
    # rank 0 path: save model + meta only (no optimizer_data)
    save_checkpoint(d, 500, _MODEL, None, _META, rank=0)
    assert find_safe_resume_step(d) == -1


# ── complete single-rank checkpoint ──────────────────────────────────────────

def test_complete_single_rank_returned(tmp_path):
    """Full checkpoint (world_size=1) → returns that step."""
    _full_checkpoint(str(tmp_path), 500, world_size=1)
    assert find_safe_resume_step(str(tmp_path), world_size=1) == 500


def test_returns_highest_complete_step(tmp_path):
    """Multiple checkpoints → highest fully-committed step returned."""
    d = str(tmp_path)
    for step in (500, 1000, 1500):
        _full_checkpoint(d, step)
    assert find_safe_resume_step(d) == 1500


def test_skips_incomplete_top_step_returns_previous(tmp_path):
    """Highest step incomplete (partial write) → falls back to previous valid step."""
    d = str(tmp_path)
    _full_checkpoint(d, 1000)
    # step 1500: model only, interrupted mid-save
    torch.save(_MODEL, os.path.join(d, "model_001500.pt"))
    assert find_safe_resume_step(d) == 1000


# ── multi-rank (world_size > 1) ───────────────────────────────────────────────

def test_multi_rank_all_shards_present(tmp_path):
    """world_size=8: all 8 optim shards present → step returned."""
    _full_checkpoint(str(tmp_path), 500, world_size=8)
    assert find_safe_resume_step(str(tmp_path), world_size=8) == 500


def test_multi_rank_one_shard_missing(tmp_path):
    """world_size=8: rank-7 shard missing → step incomplete, returns -1."""
    d = str(tmp_path)
    _full_checkpoint(d, 500, world_size=7)   # only ranks 0-6
    assert find_safe_resume_step(d, world_size=8) == -1


def test_multi_rank_falls_back_to_lower_complete_step(tmp_path):
    """world_size=4: step 1000 missing rank-3 → falls back to step 500."""
    d = str(tmp_path)
    _full_checkpoint(d, 500, world_size=4)
    # step 1000: only 3 of 4 shards (rank 3 missing)
    _full_checkpoint(d, 1000, world_size=3)
    assert find_safe_resume_step(d, world_size=4) == 500


def test_world_size_zero_raises(tmp_path):
    """world_size < 1 would make the shard check vacuously pass — reject it."""
    import pytest
    with pytest.raises(ValueError):
        find_safe_resume_step(str(tmp_path), world_size=0)


def test_atomic_save_leaves_no_tmp_and_is_discovered(tmp_path):
    """Completed save leaves no .tmp file and the step is discoverable."""
    d = str(tmp_path)
    _full_checkpoint(d, 500)
    assert not [f for f in os.listdir(d) if f.endswith(".tmp")]
    assert find_safe_resume_step(d) == 500


def test_interrupted_tmp_write_is_ignored(tmp_path):
    """A leftover model_000500.pt.tmp (killed mid-write) never counts as a checkpoint."""
    d = str(tmp_path)
    torch.save(_MODEL, os.path.join(d, "model_000500.pt.tmp"))
    _meta_path = os.path.join(d, "meta_000500.json")
    with open(_meta_path, "w", encoding="utf-8") as f:
        f.write("{}")
    assert find_safe_resume_step(d) == -1
