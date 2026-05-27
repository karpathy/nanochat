"""
Hermetic tests for clarinet's source-marker-aware dataloader.

Uses a mock tokenizer and synthetic parquet files in tmp_path so nothing
depends on a real model, real climbmix, or real reasoning-corpus download.

Run: python -m pytest tests/test_clarinet_dataloader.py -v
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import clarinet.dataloader as dl
from clarinet.dataloader import (
    SRC_GENERAL,
    SRC_REASONING,
    SRC_UNKNOWN,
    _interleave_sources,
    clarinet_data_loader,
)


class MockTokenizer:
    """
    Minimal interface match for clarinet_data_loader.
    Tokens 0-999 are unused; specials live at 1000+.
    encode() treats input chars as their ord() (no BPE — we just need the
    invariants around marker insertion and BOS-target masking, not real text).
    """
    BOS = 1000
    SRC_REASONING_ID = 1001
    SRC_GENERAL_ID = 1002
    SRC_UNKNOWN_ID = 1003

    SPECIALS = {
        "<|bos|>": BOS,
        SRC_REASONING: SRC_REASONING_ID,
        SRC_GENERAL: SRC_GENERAL_ID,
        SRC_UNKNOWN: SRC_UNKNOWN_ID,
    }
    MARKER_IDS = {SRC_REASONING_ID, SRC_GENERAL_ID, SRC_UNKNOWN_ID}

    def get_bos_token_id(self):
        return self.BOS

    def encode_special(self, name):
        return self.SPECIALS[name]

    def encode(self, text, prepend=None, append=None, num_threads=4):
        def one(s):
            ids = [ord(c) for c in s]
            if prepend is not None:
                ids = [prepend] + ids
            if append is not None:
                ids = ids + [append]
            return ids

        if isinstance(text, str):
            return one(text)
        return [one(t) for t in text]


def _write_shard(path, n_docs, doc_len=8, prefix=""):
    docs = [f"{prefix}{i:03d}".ljust(doc_len, "x") for i in range(n_docs)]
    table = pa.table({"text": docs})
    # One row group per file keeps the DDP-rank math trivial in single-process tests.
    pq.write_table(table, path, row_group_size=n_docs)


@pytest.fixture
def fake_paths(tmp_path, monkeypatch):
    climbmix = tmp_path / "climbmix"
    reasoning = tmp_path / "reasoning"
    climbmix.mkdir()
    reasoning.mkdir()
    for shard_idx in range(3):
        _write_shard(climbmix / f"shard_{shard_idx:05d}.parquet", n_docs=30, prefix="c")
        _write_shard(reasoning / f"shard_{shard_idx:05d}.parquet", n_docs=30, prefix="r")
    climbmix_paths = sorted(str(p) for p in climbmix.glob("*.parquet"))
    reasoning_paths = sorted(str(p) for p in reasoning.glob("*.parquet"))

    paths_with_source = [(p, False) for p in climbmix_paths] + [(p, True) for p in reasoning_paths]
    monkeypatch.setattr(dl, "list_parquet_files_with_source", lambda split: paths_with_source)
    return climbmix_paths, reasoning_paths


def _first_batch(tok, **kwargs):
    """Convenience: build a loader, pull one batch, return (inputs, targets) on CPU."""
    loader = clarinet_data_loader(
        tok, B=4, T=64, split="train",
        tokenizer_threads=1,
        tokenizer_batch_size=8,
        device="cpu",
        buffer_size=20,
        **kwargs,
    )
    inputs, targets, _state = next(loader)
    return inputs, targets


def _assert_path_str_bool_pairs(out):
    """Guard against the regression where interleave returned ((str, bool), bool)."""
    for entry in out:
        assert isinstance(entry, tuple) and len(entry) == 2, f"bad shape: {entry!r}"
        path, is_r = entry
        assert isinstance(path, str), f"path should be str, got {type(path).__name__}: {path!r}"
        assert isinstance(is_r, bool), f"is_reasoning should be bool, got {type(is_r).__name__}"


def test_interleave_sources_50_50():
    paths = [(f"c{i}", False) for i in range(6)] + [(f"r{i}", True) for i in range(6)]
    out = _interleave_sources(paths, 0.5)
    assert len(out) == 12
    _assert_path_str_bool_pairs(out)
    n_reasoning = sum(1 for _, r in out if r)
    assert n_reasoning == 6
    # No long run of either source — alternation should be tight at 0.5
    for i in range(len(out) - 2):
        window = [out[i][1], out[i + 1][1], out[i + 2][1]]
        assert not all(window), "three consecutive reasoning files at ratio 0.5"
        assert not all(not w for w in window), "three consecutive climbmix files at ratio 0.5"


def test_interleave_sources_30_70():
    paths = [(f"c{i}", False) for i in range(7)] + [(f"r{i}", True) for i in range(3)]
    out = _interleave_sources(paths, 0.3)
    assert len(out) == 10
    _assert_path_str_bool_pairs(out)
    n_reasoning = sum(1 for _, r in out if r)
    assert n_reasoning == 3


def test_interleave_sources_empty_one_side():
    paths = [(f"c{i}", False) for i in range(3)]
    out = _interleave_sources(paths, 0.5)
    assert out == paths  # no reasoning -> passthrough


def test_marker_always_follows_bos(fake_paths):
    tok = MockTokenizer()
    inputs, _ = _first_batch(tok, reasoning_mix_ratio=0.5, p_uncond=0.5, seed=0)
    bos_positions = (inputs == tok.BOS).nonzero(as_tuple=False)
    assert len(bos_positions) > 0, "expected at least one BOS in the packed batch"
    for row, col in bos_positions.tolist():
        if col + 1 >= inputs.size(1):
            continue  # BOS at the very last position — no room for a marker
        next_tok = inputs[row, col + 1].item()
        assert next_tok in tok.MARKER_IDS, (
            f"position ({row},{col+1}) after BOS holds {next_tok}, expected one of "
            f"{tok.MARKER_IDS}"
        )


def test_p_uncond_one_yields_only_unknown_markers(fake_paths):
    tok = MockTokenizer()
    inputs, _ = _first_batch(tok, reasoning_mix_ratio=0.5, p_uncond=1.0, seed=0)
    bos_positions = (inputs == tok.BOS).nonzero(as_tuple=False)
    markers = [inputs[r, c + 1].item() for r, c in bos_positions.tolist() if c + 1 < inputs.size(1)]
    assert markers, "no markers found"
    assert all(m == tok.SRC_UNKNOWN_ID for m in markers), (
        f"p_uncond=1.0 should override every marker to SRC_UNKNOWN, got {set(markers)}"
    )


def test_p_uncond_zero_yields_no_unknown_markers(fake_paths):
    tok = MockTokenizer()
    inputs, _ = _first_batch(tok, reasoning_mix_ratio=0.5, p_uncond=0.0, seed=0)
    bos_positions = (inputs == tok.BOS).nonzero(as_tuple=False)
    markers = [inputs[r, c + 1].item() for r, c in bos_positions.tolist() if c + 1 < inputs.size(1)]
    assert markers, "no markers found"
    assert all(m != tok.SRC_UNKNOWN_ID for m in markers), (
        f"p_uncond=0.0 should never emit SRC_UNKNOWN, got {[m for m in markers if m == tok.SRC_UNKNOWN_ID]}"
    )


def test_marker_matches_source_at_extreme_mix_ratios(fake_paths):
    tok = MockTokenizer()

    # mix_ratio=0.0 → all docs from climbmix → all markers SRC_GENERAL (with p_uncond=0)
    inputs_general, _ = _first_batch(tok, reasoning_mix_ratio=0.0, p_uncond=0.0, seed=1)
    bos_positions = (inputs_general == tok.BOS).nonzero(as_tuple=False)
    markers = [inputs_general[r, c + 1].item() for r, c in bos_positions.tolist() if c + 1 < inputs_general.size(1)]
    assert markers and all(m == tok.SRC_GENERAL_ID for m in markers), (
        f"mix_ratio=0 should produce only SRC_GENERAL markers, got {set(markers)}"
    )

    # mix_ratio=1.0 → all reasoning → all markers SRC_REASONING (with p_uncond=0)
    inputs_reasoning, _ = _first_batch(tok, reasoning_mix_ratio=1.0, p_uncond=0.0, seed=2)
    bos_positions = (inputs_reasoning == tok.BOS).nonzero(as_tuple=False)
    markers = [inputs_reasoning[r, c + 1].item() for r, c in bos_positions.tolist() if c + 1 < inputs_reasoning.size(1)]
    assert markers and all(m == tok.SRC_REASONING_ID for m in markers), (
        f"mix_ratio=1 should produce only SRC_REASONING markers, got {set(markers)}"
    )


def test_targets_masked_at_bos_input_positions(fake_paths):
    tok = MockTokenizer()
    inputs, targets = _first_batch(tok, reasoning_mix_ratio=0.5, p_uncond=0.1, seed=3)
    # Wherever input is BOS, the corresponding target (next-token) is the marker —
    # the dataloader must mask those targets to -1 so the model doesn't train to
    # predict the source.
    bos_input_mask = inputs == tok.BOS
    assert bos_input_mask.any(), "expected at least one BOS position in inputs"
    assert (targets[bos_input_mask] == -1).all(), (
        "every target at a BOS-input position must be -1, but some leaked through"
    )
    # Sanity: non-BOS-input positions should overwhelmingly have valid targets.
    non_bos = targets[~bos_input_mask]
    assert (non_bos != -1).any(), "no non-masked targets — something is wrong"


def test_loader_produces_correct_tensor_shapes(fake_paths):
    tok = MockTokenizer()
    loader = clarinet_data_loader(
        tok, B=3, T=32, split="train",
        reasoning_mix_ratio=0.5, p_uncond=0.1,
        tokenizer_threads=1, tokenizer_batch_size=8,
        device="cpu", buffer_size=10, seed=0,
    )
    inputs, targets, state = next(loader)
    assert inputs.shape == (3, 32)
    assert targets.shape == (3, 32)
    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long
    assert {"pq_idx", "rg_idx", "epoch"} <= set(state.keys())
