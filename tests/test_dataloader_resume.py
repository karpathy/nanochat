"""
Author: nanochat team
Created: 2026-11-15
Scope: FIX 1 — dataloader doc_buffer resume correctness.
       Verifies that state_dict captures the FULL doc_buffer (not capped at 20),
       and that resuming from a saved state_dict yields the same next batches as
       an uninterrupted run.
"""

import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockTokenizer:
    """Minimal deterministic tokenizer: encodes chars to ASCII codes, prepends BOS."""

    BOS = 1

    def get_bos_token_id(self) -> int:
        return self.BOS

    def encode(self, text_batch, prepend, num_threads=1):
        result = []
        for text in text_batch:
            tokens = [prepend] + [ord(c) % 126 + 2 for c in text[:30]]
            result.append(tokens)
        return result


def _make_parquet(tmp_path, n_docs: int = 250, seed: int = 42) -> str:
    """Write a single-row-group parquet file with n_docs synthetic text documents."""
    rng = random.Random(seed)
    texts = [
        "".join(rng.choices("abcdefghijklmnop", k=rng.randint(5, 15)))
        for _ in range(n_docs)
    ]
    table = pa.table({"text": texts})
    path = str(tmp_path / "data.parquet")
    pq.write_table(table, path, row_group_size=n_docs)
    return path


def _make_loader(tokenizer, parquet_path, resume_state_dict=None, buffer_size=100):
    """Return a generator configured to use the synthetic parquet as the only shard."""
    return tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer,
        B=1,
        T=16,
        split="train",
        tokenizer_batch_size=64,
        device="cpu",
        resume_state_dict=resume_state_dict,
        buffer_size=buffer_size,
    )


# ---------------------------------------------------------------------------
# § FIX 1: doc_buffer serialization
# ---------------------------------------------------------------------------

class TestDocBufferResume:
    """
    Scenario: FIX 1 — full doc_buffer serialization prevents silent data loss on resume.
    Slice: nanochat.dataloader tokenizing_distributed_data_loader_with_state_bos_bestfit
    """

    def test_state_dict_has_full_buffer_gt_20(self, tmp_path, monkeypatch):
        """
        Scenario: After packing the first batch, the leftover doc_buffer (captured at the
                  start of the second loop iteration) must hold ALL remaining documents,
                  not just the first 20 (old _DOC_BUFFER_CHECKPOINT_CAP=20 bug).
        Slice: state_dict['doc_buffer'] size after FIX 1
        ### Given: a data source that fills doc_buffer to buffer_size=100 entries
        ### When: the second batch is yielded (state_dict snapshot = leftover after batch 0)
        ### Then: state_dict['doc_buffer'] has > 20 entries
        """
        parquet_path = _make_parquet(tmp_path, n_docs=250)
        # list_parquet_files returns [train_shard, val_shard]; we use the same file for both.
        monkeypatch.setattr(
            "nanochat.dataloader.list_parquet_files",
            lambda warn_on_legacy=False: [parquet_path, parquet_path],
        )

        tokenizer = _MockTokenizer()
        gen = _make_loader(tokenizer, parquet_path, buffer_size=100)

        # First yield: state_dict captured BEFORE batch 0 packing → buffer is empty.
        next(gen)
        # Second yield: state_dict captured AFTER batch 0 packing → buffer has the
        # remaining docs from the refill.  With buffer_size=100 and T=16 (small batch),
        # most of the 100 refilled docs survive into the second state_dict.
        _, _, state_dict = next(gen)

        assert len(state_dict["doc_buffer"]) > 20, (
            f"Expected > 20 docs in state_dict['doc_buffer'], got "
            f"{len(state_dict['doc_buffer'])}. "
            "The old _DOC_BUFFER_CHECKPOINT_CAP=20 bug may still be present."
        )

    def test_resume_produces_same_batches_as_uninterrupted_run(self, tmp_path, monkeypatch):
        """
        Scenario: Resuming from a saved state_dict (with a large doc_buffer) yields exactly
                  the same next batches as an uninterrupted run.
        Slice: resume from state_dict with large doc_buffer (> 20 entries)
        ### Given: an uninterrupted dataloader run of at least 2 batches
        ### When: resumed from the state_dict saved at batch 1 (non-empty buffer)
        ### Then: the next 3 batches from the resumed loader match the uninterrupted run
        """
        parquet_path = _make_parquet(tmp_path, n_docs=250)
        monkeypatch.setattr(
            "nanochat.dataloader.list_parquet_files",
            lambda warn_on_legacy=False: [parquet_path, parquet_path],
        )

        tokenizer = _MockTokenizer()

        # --- Uninterrupted run ---
        gen_full = _make_loader(tokenizer, parquet_path, buffer_size=100)

        # Discard batch 0; its state_dict has an empty buffer (before first refill).
        next(gen_full)

        # Batch 1 — state_dict here has the leftover buffer from refilling for batch 0.
        # This is the state with a large (> 20 entry) doc_buffer that FIX 1 preserves.
        x1_full, y1_full, saved_state = next(gen_full)

        # Collect the reference batches starting from batch 1.
        ref_batches = [(x1_full.clone(), y1_full.clone())]
        x2, y2, _ = next(gen_full)
        ref_batches.append((x2.clone(), y2.clone()))
        x3, y3, _ = next(gen_full)
        ref_batches.append((x3.clone(), y3.clone()))

        # --- Resumed run (from saved_state at batch 1) ---
        gen_resume = _make_loader(tokenizer, parquet_path, resume_state_dict=saved_state, buffer_size=100)

        # The resumed run must reproduce batch 1, 2, 3.
        # Clone each tensor immediately: inputs/targets are views into a persistent
        # pre-allocated buffer that is overwritten on every next() call.  Without
        # cloning, xr1 and xr2 would be overwritten by xr3 before the comparison.
        resume_batches = []
        for _ in range(3):
            xr, yr, _ = next(gen_resume)
            resume_batches.append((xr.clone(), yr.clone()))

        for i, ((x_full, y_full), (x_res, y_res)) in enumerate(
            zip(ref_batches, resume_batches)
        ):
            assert torch.equal(x_full, x_res), (
                f"Inputs differ at resumed step {i}: "
                f"full={x_full.tolist()} resume={x_res.tolist()}"
            )
            assert torch.equal(y_full, y_res), (
                f"Targets differ at resumed step {i}: "
                f"full={y_full.tolist()} resume={y_res.tolist()}"
            )

    def test_old_cap_would_break_resume(self, tmp_path, monkeypatch):
        """
        Scenario: Falsification — if the cap were still applied (at 20), the resumed
                  loader would eventually produce DIFFERENT batches from the uninterrupted run.
        Slice: regression guard for _DOC_BUFFER_CHECKPOINT_CAP removal
        ### Given: batch-1 state_dict with its doc_buffer artificially capped to 20 entries
        ### When: resumed from that capped state_dict and run for 15 batches
        ### Then: at least one batch differs from the uninterrupted run, proving the cap
                  caused data loss (the first batch may coincidentally match, but once the
                  capped buffer depletes it refills from the wrong reader position)
        """
        parquet_path = _make_parquet(tmp_path, n_docs=250)
        monkeypatch.setattr(
            "nanochat.dataloader.list_parquet_files",
            lambda warn_on_legacy=False: [parquet_path, parquet_path],
        )

        tokenizer = _MockTokenizer()

        # Uninterrupted run: advance past batch 0 to get the state_dict with a large buffer.
        gen_full = _make_loader(tokenizer, parquet_path, buffer_size=100)
        next(gen_full)                               # batch 0 — empty buffer in state_dict
        _, _, saved_state = next(gen_full)           # batch 1 — large buffer in state_dict

        # Only run the falsification if the buffer is actually large (> 20).
        if len(saved_state["doc_buffer"]) <= 20:
            pytest.skip("Buffer too small to demonstrate the cap bug (need > 20 entries)")

        _N_BATCHES = 15  # enough for the capped buffer to need an early refill and diverge

        # Reference: collect N batches from the full run (starting at batch 1).
        ref_batches = []
        for _ in range(_N_BATCHES):
            x, _, _ = next(gen_full)
            ref_batches.append(x.clone())

        # Artificially apply the old cap.
        capped_state = dict(saved_state)
        capped_state["doc_buffer"] = saved_state["doc_buffer"][:20]

        gen_capped = _make_loader(tokenizer, parquet_path, resume_state_dict=capped_state, buffer_size=100)
        capped_batches = []
        for _ in range(_N_BATCHES):
            x, _, _ = next(gen_capped)
            capped_batches.append(x.clone())

        # With the cap applied, the capped run's buffer depletes below buffer_size faster
        # (only 20 docs vs 126), forcing an early refill from reader_position=(0,0,128).
        # This skips parquet docs 20–127, so at least one of the N batches will differ.
        assert any(
            not torch.equal(ref, cap) for ref, cap in zip(ref_batches, capped_batches)
        ), (
            "Expected at least one of the first 15 batches to differ between full and "
            "capped resume, but all were identical.  The cap may not cause data loss "
            "with this synthetic dataset."
        )
