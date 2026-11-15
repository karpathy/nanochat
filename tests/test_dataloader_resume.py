"""
Author: Mani Sarkar
Created: 2026-11-14
Scope: nanochat/dataloader.py resume position per DDP rank (review blocker B3).
       Rank 0 writes the checkpoint, so other ranks must re-derive their own row group.
"""
import pytest

from nanochat import dataloader


def _as_rank(monkeypatch, rank: int, world: int) -> None:
    monkeypatch.setattr(dataloader, "get_dist_info", lambda: (True, rank, 0, world))


class TestStartPositionPerRank:
    """
    Scenario: Each DDP rank resumes on its own row group, not rank 0's.
    Slice: nanochat.dataloader._start_position
    """

    @pytest.mark.parametrize("rank,saved_rg,expected", [
        (0, 4, 4),   # rank 0 keeps its saved group
        (1, 4, 5),   # rank 1 takes the same stride block: 4 + 1
        (2, 5, 6),   # saved group 5 is in block [4..7]; rank 2 -> 6
        (3, 4, 7),   # rank 3 -> 7
    ])
    def test_ranks_get_distinct_row_groups(self, monkeypatch, rank, saved_rg, expected):
        """
        Scenario: World size 4, checkpoint saved by rank 0 at row group saved_rg.
        Slice: _start_position rg derivation
        ### Given: world_size=4 and state rg_idx=saved_rg
        ### When: _start_position runs for this rank
        ### Then: rg_idx equals expected (distinct per rank)
        """
        _as_rank(monkeypatch, rank, world=4)
        _, rg_idx, _, _ = dataloader._start_position({"rg_idx": saved_rg, "pq_idx": 0, "epoch": 1})
        assert rg_idx == expected

    def test_single_process_resume_is_unchanged(self, monkeypatch):
        """
        Scenario: world_size=1 (local CPU/MPS run).
        Slice: _start_position
        ### Given: rg_idx=7, doc_offset=3
        ### When: resumed on rank 0 of 1
        ### Then: position is exactly the saved one
        """
        _as_rank(monkeypatch, 0, world=1)
        state = {"pq_idx": 2, "rg_idx": 7, "doc_offset": 3, "epoch": 2}
        assert dataloader._start_position(state) == (2, 7, 3, 2)

    def test_no_saved_state_starts_at_rank_index(self, monkeypatch):
        """
        Scenario: Fresh start, no resume state.
        Slice: _start_position default
        ### Given: no state
        ### When: rank 3 of 4 starts
        ### Then: row group is 3 and offset 0
        """
        _as_rank(monkeypatch, 3, world=4)
        assert dataloader._start_position(None) == (0, 3, 0, 1)

    def test_non_zero_rank_does_not_reuse_rank0_doc_offset(self, monkeypatch):
        """
        Scenario: doc_offset in the checkpoint points inside rank 0's row group.
        Slice: _start_position doc_offset
        ### Given: doc_offset=5 saved by rank 0
        ### When: rank 1 resumes
        ### Then: rank 1 starts at offset 0 (its row group differs)
        """
        _as_rank(monkeypatch, 1, world=2)
        _, _, doc_offset, _ = dataloader._start_position({"rg_idx": 2, "doc_offset": 5})
        assert doc_offset == 0


class TestResumeDocBuffer:
    """
    Scenario: The saved document buffer belongs to rank 0 only.
    Slice: nanochat.dataloader._resume_doc_buffer
    """

    def test_rank0_keeps_saved_buffer(self, monkeypatch):
        """
        ### Given: rank 0 and a saved buffer
        ### When: buffer is restored
        ### Then: buffer is returned as a list copy
        """
        _as_rank(monkeypatch, 0, world=2)
        assert dataloader._resume_doc_buffer({"doc_buffer": [[1, 2], [3]]}) == [[1, 2], [3]]

    def test_non_zero_rank_starts_with_empty_buffer(self, monkeypatch):
        """
        ### Given: rank 1 and a buffer saved by rank 0
        ### When: buffer is restored
        ### Then: empty, so refill_buffer loads this rank's own data
        """
        _as_rank(monkeypatch, 1, world=2)
        assert dataloader._resume_doc_buffer({"doc_buffer": [[1, 2]]}) == []

    def test_no_state_gives_empty_buffer(self, monkeypatch):
        """
        ### Given: no resume state
        ### When: buffer is restored
        ### Then: empty list
        """
        _as_rank(monkeypatch, 0, world=1)
        assert dataloader._resume_doc_buffer(None) == []
