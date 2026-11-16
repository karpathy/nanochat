"""
Author: nanochat team
Created: 2026-11-15
Scope: NaN-guard for SFT training — safe loss on all-masked batches,
       fail-fast check_finite_loss helper, and collective_check_finite_loss
       (FIX 2: single-process path; collective path tested via mock dist).
"""

import pytest
import torch
import torch.nn.functional as F

from nanochat.common import check_finite_loss, collective_check_finite_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_logits_and_targets(vocab: int = 32, seq: int = 4) -> tuple:
    """Return (logits [1, seq, vocab], targets [1, seq]) with valid token ids."""
    torch.manual_seed(0)
    logits = torch.randn(1, seq, vocab, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, vocab, (1, seq), dtype=torch.long)
    return logits, targets


def _compute_loss_safe(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Replicate the fixed loss logic from gpt.py for unit-testing without the full model."""
    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = targets.view(-1)
    n_valid = (flat_targets != -1).sum()
    if n_valid == 0:
        return (flat_logits * 0).sum()
    return F.cross_entropy(flat_logits, flat_targets, ignore_index=-1, reduction='sum') / n_valid


# ---------------------------------------------------------------------------
# Fix 1: safe loss for all-masked micro-batches
# ---------------------------------------------------------------------------

class TestSafeLossAllMasked:
    def test_all_ignored_targets_gives_finite_loss(self):
        """
        Scenario: micro-batch where every target position is -1 (fully masked).
        Slice: gpt.py safe-mean path with n_valid == 0.

        Given a logits tensor and targets that are all -1 (ignore_index),
        When the safe loss function is applied,
        Then the returned loss is finite (not NaN, not inf).
        """
        ### Given
        logits, _ = _small_logits_and_targets()
        targets_all_masked = torch.full((1, 4), -1, dtype=torch.long)

        ### When
        loss = _compute_loss_safe(logits, targets_all_masked)

        ### Then
        assert torch.isfinite(loss), f"Expected finite loss, got {loss.item()}"

    def test_all_ignored_targets_gives_zero_loss(self):
        """
        Scenario: all-masked batch produces zero loss value.
        Slice: gpt.py safe-mean path with n_valid == 0.

        Given all targets are -1,
        When the safe loss is computed,
        Then the loss value is exactly 0.0.
        """
        ### Given
        logits, _ = _small_logits_and_targets()
        targets_all_masked = torch.full((1, 4), -1, dtype=torch.long)

        ### When
        loss = _compute_loss_safe(logits, targets_all_masked)

        ### Then
        assert loss.item() == 0.0, f"Expected 0.0 loss, got {loss.item()}"

    def test_all_ignored_targets_gives_zero_gradient(self):
        """
        Scenario: all-masked batch produces zero gradient on logits.
        Slice: gpt.py safe-mean path with n_valid == 0.

        Given all targets are -1,
        When backward is called on the safe loss,
        Then the gradient on logits is all zeros (no weight update from this batch).
        """
        ### Given
        logits, _ = _small_logits_and_targets()
        targets_all_masked = torch.full((1, 4), -1, dtype=torch.long)

        ### When
        loss = _compute_loss_safe(logits, targets_all_masked)
        loss.backward()

        ### Then
        assert logits.grad is not None
        assert logits.grad.abs().max().item() == 0.0, "Expected zero gradient for all-masked batch"

    def test_normal_batch_loss_matches_stock_cross_entropy(self):
        """
        Scenario: a batch with no masked tokens should give the same result as
                  the original F.cross_entropy(..., reduction='mean').
        Slice: gpt.py safe-mean path with n_valid == seq_len.

        Given logits and fully-unmasked targets,
        When the safe loss is computed,
        Then it equals F.cross_entropy with reduction='mean' to within fp32 tolerance.
        """
        ### Given
        logits, targets = _small_logits_and_targets()

        ### When
        safe_loss = _compute_loss_safe(logits, targets)
        reference_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction='mean',
        )

        ### Then
        assert torch.allclose(safe_loss, reference_loss, atol=1e-6), (
            f"Safe loss {safe_loss.item()} != reference {reference_loss.item()}"
        )

    def test_partial_mask_loss_matches_stock_cross_entropy(self):
        """
        Scenario: a partially masked batch (some -1 targets) should agree with the
                  reference implementation.
        Slice: gpt.py safe-mean path with 0 < n_valid < seq_len.

        Given targets where some positions are -1,
        When the safe loss is computed,
        Then it matches F.cross_entropy with the same ignore_index and reduction='mean'.
        """
        ### Given
        logits, targets = _small_logits_and_targets(seq=6)
        targets[0, 1] = -1
        targets[0, 3] = -1

        ### When
        safe_loss = _compute_loss_safe(logits, targets)
        reference_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction='mean',
        )

        ### Then
        assert torch.allclose(safe_loss, reference_loss, atol=1e-6), (
            f"Safe loss {safe_loss.item()} != reference {reference_loss.item()}"
        )


# ---------------------------------------------------------------------------
# Fix 2: check_finite_loss helper
# ---------------------------------------------------------------------------

class TestCheckFiniteLoss:
    def _make_targets(self, n_unmasked: int = 2, total: int = 4) -> torch.Tensor:
        t = torch.full((1, total), -1, dtype=torch.long)
        for i in range(n_unmasked):
            t[0, i] = i + 1
        return t

    def test_finite_loss_does_not_raise(self):
        """
        Scenario: normal finite loss passes through without error.
        Slice: check_finite_loss happy path.

        Given a finite loss tensor,
        When check_finite_loss is called,
        Then no exception is raised.
        """
        ### Given
        loss = torch.tensor(1.5)
        targets = self._make_targets(n_unmasked=2)

        ### When / Then
        check_finite_loss(loss, step=0, targets=targets)  # must not raise

    def test_nan_loss_raises_runtime_error(self):
        """
        Scenario: NaN loss triggers immediate RuntimeError.
        Slice: check_finite_loss NaN path.

        Given a NaN loss tensor,
        When check_finite_loss is called,
        Then a RuntimeError is raised.
        """
        ### Given
        loss = torch.tensor(float('nan'))
        targets = self._make_targets(n_unmasked=0)

        ### When / Then
        with pytest.raises(RuntimeError):
            check_finite_loss(loss, step=4, targets=targets)

    def test_inf_loss_raises_runtime_error(self):
        """
        Scenario: infinite loss triggers immediate RuntimeError.
        Slice: check_finite_loss inf path.

        Given an infinite loss tensor,
        When check_finite_loss is called,
        Then a RuntimeError is raised.
        """
        ### Given
        loss = torch.tensor(float('inf'))
        targets = self._make_targets(n_unmasked=1)

        ### When / Then
        with pytest.raises(RuntimeError):
            check_finite_loss(loss, step=7, targets=targets)

    def test_error_message_contains_step(self):
        """
        Scenario: error message names the failing step for fast diagnosis.
        Slice: check_finite_loss message content.

        Given a NaN loss at step 42,
        When check_finite_loss is called,
        Then the error message contains the step number.
        """
        ### Given
        loss = torch.tensor(float('nan'))
        targets = self._make_targets(n_unmasked=0)

        ### When
        with pytest.raises(RuntimeError, match="42"):
            check_finite_loss(loss, step=42, targets=targets)

    def test_error_message_contains_unmasked_count(self):
        """
        Scenario: error message reports how many unmasked tokens the batch had.
        Slice: check_finite_loss message content.

        Given a NaN loss with 3 unmasked tokens,
        When check_finite_loss is called,
        Then the error message contains the unmasked count.
        """
        ### Given
        loss = torch.tensor(float('nan'))
        targets = self._make_targets(n_unmasked=3, total=8)

        ### When
        with pytest.raises(RuntimeError, match="3"):
            check_finite_loss(loss, step=1, targets=targets)


# ---------------------------------------------------------------------------
# FIX 2: collective_check_finite_loss — single-process path (no dist)
# ---------------------------------------------------------------------------

class TestCollectiveCheckFiniteLoss:
    """
    Scenario: FIX 2 — collective NaN guard delegates to check_finite_loss on
              a single process and raises on all ranks when dist is active.
    Slice: nanochat.common.collective_check_finite_loss

    The collective (multi-rank) path is not exercised here because tests run
    single-process.  It is covered by the mock-dist tests below.
    """

    def _targets(self, n: int = 4, n_unmasked: int = 2) -> torch.Tensor:
        t = torch.full((1, n), -1, dtype=torch.long)
        for i in range(n_unmasked):
            t[0, i] = i + 1
        return t

    def test_finite_loss_does_not_raise_single_process(self):
        """
        Scenario: single-process run with a finite loss — no exception.
        Slice: collective_check_finite_loss single-process happy path.

        Given a finite loss and not-distributed,
        When collective_check_finite_loss is called,
        Then no exception is raised.
        """
        ### Given
        loss = torch.tensor(1.0)
        targets = self._targets()
        ### When / Then
        collective_check_finite_loss(loss, step=0, targets=targets)

    def test_nan_loss_raises_single_process(self):
        """
        Scenario: single-process run with NaN loss — raises RuntimeError.
        Slice: collective_check_finite_loss single-process NaN path.

        Given a NaN loss and not-distributed,
        When collective_check_finite_loss is called,
        Then RuntimeError is raised (delegates to check_finite_loss).
        """
        ### Given
        loss = torch.tensor(float('nan'))
        targets = self._targets()
        ### When / Then
        with pytest.raises(RuntimeError):
            collective_check_finite_loss(loss, step=5, targets=targets)

    def test_inf_loss_raises_single_process(self):
        """
        Scenario: single-process run with infinite loss — raises RuntimeError.
        Slice: collective_check_finite_loss single-process inf path.

        Given an infinite loss,
        When collective_check_finite_loss is called,
        Then RuntimeError is raised.
        """
        ### Given
        loss = torch.tensor(float('inf'))
        targets = self._targets()
        ### When / Then
        with pytest.raises(RuntimeError):
            collective_check_finite_loss(loss, step=3, targets=targets)

    def test_collective_raises_on_all_ranks_when_flag_set(self, monkeypatch):
        """
        Scenario: FIX 2 collective path — when the all-reduced flag is 1, every rank raises.
        Slice: collective_check_finite_loss mock-dist path.

        Given a mocked is_ddp_initialized() returning True and a mocked dist.all_reduce
              that always sets the flag to 1 (simulating a peer rank that saw NaN),
        When collective_check_finite_loss is called with a FINITE local loss,
        Then RuntimeError is raised on this rank too (collective decision).
        """
        import torch.distributed as dist
        from nanochat import common as common_mod

        ### Given: fake dist is "initialized"
        monkeypatch.setattr(common_mod, "is_ddp_initialized", lambda: True)

        # Simulate an all_reduce that always sets flag to 1 (some rank saw NaN).
        def _fake_all_reduce(tensor, op):
            tensor.fill_(1.0)

        monkeypatch.setattr(dist, "all_reduce", _fake_all_reduce)

        loss = torch.tensor(0.5)   # locally finite
        targets = self._targets(n_unmasked=2)

        ### When / Then
        with pytest.raises(RuntimeError):
            collective_check_finite_loss(loss, step=10, targets=targets)

    def test_collective_does_not_raise_when_flag_zero(self, monkeypatch):
        """
        Scenario: all-reduced flag is 0 — no rank saw NaN; must not raise.
        Slice: collective_check_finite_loss mock-dist no-raise path.

        Given a mocked dist that keeps the flag at 0 (all ranks finite),
        When collective_check_finite_loss is called with a finite loss,
        Then no exception is raised.
        """
        import torch.distributed as dist
        from nanochat import common as common_mod

        ### Given
        monkeypatch.setattr(common_mod, "is_ddp_initialized", lambda: True)

        def _fake_all_reduce(tensor, op):
            pass  # leave flag at 0

        monkeypatch.setattr(dist, "all_reduce", _fake_all_reduce)

        loss = torch.tensor(1.5)
        targets = self._targets(n_unmasked=2)

        ### When / Then (no exception)
        collective_check_finite_loss(loss, step=7, targets=targets)

    # ------------------------------------------------------------------
    # S1 + S2: nan_flag parameter — multi-micro-step accumulation
    # ------------------------------------------------------------------

    def test_nan_flag_true_raises_on_finite_last_loss(self):
        """
        Scenario: S1 — a NaN occurred in an earlier micro-step; the last
                  micro-step's loss is finite.  The old code (checking only
                  the last loss) would NOT have raised; the new code must.
        Slice: collective_check_finite_loss nan_flag=True single-process path.

        Given nan_flag=True (earlier micro-step had NaN) and a FINITE last loss,
        When collective_check_finite_loss is called,
        Then RuntimeError is raised — proving the flag covers ALL micro-steps.

        Old-code counterfactual: calling collective_check_finite_loss(finite_loss, ...)
        without nan_flag would NOT raise, silently swallowing the earlier NaN.
        """
        ### Given
        loss = torch.tensor(1.0)   # finite — the LAST micro-step was fine
        targets = self._targets(n_unmasked=2)
        ### When / Then
        with pytest.raises(RuntimeError):
            collective_check_finite_loss(loss, step=3, targets=targets, nan_flag=True)

    def test_nan_flag_false_does_not_raise_on_nan_loss(self):
        """
        Scenario: nan_flag=False explicitly says this rank had no NaN; must not raise
                  even if the loss tensor is non-finite (caller says no NaN this rank).
        Slice: collective_check_finite_loss nan_flag=False single-process path.

        Given nan_flag=False and a NaN loss tensor,
        When collective_check_finite_loss is called,
        Then no exception is raised (caller's accumulated flag takes precedence).
        """
        ### Given
        loss = torch.tensor(float('nan'))
        targets = self._targets(n_unmasked=2)
        ### When / Then (no exception — caller says no NaN on this rank)
        collective_check_finite_loss(loss, step=0, targets=targets, nan_flag=False)

    def test_nan_flag_reports_nan_batch_unmasked_count(self):
        """
        Scenario: S2 — the error message must report unmasked tokens from the
                  micro-batch that produced the NaN, not from a later prefetched batch.
        Slice: collective_check_finite_loss nan_flag=True targets-reporting path.

        Given nan_flag=True, a finite last loss, and targets with exactly 3
              unmasked tokens (the NaN batch),
        When collective_check_finite_loss is called,
        Then the RuntimeError message contains '3'.
        """
        ### Given: targets from the NaN micro-batch have 3 unmasked tokens
        nan_batch_targets = self._targets(n=8, n_unmasked=3)
        loss = torch.tensor(1.0)  # last micro-step was fine
        ### When / Then
        with pytest.raises(RuntimeError, match="3"):
            collective_check_finite_loss(
                loss, step=0, targets=nan_batch_targets, nan_flag=True
            )

    def test_nan_flag_none_behaves_like_old_api(self):
        """
        Scenario: nan_flag=None (default) is backward-compatible — behaviour
                  is identical to the original signature.
        Slice: collective_check_finite_loss nan_flag=None backward-compat path.

        Given nan_flag=None (omitted) and a NaN loss,
        When collective_check_finite_loss is called,
        Then RuntimeError is raised (same as the original API).
        """
        ### Given
        loss = torch.tensor(float('nan'))
        targets = self._targets(n_unmasked=1)
        ### When / Then
        with pytest.raises(RuntimeError):
            collective_check_finite_loss(loss, step=1, targets=targets)
