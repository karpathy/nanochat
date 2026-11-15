"""
Author: nanochat team
Created: 2026-11-15
Scope: NaN-guard for SFT training — safe loss on all-masked batches and
       fail-fast check_finite_loss helper.
"""

import pytest
import torch
import torch.nn.functional as F

from nanochat.common import check_finite_loss


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
