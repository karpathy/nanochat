"""Tests for the SFT packing and zero-token guards in nanochat/sft_loader.py.

Author: nanochat maintainers
Created: 2026-11-16
Scope: is_packable (oversize conversations) and next_zero_token_streak (empty-batch halt).
"""
from nanochat.sft_loader import (
    ZERO_TOKEN_STREAK_LIMIT,
    is_packable,
    next_zero_token_streak,
)


class TestIsPackable:
    def test_conversation_exactly_one_row_is_packable(self):
        """Scenario: a conversation of exactly row_capacity tokens fits in one row.
        Slice: is_packable boundary.
        ### Given: 513 token ids and a row capacity of 513
        ### When: is_packable is called
        ### Then: it returns True
        """
        # -- Given --
        ids = list(range(513))
        # -- When --
        result = is_packable(ids, row_capacity=513)
        # -- Then --
        assert result is True

    def test_conversation_one_token_too_long_is_skipped(self):
        """Scenario: a conversation one token over row_capacity can never be placed.
        Slice: is_packable boundary (the stall seen in the step-572 buffer).
        ### Given: 514 token ids and a row capacity of 513
        ### When: is_packable is called
        ### Then: it returns False
        """
        # -- Given --
        ids = list(range(514))
        # -- When --
        result = is_packable(ids, row_capacity=513)
        # -- Then --
        assert result is False

    def test_buffer_of_only_oversize_conversations_has_nothing_packable(self):
        """Scenario: the saved stall state, where every buffered conversation is too long.
        Slice: is_packable over a buffer like the step-572 checkpoint (min 517 tokens).
        ### Given: 100 conversations of 517 to 2048 tokens and a row capacity of 513
        ### When: each is checked with is_packable
        ### Then: none is packable, so the loader would never consume one
        """
        # -- Given --
        lengths = [517 + (i * 15) for i in range(100)]
        buffer = [list(range(n)) for n in lengths]
        # -- When --
        packable = [ids for ids in buffer if is_packable(ids, row_capacity=513)]
        # -- Then --
        assert packable == []


class TestNextZeroTokenStreak:
    def test_zero_supervised_tokens_increments_the_streak(self):
        """Scenario: an optimizer step with no supervised tokens extends the streak.
        Slice: next_zero_token_streak increment.
        ### Given: a streak of 3
        ### When: a step with 0 supervised tokens is recorded
        ### Then: the streak is 4
        """
        # -- Given --
        streak = 3
        # -- When --
        result = next_zero_token_streak(streak, supervised_tokens=0)
        # -- Then --
        assert result == 4

    def test_any_supervised_token_resets_the_streak(self):
        """Scenario: one step with supervised tokens breaks the run of empty steps.
        Slice: next_zero_token_streak reset.
        ### Given: a streak of 7
        ### When: a step with 12 supervised tokens is recorded
        ### Then: the streak is 0
        """
        # -- Given --
        streak = 7
        # -- When --
        result = next_zero_token_streak(streak, supervised_tokens=12)
        # -- Then --
        assert result == 0

    def test_streak_reaches_limit_after_limit_empty_steps(self):
        """Scenario: the halt condition trips after exactly ZERO_TOKEN_STREAK_LIMIT empty steps.
        Slice: next_zero_token_streak against the halt limit.
        ### Given: a streak of 0
        ### When: ZERO_TOKEN_STREAK_LIMIT steps with no supervised tokens are recorded in a row
        ### Then: the streak equals ZERO_TOKEN_STREAK_LIMIT
        """
        # -- Given --
        streak = 0
        # -- When --
        for _ in range(ZERO_TOKEN_STREAK_LIMIT):
            streak = next_zero_token_streak(streak, supervised_tokens=0)
        # -- Then --
        assert streak == ZERO_TOKEN_STREAK_LIMIT
