"""Focused tests for SFT best-fit packing fallbacks."""

from nanochat.dataloader import has_sft_supervised_tokens, pop_best_fit_conversation


def conversation(length, supervised_positions=()):
    mask = [0] * length
    for position in supervised_positions:
        mask[position] = 1
    return list(range(length)), mask


def test_best_fit_is_preferred_over_truncation():
    oversized = conversation(8, supervised_positions=(6,))
    exact_fit = conversation(5, supervised_positions=(4,))
    buffer = [oversized, exact_fit]

    selected = pop_best_fit_conversation(buffer, max_length=5, truncate_if_needed=True)

    assert selected == exact_fit
    assert buffer == [oversized]


def test_shortest_oversized_conversation_is_truncated_and_removed():
    longest = conversation(9, supervised_positions=(8,))
    shortest = conversation(7, supervised_positions=(4, 6))
    buffer = [longest, shortest]

    ids, mask = pop_best_fit_conversation(buffer, max_length=5, truncate_if_needed=True)

    assert ids == shortest[0][:5]
    assert mask == shortest[1][:5]
    assert buffer == [longest]


def test_partial_row_does_not_truncate_oversized_conversation():
    oversized = conversation(8)
    buffer = [oversized]

    assert pop_best_fit_conversation(buffer, max_length=3) is None
    assert buffer == [oversized]


def test_supervised_target_detection_accounts_for_shift():
    assert not has_sft_supervised_tokens([[0, 0, 0], [0, 0, 0]])
    assert not has_sft_supervised_tokens([[1, 0, 0]])
    assert has_sft_supervised_tokens([[0, 0, 1]])
