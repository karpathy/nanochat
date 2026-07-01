"""
Tests for few-shot index sampling in core_eval.

Run: python -m pytest tests/test_core_eval.py -v
"""

from nanochat.core_eval import sample_fewshot_indices


def test_returns_requested_count_when_population_is_large_enough():
    indices = sample_fewshot_indices(idx=0, num_fewshot=3, data_len=10)
    assert len(indices) == 3
    assert len(set(indices)) == 3  # no duplicates
    assert 0 not in indices  # current item is excluded


def test_excludes_current_item_for_every_index():
    for idx in range(5):
        indices = sample_fewshot_indices(idx=idx, num_fewshot=2, data_len=5)
        assert idx not in indices
        assert all(0 <= i < 5 for i in indices)


def test_clamps_when_population_smaller_than_num_fewshot():
    # Reproduces #550: --core-metric-max-per-task restricts the task data below
    # a task's hardcoded num_fewshot, which previously raised ValueError from
    # random.sample("Sample larger than population").
    indices = sample_fewshot_indices(idx=0, num_fewshot=10, data_len=3)
    assert len(indices) == 2  # only two other examples available
    assert set(indices) == {1, 2}
    assert 0 not in indices


def test_returns_empty_when_only_one_example_available():
    indices = sample_fewshot_indices(idx=0, num_fewshot=5, data_len=1)
    assert indices == []


def test_is_deterministic_per_index():
    a = sample_fewshot_indices(idx=4, num_fewshot=3, data_len=20)
    b = sample_fewshot_indices(idx=4, num_fewshot=3, data_len=20)
    assert a == b


def test_zero_fewshot_returns_empty():
    assert sample_fewshot_indices(idx=0, num_fewshot=0, data_len=10) == []
