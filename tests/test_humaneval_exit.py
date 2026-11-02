"""Premature interpreter exit must not bypass HumanEval's appended check."""
import pytest

from tasks.humaneval import HumanEval


def conversation():
    return {
        "messages": [{"role": "user", "content": "def add_one(x):\n    pass"}],
        "entry_point": "add_one",
        "test": "def check(candidate):\n    assert candidate(1) == 2",
    }


@pytest.mark.parametrize("completion", ["import sys; sys.exit(0)", "raise SystemExit(0)", "import os; os._exit(0)"])
def test_humaneval_rejects_premature_successful_exit(completion):
    task = HumanEval.__new__(HumanEval)  # no dataset/network needed for evaluation
    assert not task.evaluate(conversation(), completion)


def test_humaneval_still_checks_correct_and_incorrect_programs():
    task = HumanEval.__new__(HumanEval)
    assert task.evaluate(conversation(), "def add_one(x):\n    return x + 1")
    assert not task.evaluate(conversation(), "def add_one(x):\n    return x")
