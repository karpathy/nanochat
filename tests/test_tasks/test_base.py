"""Tests for task base classes and utilities."""

import pytest
from nanochat.tasks.base import Task, TaskMixture, TaskSequence, render_mc
from nanochat.tasks.types import Conversation, Message


class DummyTask(Task):
    """Simple task for testing."""
    
    def __init__(self, size: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.size = size
    
    @property
    def eval_type(self) -> str:
        return "categorical"
    
    def num_examples(self) -> int:
        return self.size
    
    def get_example(self, index: int) -> Conversation:
        return [
            {"role": "user", "content": f"Question {index}"},
            {"role": "assistant", "content": f"Answer {index}"}
        ]
    
    def evaluate(self, problem, completion):
        return problem == completion


def test_task_length():
    """Test task length calculation."""
    task = DummyTask(size=10)
    assert len(task) == 10


def test_task_slicing():
    """Test task slicing with start/stop/step."""
    task = DummyTask(size=20, start=5, stop=15, step=2)
    assert len(task) == 5  # (15-5)//2 = 5
    
    conv = task[0]
    assert conv[0]["content"] == "Question 5"


def test_task_mixture():
    """Test TaskMixture combines multiple tasks."""
    task1 = DummyTask(size=5)
    task2 = DummyTask(size=3)
    
    mixture = TaskMixture([task1, task2])
    assert len(mixture) == 8
    
    # Check all conversations are accessible
    for i in range(len(mixture)):
        conv = mixture[i]
        assert len(conv) == 2
        assert conv[0]["role"] == "user"


def test_task_sequence():
    """Test TaskSequence concatenates tasks."""
    task1 = DummyTask(size=5)
    task2 = DummyTask(size=3)
    
    sequence = TaskSequence([task1, task2])
    assert len(sequence) == 8
    
    # First 5 should be from task1
    conv = sequence[0]
    assert "Question" in conv[0]["content"]


def test_render_mc():
    """Test multiple choice rendering format."""
    question = "What is 2+2?"
    letters = ["A", "B", "C"]
    choices = ["3", "4", "5"]
    
    result = render_mc(question, letters, choices)
    
    assert "What is 2+2?" in result
    assert "- 3=A" in result
    assert "- 4=B" in result
    assert "- 5=C" in result
    assert "Respond only with the letter" in result
