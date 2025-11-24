"""
This module provides the base classes and common utilities for defining evaluation
and training tasks in nanochat.

The core components are:
- `Task`: An abstract base class that represents a dataset of conversations.
- `TaskMixture`: A class for combining multiple tasks into a single, shuffled dataset.
- `TaskSequence`: A class for combining multiple tasks in a sequential manner.
- `render_mc`: A helper function for formatting multiple-choice questions.
"""

import random

class Task:
    """
    Abstract base class for a task, which is essentially a dataset of conversations.

    This class supports lightweight slicing of the underlying dataset using `start`,
    `stop`, and `step` parameters, similar to Python's list slicing.
    """

    def __init__(self, start=0, stop=None, step=1):
        # allows a lightweight logical view over a dataset
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, f"Stop should be greater than or equal to start, got {stop} and {start}"
        assert step >= 1, f"Step must be strictly positive, got {step}"
        self.start = start
        self.stop = stop # could be None here
        self.step = step

    @property
    def eval_type(self):
        """The type of evaluation for this task, either 'generative' or 'categorical'."""
        # one of 'generative' | 'categorical'
        raise NotImplementedError

    def num_examples(self):
        """Returns the total number of examples in the underlying dataset."""
        raise NotImplementedError

    def get_example(self, index):
        """Retrieves a single example from the underlying dataset by its physical index."""
        raise NotImplementedError

    def __len__(self):
        """Returns the number of examples in the (potentially sliced) view of the dataset."""
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step # ceil_div(span, step)
        assert num >= 0, f"Negative number of examples???: {num}" # prevent footguns
        return num

    def __getitem__(self, index: int):
        """Retrieves an example by its logical index in the (sliced) view."""
        assert isinstance(index, int), f"Index must be an integer, got {type(index)}"
        physical_index = self.start + index * self.step
        conversation = self.get_example(physical_index)
        return conversation

    def evaluate(self, problem, completion):
        """Evaluates a model's completion for a given problem."""
        raise NotImplementedError


class TaskMixture(Task):
    """
    Combines multiple tasks into a single, deterministically shuffled dataset.
    This is useful for creating a diverse training mixture for SFT. To oversample
    a task, simply include it multiple times in the `tasks` list.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        # tasks is a list of Task objects
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)
        # Build list of all (task_idx, local_idx) pairs
        self.index_map = []
        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))
        # Deterministically shuffle to mix tasks throughout training
        rng = random.Random(42)
        rng.shuffle(self.index_map)
        # Note: this is not the most elegant or best solution, but it's ok for now

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        """
        Access conversations according to a deterministic shuffle of all examples.
        This ensures tasks are mixed throughout training, regardless of dataset size.
        """
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for mixture with {self.num_conversations} conversations"
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


class TaskSequence(Task):
    """
    Combines multiple tasks sequentially, which is useful for creating a
    training curriculum.
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for sequence with {self.num_conversations} conversations"
        for task_idx, task_length in enumerate(self.lengths):
            if index < task_length:
                return self.tasks[task_idx][index]
            index -= task_length


def render_mc(question, letters, choices):
    """
    Formats a multiple-choice question into a standardized prompt.

    Args:
        question (str): The question text.
        letters (list[str]): The letters for the choices (e.g., ['A', 'B', 'C']).
        choices (list[str]): The text of the choices.

    Returns:
        str: The formatted prompt.
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query


if __name__ == "__main__":
    # very lightweight test of slicing
    from tasks.mmlu import MMLU

    ds = MMLU(subset="auxiliary_train", split="train")
    print("Length of MMLU: ", len(ds))
    ex = ds[5]
    print("5th example: ", ex)

    ds = MMLU(subset="auxiliary_train", split="train", start=5, stop=10)
    print("Length of sliced MMLU[5:10]: ", len(ds))
    print("0th example of sliced MMLU: ", ds[0])

    print("They match: ", ex == ds[0])
