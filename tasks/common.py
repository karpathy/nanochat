"""
Base class for all Tasks.
A Task is basically a dataset of conversations, together with some
metadata and often also evaluation criteria.
Example tasks: MMLU, ARC-Easy, ARC-Challenge, GSM8K, HumanEval, SmolTalk.
"""

import random

class Task:
    """
    Base class of a Task. Allows for lightweight slicing of the underlying dataset.
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
        # one of 'generative' | 'categorical'
        raise NotImplementedError

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def __len__(self):
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step # ceil_div(span, step)
        assert num >= 0, f"Negative number of examples???: {num}" # prevent footguns
        return num

    def __getitem__(self, index: int):
        assert isinstance(index, int), f"Index must be an integer, got {type(index)}"
        physical_index = self.start + index * self.step
        conversation = self.get_example(physical_index)
        return conversation

    def evaluate(self, problem, completion):
        raise NotImplementedError


class TaskMixture(Task):
    """
    For SFT Training it becomes useful to train on a mixture of datasets.
    Fun trick: if you wish to oversample any task, just pass it in multiple times in the list.
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
    For SFT Training sometimes we want to sequentially train on a list of tasks.
    This is useful for cases that require a training curriculum.
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
    The common multiple choice rendering format we will use.

    Note two important design decisions:
    1)
    Bigger models don't care as much, but smaller models prefer to have
    the letter *after* the choice, which results in better binding.
    2)
    There is no whitespace between the delimiter (=) and the letter.
    This is actually critical because the tokenizer has different token ids
    for " A" vs. "A". The assistant responses will be just the letter itself,
    i.e. "A", so it is important that here in the prompt it is the exact same
    token, i.e. "A" with no whitespace before it. Again, bigger models don't care
    about this too much, but smaller models do care about some of these details.
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
