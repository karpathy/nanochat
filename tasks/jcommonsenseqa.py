"""
JCommonsenseQA from JGLUE benchmark.
https://huggingface.co/datasets/sbintuitions/JCommonsenseQA

A Japanese commonsense question answering dataset with 5 choices.
Used for evaluating Japanese language understanding.
"""

from datasets import load_dataset
from tasks.common import Task, render_mc


class JCommonsenseQA(Task):
    """
    JCommonsenseQA: Japanese Commonsense Question Answering.
    A 5-choice multiple choice task from JGLUE benchmark.
    """

    def __init__(self, split="validation", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "validation"], "JCommonsenseQA split must be train|validation"
        self.ds = load_dataset("sbintuitions/JCommonsenseQA", split=split).shuffle(seed=42)
        self.letters = ["A", "B", "C", "D", "E"]

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        # Collect choices from choice0 to choice4
        choices = [row[f"choice{i}"] for i in range(5)]
        label = row["label"]  # 0-4
        answer_letter = self.letters[label]

        # Create the user message with multiple choice format
        user_message = render_mc(question, self.letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_letter}
        ]

        conversation = {
            "messages": messages,
            "letters": self.letters,  # useful during evaluation
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        # Check if the assistant's response matches the expected answer
        assert assistant_response in conversation['letters'], \
            f"JCommonsenseQA answer {assistant_response} must be one of {conversation['letters']}"
        expected_answer = conversation['messages'][-1]['content']  # e.g., "A"
        return assistant_response == expected_answer
