"""
This module implements the AI2 Reasoning Challenge (ARC) task. The ARC dataset is
a collection of multiple-choice science questions designed to test a model's
reasoning and common-sense knowledge.

**Reference:**
- The ARC dataset: https://huggingface.co/datasets/allenai/ai2_arc
"""

from datasets import load_dataset
from .common import Task, render_mc

class ARC(Task):
    """
    The ARC (AI2 Reasoning Challenge) task.

    Args:
        subset (str): "ARC-Easy" or "ARC-Challenge".
        split (str): "train", "validation", or "test".
    """

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["ARC-Easy", "ARC-Challenge"], "ARC subset must be ARC-Easy or ARC-Challenge"
        assert split in ["train", "validation", "test"], "ARC split must be train|validation|test"
        self.ds = load_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        """Specifies that this is a categorical evaluation task."""
        return 'categorical'

    def num_examples(self):
        """Returns the total number of examples in the dataset."""
        return len(self.ds)

    def get_example(self, index):
        """
        Formats a single example from the dataset into a conversation dictionary.
        """
        row = self.ds[index]
        question = row["question"] # the question text
        choices = row["choices"]["text"] # the text of each choice
        answer_string = row["answerKey"] # e.g. "A", "B", "C", "D"
        letters = row["choices"]["label"] # e.g. ["A", "B", "C", "D"]
        assert answer_string in letters, f"ARC answer {answer_string} must be one of {letters}" # sanity check
        # create and return the Conversation object
        user_message = render_mc(question, letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string}
        ]
        conversation = {
            "messages": messages,
            "letters": letters, # useful during evaluation, so we can narrow and clamp the assistant prediction to one of the letters
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Evaluates the model's response for a given example.

        Args:
            conversation (dict): The conversation dictionary for the example.
            assistant_response (str): The model's predicted answer.

        Returns:
            bool: True if the prediction is correct, False otherwise.
        """
        # the assert here is not strictly speaking needed, but currently the way we eval, we expect this to be true
        # I'm going to leave the assert here to prevent footguns, but possibly in the future can remove it.
        assert assistant_response in conversation['letters'], f"ARC answer {assistant_response} is expected to be one of {conversation['letters']}"
        assistant_message = conversation['messages'][-1]['content'] # e.g. "A"
        return assistant_response == assistant_message
