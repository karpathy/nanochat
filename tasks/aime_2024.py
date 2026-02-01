"""
AIME 2024 evaluation.
https://huggingface.co/datasets/MathArena/aime_2024_I
https://huggingface.co/datasets/MathArena/aime_2024_II

AIME (American Invitational Mathematics Examination) problems have integer answers.
We evaluate by extracting the answer from \boxed{} and doing exact string match.
"""

import json
import re
import os
from tasks.common import Task


# Regex to extract answer from \boxed{...}
BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def extract_boxed_answer(completion):
    """
    Extract the answer from \boxed{} in the completion.
    Returns the content inside the box, or None if not found.
    """
    # Find all boxed answers
    matches = BOXED_RE.findall(completion)
    if matches:
        # Take the last one (most likely the final answer)
        return matches[-1].strip()
    return None


class AIME2024I(Task):
    """AIME 2024 Part I (15 problems)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        data_path = os.path.join(os.path.dirname(__file__), "..", "dev-aime", "data", "aime_2024_I.jsonl")
        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.data)

    def get_example(self, index):
        """Get a single problem from the dataset."""
        row = self.data[index]
        problem = row["problem"]
        answer = row["answer"]
        
        # Add the instruction suffix
        instruction = "\nPlease reason step by step, and put your final answer within \\boxed{}."
        prompt = problem + instruction
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        conversation = {
            "messages": messages,
            "question_id": row["question_id"],
            "answer": answer,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        """
        # Extract ground truth answer
        ground_truth = conversation["answer"]
        
        # Extract predicted answer from boxed
        predicted = extract_boxed_answer(assistant_response)
        
        # Exact string match
        is_correct = int(predicted == ground_truth)
        return is_correct


class AIME2024II(Task):
    """AIME 2024 Part II (15 problems)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        data_path = os.path.join(os.path.dirname(__file__), "..", "dev-aime", "data", "aime_2024_II.jsonl")
        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.data)

    def get_example(self, index):
        """Get a single problem from the dataset."""
        row = self.data[index]
        problem = row["problem"]
        answer = row["answer"]
        
        # Add the instruction suffix
        instruction = "\nPlease reason step by step, and put your final answer within \\boxed{}."
        prompt = problem + instruction
        
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        conversation = {
            "messages": messages,
            "question_id": row["question_id"],
            "answer": answer,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        """
        # Extract ground truth answer
        ground_truth = conversation["answer"]
        
        # Extract predicted answer from boxed
        predicted = extract_boxed_answer(assistant_response)
        
        # Exact string match
        is_correct = int(predicted == ground_truth)
        return is_correct


class AIME2024(Task):
    """AIME 2024 (30 problems - both Part I and Part II)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        data_path = os.path.join(os.path.dirname(__file__), "..", "dev-aime", "data", "aime_2024.jsonl")
        self.data = []
        with open(data_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.data)

    def get_example(self, index):
        """Get a single problem from the dataset."""
        row = self.data[index]
        problem = row["problem"]
        answer = row["answer"]

        # Add the instruction suffix
        instruction = "\nPlease reason step by step, and put your final answer within \\boxed{}."
        prompt = problem + instruction

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ]
        conversation = {
            "messages": messages,
            "question_id": row["question_id"],
            "answer": answer,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        """
        # Extract ground truth answer
        ground_truth = conversation["answer"]

        # Extract predicted answer from boxed
        predicted = extract_boxed_answer(assistant_response)

        # Exact string match
        is_correct = int(predicted == ground_truth)
        return is_correct
