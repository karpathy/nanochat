"""
AIME (American Invitational Mathematics Examination) 2024/2025 evaluation.
https://huggingface.co/datasets/Maxwell-Jia/AIME_2024

AIME is a challenging mathematics competition with integer answers.
The dataset contains problems from AIME 2024 and 2025 competitions.
"""

import re
from datasets import load_dataset
from tasks.common import Task


AIME_BOXED_RE = re.compile(r'\\boxed\{(\d+)\}')
AIME_RE = re.compile(r'(\-?\d+)')

def extract_answer(completion):
    """
    Extract the integer answer from the completion.
    AIME answers are always integers (0-999).
    
    Priority order:
    1. Look for \boxed{...} format (standard AIME answer format)
    2. Otherwise, look for the last integer in the response
    """
    # First, try to find \boxed{...} format (standard AIME answer format)
    boxed_match = AIME_BOXED_RE.search(completion)
    if boxed_match:
        try:
            answer_str = boxed_match.group(1)
            # Remove leading zeros and convert to integer
            answer = int(answer_str)
            # AIME answers are always between 0 and 999
            if 0 <= answer <= 999:
                return str(answer)
        except ValueError:
            pass
    
    # Fallback: find all integers in the completion
    matches = AIME_RE.findall(completion)
    if matches:
        # Return the last integer found (usually the final answer)
        # AIME answers are in range 0-999, so we validate
        try:
            answer = int(matches[-1])
            # AIME answers are always between 0 and 999
            if 0 <= answer <= 999:
                return str(answer)
        except ValueError:
            pass
    return None


class AIME(Task):

    def __init__(self, year, split, **kwargs):
        super().__init__(**kwargs)
        assert year in ["2024", "2025"], "AIME year must be 2024 or 2025"
        assert split in ["train", "test"], "AIME split must be train|test"
        
        # Load the dataset - adjust the dataset name if needed
        dataset_name = f"Maxwell-Jia/AIME_{year}"
        
        try:
            # Try to load the requested split
            self.ds = load_dataset(dataset_name, split=split).shuffle(seed=42)
        except ValueError as e:
            # If the requested split doesn't exist, try to use train split
            if "Unknown split" in str(e) and split == "test":
                print(f"Warning: {dataset_name} doesn't have 'test' split, using 'train' split instead")
                self.ds = load_dataset(dataset_name, split="train").shuffle(seed=42)
            else:
                raise e

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        # AIME dataset typically has 'problem' or 'question' field
        # and 'answer' or 'solution' field
        # Adjust field names based on actual dataset structure
        question = row.get('problem', row.get('question', ''))
        answer = row.get('answer', row.get('solution', ''))
        
        # Ensure answer is a string representation of integer
        if isinstance(answer, int):
            answer_str = str(answer)
        else:
            answer_str = str(answer).strip()
        
        # Create and return the Conversation object
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_str}
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        AIME answers are integers, so we extract and compare the integer answers.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        
        # Get the ground truth answer from the conversation
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        ref_answer = assistant_message['content'].strip()
        
        # Extract predicted answer from the model response
        pred_answer = extract_answer(assistant_response)
        
        # Compare answers
        if pred_answer is None:
            return 0
        
        # Normalize both answers (remove leading zeros, etc.)
        try:
            ref_int = int(ref_answer)
            pred_int = int(pred_answer)
            is_correct = int(ref_int == pred_int)
            return is_correct
        except ValueError:
            return 0

    def reward(self, conversation, assistant_response):
        """
        Used during RL. To keep things simple, just re-use the evaluation above.
        """
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float