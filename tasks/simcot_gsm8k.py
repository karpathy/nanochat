"""
SIM-CoT enhanced GSM8K dataset with step-level annotations.

This extends the base GSM8K task to provide step boundaries for
Supervised Implicit Chain-of-Thought (SIM-CoT) training.

Each step corresponds to a tool call in the GSM8K reasoning chain.
"""

import re
from datasets import load_dataset
from tasks.common import Task


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

def extract_answer(completion):
    """
    Extract the numerical answer after #### marker.
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


class SIMCoTGSM8K(Task):
    """
    GSM8K with step-level annotations for SIM-CoT training.

    Each calculator tool call <<expr=result>> is treated as one reasoning step.
    We track the boundaries of these steps to enable step-level supervision.
    """

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K subset must be main|socratic"
        assert split in ["train", "test"], "GSM8K split must be train|test"
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """
        Get a single problem from the dataset with step annotations.

        Returns a conversation dict with an additional 'step_boundaries' key
        that marks where each reasoning step begins and ends in the token sequence.
        """
        row = self.ds[index]
        question = row['question']
        answer = row['answer']

        # Parse the answer to extract reasoning steps
        assistant_message_parts = []
        step_markers = []  # Track where each step begins (in the parts list)

        parts = re.split(r'(<<[^>]+>>)', answer)
        for part_idx, part in enumerate(parts):
            if part.startswith('<<') and part.endswith('>>'):
                # This is a calculator tool call - marks a reasoning step
                inner = part[2:-2]  # Remove << >>
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""

                # Mark the start of a new reasoning step
                step_markers.append(len(assistant_message_parts))

                # Add the tool call and result
                assistant_message_parts.append({"type": "python", "text": expr})
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                # Regular text between tool calls
                if part.strip():  # Only add non-empty text
                    assistant_message_parts.append({"type": "text", "text": part})

        # Build the conversation
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_message_parts},
        ]

        conversation = {
            "messages": messages,
            "step_boundaries": step_markers,  # Where each reasoning step starts
            "num_steps": len(step_markers),    # Total number of reasoning steps
        }

        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Evaluate if the final answer is correct.
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"

        # Extract ground truth answer
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant"
        assert isinstance(assistant_message['content'], list)
        last_text_part = assistant_message['content'][-1]['text']

        # Compare answers
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """
        Reward function for RL training.
        """
        is_correct = self.evaluate(conversation, assistant_response)
        return float(is_correct)

    def compute_step_rewards(self, conversation, predicted_steps):
        """
        Compute per-step rewards for intermediate reasoning.

        Args:
            conversation: The ground truth conversation with step_boundaries
            predicted_steps: List of predicted intermediate results

        Returns:
            List of rewards (0.0 or 1.0) for each step
        """
        # Extract ground truth steps from conversation
        assistant_message = conversation['messages'][-1]
        gt_parts = assistant_message['content']

        # Find all tool outputs in ground truth
        gt_results = []
        for part in gt_parts:
            if part['type'] == 'python_output':
                gt_results.append(part['text'].strip())

        # Compare predicted vs ground truth
        step_rewards = []
        for pred, gt in zip(predicted_steps, gt_results):
            reward = 1.0 if str(pred).strip() == gt else 0.0
            step_rewards.append(reward)

        return step_rewards
