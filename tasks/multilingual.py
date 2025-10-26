"""
Generic task for loading multilingual conversational data from any HuggingFace dataset.

This task allows users to add any conversational dataset to their training pipeline,
enabling multilingual training by mixing different language datasets.

Example usage:
    from tasks.multilingual import MultilingualTask
    train_ds = TaskMixture([
        MultilingualTask("HuggingFaceTB/smol-talk-lt", split="train"),  # Lithuanian
        MultilingualTask("tatsu-lab/alpaca", split="train"),  # English
    ])

Expected dataset format:
    Each row must have a "messages" field containing a list of messages:
    [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
"""

from datasets import load_dataset
from tasks.common import Task

class MultilingualTask(Task):
    """
    Generic task for loading any HuggingFace dataset with conversational format.
    
    Args:
        hf_dataset: HuggingFace dataset identifier (e.g., "user/dataset-name")
        split: Dataset split to use (e.g., "train", "test", "validation")
        start: Starting index for dataset slice (inherited from Task)
        stop: Ending index for dataset slice (inherited from Task)
        step: Step size for dataset slice (inherited from Task)
    """

    def __init__(self, hf_dataset, split="train", **kwargs):
        super().__init__(**kwargs)
        self.split = split
        self.hf_dataset = hf_dataset
        
        try:
            self.ds = load_dataset(hf_dataset, split=split).shuffle(seed=42)
            self.length = len(self.ds)
        except Exception as e:
            raise ValueError(f"Failed to load dataset '{hf_dataset}' with split '{split}': {e}")
        
        if self.length == 0:
            raise ValueError(f"Dataset '{hf_dataset}' split '{split}' is empty")

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.length

    def get_example(self, index):
        """
        Get a single conversation example from the dataset.
        
        Args:
            index: Index of the example to retrieve
            
        Returns:
            Dictionary with "messages" field containing the conversation
        """
        if index >= self.length:
            raise IndexError(f"Index {index} out of range for dataset with {self.length} examples")
        
        row = self.ds[index]
        
        # Validate that the dataset has the expected structure
        if "messages" not in row:
            raise ValueError(
                f"Dataset '{self.hf_dataset}' does not have 'messages' field. "
                f"Available fields: {list(row.keys())}"
            )
        
        messages = row["messages"]
        
        # Basic validation of messages structure
        if not isinstance(messages, list):
            raise ValueError(
                f"Dataset '{self.hf_dataset}' 'messages' field must be a list, got {type(messages)}"
            )
        
        if len(messages) < 1:
            raise ValueError(
                f"Dataset '{self.hf_dataset}' 'messages' list is empty"
            )
        
        # Validate message structure (optional system message followed by alternating user/assistant)
        first_message = messages[0]
        if not isinstance(first_message, dict):
            raise ValueError(
                f"Dataset '{self.hf_dataset}' first message must be a dictionary, got {type(first_message)}"
            )
        if first_message.get("role") == "system":
            rest_messages = messages[1:]
        else:
            rest_messages = messages
        
        if len(rest_messages) < 2:
            raise ValueError(
                f"Dataset '{self.hf_dataset}' must have at least 2 non-system messages, got {len(rest_messages)}"
            )
        
        for i, message in enumerate(rest_messages):
            if "role" not in message or "content" not in message:
                raise ValueError(
                    f"Dataset '{self.hf_dataset}' message {i} missing 'role' or 'content' field"
                )
            if not isinstance(message["content"], str):
                raise ValueError(
                    f"Dataset '{self.hf_dataset}' message {i} 'content' must be a string, got {type(message['content'])}"
                )
        
        conversation = {
            "messages": messages,
        }
        return conversation

