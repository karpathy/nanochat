"""
This module implements the `CustomJSON` task, which allows for loading conversational
data from a custom JSONL file. This is useful for fine-tuning the model on specific
datasets, such as synthetic data for instilling a persona.

Each line in the JSONL file should be a JSON array of message objects, where each
object has a "role" and a "content" field.
"""

import os
import json
from .common import Task

class CustomJSON(Task):
    """
    Load conversations from a JSONL file.
    Each line should be a JSON array of message objects with 'role' and 'content' fields.
    Example line: [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]
    """

    def __init__(self, filepath, **kwargs):
        """
        Initializes the CustomJSON task.

        Args:
            filepath (str): The path to the JSONL file.
        """
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        # Load all conversations from the JSONL file
        if not os.path.exists(filepath):
            # Helpful error message due to recent change. Will be removed in the future.
            print("-" * 80)
            print(f"Warning: File {filepath} does not exist")
            print("HINT (Oct 21 2025)")
            print("If you recently did a git pull and suddely see this, it might be due to the new addition of identity conversations")
            print("See this discussion for more details: https://github.com/karpathy/nanochat/discussions/139")
            print("Quick fix: simply run the following command to download the file and you're done:")
            print(f"curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl")
            print("-" * 80)

        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # skip empty lines
                        continue
                    messages = json.loads(line)
                    # Validate the conversation structure
                    assert isinstance(messages, list), f"Expected list of messages, got {type(messages)}"
                    assert len(messages) >= 2, f"Conversation must have at least 2 messages, got {len(messages)}"
                    # Validate message structure and alternating roles
                    for i, message in enumerate(messages):
                        assert "role" in message, f"Message {i} missing 'role' field"
                        assert "content" in message, f"Message {i} missing 'content' field"
                        expected_role = "user" if i % 2 == 0 else "assistant"
                        assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
                        assert isinstance(message["content"], str), f"Message {i} content must be a string"

                    self.conversations.append(messages)

        self.length = len(self.conversations)

    def num_examples(self):
        """Returns the total number of conversations loaded from the file."""
        return self.length

    def get_example(self, index):
        """
        Retrieves a single conversation by its index.

        Args:
            index (int): The index of the conversation to retrieve.

        Returns:
            dict: A dictionary representing the conversation.
        """
        messages = self.conversations[index]
        conversation = {
            "messages": messages,
        }
        return conversation

