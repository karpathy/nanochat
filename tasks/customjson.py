"""
CustomJSON task for loading conversations from JSONL files.
Each line in the JSONL file should be a JSON array of messages.
"""

import os
import json
from tasks.common import Task

class CustomJSON(Task):
    """
    Load conversations from a JSONL file.
    Each line should be a JSON array of message objects with 'role' and 'content' fields.
    Example line: [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]
    """

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        # Load all conversations from the JSONL file
        if not os.path.exists(filepath):
            # Helpful error message due to recent change. Will be removed in the future.
            print("-" * 80)
            print(f"Warning: File {filepath} does not exist")
            print("HINT (Oct 21 2025)")
            print("If you recently did a git pull and suddenly see this, it might be due to the new addition of identity conversations")
            print("See this discussion for more details: https://github.com/karpathy/nanochat/discussions/139")
            print("Quick fix: simply run the following command to download the file and you're done:")
            print(f"curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl")
            print("-" * 80)

        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, raw_line in enumerate(f, start=1):
                    line = raw_line.strip()
                    if not line:  # skip empty lines
                        continue
                    try:
                        messages = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{filepath}:{line_num}: invalid JSON ({exc.msg})") from exc
                    self._validate_conversation(messages, filepath, line_num)
                    self.conversations.append(messages)

        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    @staticmethod
    def _validate_conversation(messages, filepath, line_num):
        if not isinstance(messages, list):
            raise ValueError(
                f"{filepath}:{line_num}: expected a JSON array of messages, got {type(messages).__name__}"
            )
        if len(messages) < 2:
            raise ValueError(
                f"{filepath}:{line_num}: conversation must have at least 2 messages, got {len(messages)}"
            )
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(
                    f"{filepath}:{line_num}: message {i} must be an object, got {type(message).__name__}"
                )
            if "role" not in message:
                raise ValueError(f"{filepath}:{line_num}: message {i} missing 'role' field")
            if "content" not in message:
                raise ValueError(f"{filepath}:{line_num}: message {i} missing 'content' field")
            expected_role = "user" if i % 2 == 0 else "assistant"
            if message["role"] != expected_role:
                raise ValueError(
                    f"{filepath}:{line_num}: message {i} has role {message['role']} but should be {expected_role}"
                )
            if not isinstance(message["content"], str):
                raise ValueError(f"{filepath}:{line_num}: message {i} content must be a string")

    def get_example(self, index):
        messages = self.conversations[index]
        conversation = {
            "messages": messages,
        }
        return conversation
