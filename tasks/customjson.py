"""
CustomJSON task for loading conversations from JSONL files.
Each line in the JSONL file should be a JSON array of messages.
"""

import os
import json
from tasks.common import Task


def _validate_assistant_content(content, message_index):
    """Assistant turns may be a plain string or a list of parts (tools / GSM8K-style)."""
    if isinstance(content, str):
        return
    if not isinstance(content, list):
        raise AssertionError(f"Message {message_index}: assistant content must be str or list, got {type(content)}")
    for j, part in enumerate(content):
        if not isinstance(part, dict):
            raise AssertionError(f"Message {message_index} part {j}: expected dict, got {type(part)}")
        ptype = part.get("type")
        if ptype == "text":
            assert "text" in part, f"Message {message_index} part {j}: text part missing 'text'"
        elif ptype in ("tool_call", "python"):
            assert "text" in part or part.get("tool_name"), (
                f"Message {message_index} part {j}: tool part needs 'text' or 'tool_name'"
            )
        elif ptype in ("tool_result", "python_output"):
            assert "text" in part or part.get("tool_name") is not None, (
                f"Message {message_index} part {j}: result part missing 'text' or 'tool_name'"
            )
        else:
            raise AssertionError(f"Message {message_index} part {j}: unknown type {ptype!r}")


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
                        if message["role"] == "user":
                            assert isinstance(message["content"], str), f"Message {i} user content must be a string"
                        else:
                            _validate_assistant_content(message["content"], i)

                    self.conversations.append(messages)

        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        conversation = {
            "messages": messages,
        }
        return conversation
