"""Type definitions for conversations and messages."""

from typing import List, Literal, TypedDict


class Message(TypedDict):
    """A single message in a conversation."""

    role: Literal["user", "assistant", "system"]
    content: str


Conversation = List[Message]
