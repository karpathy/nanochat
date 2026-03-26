import importlib
import sys

import pytest
from fastapi import HTTPException


def load_chat_web(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["chat_web.py"])
    module_name = "scripts.chat_web"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def test_validate_chat_request_allows_initial_system_message(monkeypatch):
    chat_web = load_chat_web(monkeypatch)
    request = chat_web.ChatRequest(messages=[
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Hello"},
    ])

    chat_web.validate_chat_request(request)


def test_validate_chat_request_rejects_system_message_after_first_turn(monkeypatch):
    chat_web = load_chat_web(monkeypatch)
    request = chat_web.ChatRequest(messages=[
        {"role": "user", "content": "Hello"},
        {"role": "system", "content": "You are concise."},
    ])

    with pytest.raises(HTTPException):
        chat_web.validate_chat_request(request)


def test_build_conversation_tokens_delegates_to_tokenizer_helper(monkeypatch):
    chat_web = load_chat_web(monkeypatch)

    class RecordingTokenizer:
        def __init__(self):
            self.calls = []

        def render_for_assistant_reply(self, conversation, max_tokens):
            self.calls.append((conversation, max_tokens))
            return [7, 8, 9]

    tokenizer = RecordingTokenizer()
    messages = [
        chat_web.ChatMessage(role="system", content="rules"),
        chat_web.ChatMessage(role="user", content="hello"),
    ]

    tokens = chat_web.build_conversation_tokens(tokenizer, messages, sequence_len=17)

    assert tokens == [7, 8, 9]
    assert tokenizer.calls == [
        (
            {
                "messages": [
                    {"role": "system", "content": "rules"},
                    {"role": "user", "content": "hello"},
                ]
            },
            17,
        )
    ]
