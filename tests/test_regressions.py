import importlib
import sys

import pytest
import torch
from fastapi import HTTPException

import nanochat.engine as engine_module
from nanochat.engine import Engine


class DummyModel:
    def get_device(self):
        return torch.device("cpu")


def test_engine_kv_cache_uses_compute_dtype(monkeypatch):
    monkeypatch.setattr(engine_module, "COMPUTE_DTYPE", torch.float16)
    engine = Engine(DummyModel(), tokenizer=None)
    assert engine._get_kv_cache_dtype() == torch.float16


def test_chat_web_rejects_system_role_with_consistent_error(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["chat_web_test", "--definitely-not-a-valid-flag"])
    sys.modules.pop("scripts.chat_web", None)
    chat_web = importlib.import_module("scripts.chat_web")

    request = chat_web.ChatRequest(
        messages=[chat_web.ChatMessage(role="system", content="You are helpful.")]
    )

    assert chat_web._runtime_initialized is False

    with pytest.raises(HTTPException) as exc_info:
        chat_web.validate_chat_request(request)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Message 0 has invalid role. Must be 'user' or 'assistant'"
