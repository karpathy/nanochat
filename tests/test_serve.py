
import asyncio
import json
import httpx
import pytest
import multiprocessing
import time
import os
import signal
import sys
from unittest.mock import MagicMock

# Define a mock for the engine or just use subprocess to run the real server if possible?
# Running real server might be heavy and require GPUs.
# We can mock the logic inside scripts/serve.py if we import it.

# However, importing `scripts.serve` executes the module level code.
# Ideally we should verify the API contract.

# Let's try to run the server in a separate process, but mock the engine part?
# Modifying `scripts/serve.py` to be testable with mocks is better.
# But `scripts/serve.py` imports `nanochat.engine` and `load_model`.

# Plan:
# 1. Use `TestClient` from `fastapi.testclient` to test the app directly.
# 2. Mock `app.state.worker_pool` to avoid loading real models.

from fastapi.testclient import TestClient
from scripts.serve import app, WorkerPool, Worker

# Mock dependencies
@pytest.fixture
def mock_worker_pool(monkeypatch):
    async def mock_init(self, *args, **kwargs):
        # Create a dummy worker
        mock_engine = MagicMock()
        # Mock generate to yield tokens
        def mock_generate(*args, **kwargs):
            # Yield (token_column, token_masks)
            # Token 50256 is usually BOS/EOS in gpt2, let's just use dummy ints
            # 123 is 'hello', 456 is ' world'
            yield ([123], [1])
            yield ([456], [1])
            # End token
            yield ([50256], [1])

        mock_engine.generate = mock_generate

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.encode_special.return_value = 999
        mock_tokenizer.get_bos_token_id.return_value = 50256
        mock_tokenizer.decode.side_effect = lambda tokens: "Hello" if tokens == [123] else "Hello World"

        mock_autocast = MagicMock()
        mock_autocast.__enter__ = MagicMock()
        mock_autocast.__exit__ = MagicMock()

        worker = Worker(0, "cpu", mock_engine, mock_tokenizer, mock_autocast)
        self.workers = [worker]
        await self.available_workers.put(worker)

    monkeypatch.setattr(WorkerPool, "initialize", mock_init)

client = TestClient(app)

def test_health(mock_worker_pool):
    # Trigger startup event manually or via TestClient with context manager
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

def test_chat_completions_non_streaming(mock_worker_pool):
    with TestClient(app) as client:
        payload = {
            "model": "nanochat",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) == 1
        assert "Hello World" in data["choices"][0]["message"]["content"]

def test_chat_completions_streaming(mock_worker_pool):
    with TestClient(app) as client:
        payload = {
            "model": "nanochat",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True
        }
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        lines = []
        for line in response.iter_lines():
            if line:
                lines.append(line)

        # Verify SSE format
        assert any("data: " in line for line in lines)
        # Check for DONE signal (it might be in the last non-empty line)
        assert any("data: [DONE]" in line for line in lines)

def test_ui_exists(mock_worker_pool):
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "nanochat" in response.text
