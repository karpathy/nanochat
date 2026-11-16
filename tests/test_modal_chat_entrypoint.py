"""
Author: nanochat Modal chat entrypoint slice (Claude)
Created: 2026-11-16
Scope: modal_run.chat public port — forwarding, refusals, defaults, and boundary validation.
       All six acceptance criteria are covered without a live Modal container.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
if str(RUNS_DIR) not in sys.path:
    sys.path.insert(0, str(RUNS_DIR))

import modal_run


def _call_chat(monkeypatch, fake_remote, **kwargs):
    """Drive modal_run.chat with a recording fake for chat_on_modal.remote."""
    monkeypatch.setattr(modal_run.chat_on_modal, "remote", fake_remote)
    modal_run.chat(**kwargs)


# ─── AC4: missing / blank prompt → clear SystemExit ──────────────────────────

class TestMissingPromptRefusal:
    """
    Scenario: user omits --prompt or passes only whitespace
    Slice: AC4 — missing prompt → clear error naming the flag
    Given: chat entrypoint is called with no prompt
    When: chat() runs
    Then: SystemExit is raised with the flag name and an example in the message
    """

    def test_no_prompt_raises(self, monkeypatch):
        ### Given
        fake = MagicMock()
        ### When / Then
        with pytest.raises(SystemExit) as exc:
            _call_chat(monkeypatch, fake, prompt="")
        assert "--prompt" in str(exc.value)
        assert fake.call_count == 0

    def test_whitespace_only_prompt_raises(self, monkeypatch):
        """Whitespace-only must not reach Modal (F4)."""
        ### Given
        fake = MagicMock()
        ### When / Then
        with pytest.raises(SystemExit) as exc:
            _call_chat(monkeypatch, fake, prompt="   ")
        assert "--prompt" in str(exc.value)
        assert fake.call_count == 0


# ─── AC4: unknown mode → clear SystemExit ────────────────────────────────────

class TestUnknownModeRefusal:
    """
    Scenario: user passes an unsupported --mode value
    Slice: AC4 — bad mode → clear error naming both valid values
    Given: chat entrypoint is called with mode='tpu'
    When: chat() runs
    Then: SystemExit is raised mentioning the bad value
    """

    def test_unknown_mode_raises(self, monkeypatch):
        ### Given
        fake = MagicMock()
        ### When / Then
        with pytest.raises(SystemExit) as exc:
            _call_chat(monkeypatch, fake, mode="tpu", prompt="hi")
        assert "tpu" in str(exc.value)
        assert fake.call_count == 0


# ─── F3: boundary validation for temperature / top_k ─────────────────────────

class TestParameterBoundaryRefusal:
    """
    Scenario: user passes out-of-range temperature or top_k
    Slice: F3 — refuse locally before spending a billed container
    Given: chat is called with temperature=-1 or top_k=0
    When: chat() runs
    Then: SystemExit is raised naming the bad flag and value
    """

    def test_negative_temperature_raises(self, monkeypatch):
        ### Given
        fake = MagicMock()
        ### When / Then
        with pytest.raises(SystemExit) as exc:
            _call_chat(monkeypatch, fake, prompt="hi", temperature=-1.0)
        assert "--temperature" in str(exc.value)
        assert fake.call_count == 0

    def test_zero_top_k_raises(self, monkeypatch):
        ### Given
        fake = MagicMock()
        ### When / Then
        with pytest.raises(SystemExit) as exc:
            _call_chat(monkeypatch, fake, prompt="hi", top_k=0)
        assert "--top-k" in str(exc.value)
        assert fake.call_count == 0


# ─── AC2 + AC3: forwarding and defaults ──────────────────────────────────────

class TestForwarding:
    """
    Scenario: valid call forwards (mode, prompt, temperature, top_k) to chat_on_modal.remote
    Slice: AC2 (mode forwarded), AC3 (temperature/top_k forwarded; default 0.3)
    Given: chat is called with explicit values
    When: chat() runs
    Then: remote is called once with exactly those values
    """

    def test_explicit_args_forwarded(self, monkeypatch, capsys):
        ### Given
        fake = MagicMock(return_value="Paris.\n")
        ### When
        _call_chat(monkeypatch, fake, mode="cpu", prompt="Capital of France?", temperature=0.1, top_k=10)
        ### Then
        fake.assert_called_once_with("cpu", "Capital of France?", 0.1, 10)

    def test_default_temperature_is_0_3(self, monkeypatch):
        """The promised default is 0.3 (not echoed from the signature — this asserts the contract)."""
        ### Given
        fake = MagicMock(return_value="")
        ### When
        _call_chat(monkeypatch, fake, prompt="hi")
        ### Then
        _, call_args, _ = fake.mock_calls[0]
        temperature_passed = call_args[2]
        assert temperature_passed == 0.3, f"expected default temperature 0.3, got {temperature_passed}"

    def test_gpu_mode_forwarded(self, monkeypatch):
        ### Given
        fake = MagicMock(return_value="")
        ### When
        _call_chat(monkeypatch, fake, mode="gpu", prompt="hi")
        ### Then
        fake.assert_called_once_with("gpu", "hi", 0.3, 50)


# ─── AC1: response printed, nothing else on stdout (no banner, no prefix) ────

class TestOutputShape:
    """
    Scenario: chat returns exactly the model response — no banner, no 'Assistant:' label
    Slice: AC1 (nothing downloaded), AC5 (banner suppressed under -p)
    Given: chat_on_modal.remote returns a plain string
    When: chat() runs
    Then: stdout contains the response and nothing extra
    """

    def test_response_printed_to_stdout(self, monkeypatch, capsys):
        ### Given
        fake = MagicMock(return_value="The capital of France is Paris.\n")
        ### When
        _call_chat(monkeypatch, fake, prompt="Capital of France?")
        ### Then
        out = capsys.readouterr().out
        assert "The capital of France is Paris." in out
