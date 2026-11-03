"""Exercise CLI output with real byte-level BPE tokens and scripted generation.

python -m pytest tests/test_chat_cli.py -v
"""

import runpy
import sys

import pytest
import tiktoken

from nanochat import checkpoint_manager, common, engine
from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS, SPLIT_PATTERN


@pytest.fixture
def tokenizer():
    # No merges: every UTF-8 byte is a token, so boundaries are deterministic.
    enc = tiktoken.Encoding(
        name="cli-test",
        pat_str=SPLIT_PATTERN,
        mergeable_ranks={bytes([i]): i for i in range(256)},
        special_tokens={name: 256 + i for i, name in enumerate(SPECIAL_TOKENS)},
    )
    return RustBPETokenizer(enc, "<|bos|>")


def run_cli(monkeypatch, tokenizer, responses, prompts):
    histories = []
    response_iter = iter(responses)

    class ScriptedEngine:
        def __init__(self, model, tokenizer):
            pass

        def generate(self, tokens, **kwargs):
            histories.append(tokens.copy())
            for token in next(response_iter):
                yield [token], [1]

    monkeypatch.setattr(common, "compute_init", lambda _: (False, 0, 0, 1, None))
    monkeypatch.setattr(checkpoint_manager, "load_model", lambda *a, **kw: (None, tokenizer, {}))
    monkeypatch.setattr(engine, "Engine", ScriptedEngine)
    monkeypatch.setattr(sys, "argv", ["chat_cli", "--device-type", "cpu"])
    prompt_iter = iter([*prompts, "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(prompt_iter))
    runpy.run_module("scripts.chat_cli", run_name="__main__")
    return histories


@pytest.mark.parametrize("text", ["Hello world!", "café", "你好", "🙂", "literal �"])
def test_cli_preserves_utf8_across_tokens(monkeypatch, capsys, tokenizer, text):
    response = tokenizer.encode(text)
    run_cli(monkeypatch, tokenizer, [response], ["hello"])
    assert f"Assistant: {text}\n" in capsys.readouterr().out


@pytest.mark.parametrize("response", [[0xFF], [0xE2, 0x82]])
def test_cli_flushes_invalid_or_incomplete_utf8(monkeypatch, capsys, tokenizer, response):
    run_cli(monkeypatch, tokenizer, [response], ["hello"])
    assert f"Assistant: {tokenizer.decode(response)}\n" in capsys.readouterr().out


def test_cli_decoder_is_reset_and_history_preserved(monkeypatch, capsys, tokenizer):
    responses = [[0xE2], tokenizer.encode("café")]
    histories = run_cli(monkeypatch, tokenizer, responses, ["one", "two"])
    output = capsys.readouterr().out
    assert "Assistant: �\n" in output
    assert "Assistant: café\n" in output
    end = tokenizer.encode_special("<|assistant_end|>")
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assert histories[1] == [
        *histories[0], *responses[0], end,
        user_start, *tokenizer.encode("two"), user_end, assistant_start,
    ]
