import json

import torch

from nanochat.engine import Engine
from nanochat.tools import CalculatorTool, MockSearchBackend, ToolRegistry, WebSearchTool, parse_tool_call_payload, serialize_tool_call
from tasks.tool_json import ToolJSON


class MockConfig:
    n_kv_head = 4
    n_head = 4
    n_embd = 64
    n_layer = 2
    sequence_len = 128


class ByteTokenizer:
    def __init__(self):
        self._special_tokens = {
            "<|python_start|>": 256,
            "<|python_end|>": 257,
            "<|output_start|>": 258,
            "<|output_end|>": 259,
            "<|assistant_end|>": 260,
            "<|bos|>": 261,
        }
        self._bos = 261

    def encode_special(self, s):
        return self._special_tokens[s]

    def get_bos_token_id(self):
        return self._bos

    def encode(self, s, prepend=None):
        tokens = list(s.encode("utf-8"))
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, tokens):
        byte_tokens = [t for t in tokens if t < 256]
        return bytes(byte_tokens).decode("utf-8", errors="replace")


class SequencedModel:
    def __init__(self, sequence, vocab_size=262):
        self.sequence = sequence
        self.vocab_size = vocab_size
        self.config = MockConfig()
        self._device = torch.device("cpu")
        self.call_index = 0

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None):
        if kv_cache is not None:
            kv_cache.advance(ids.shape[1])
        target = self.sequence[min(self.call_index, len(self.sequence) - 1)]
        logits = torch.full((ids.shape[0], ids.shape[1], self.vocab_size), -1e9)
        logits[:, -1, target] = 0.0
        self.call_index += 1
        return logits


def test_parse_tool_call_payload_supports_json_and_legacy():
    legacy = parse_tool_call_payload("6*7")
    assert legacy.tool_name == "calculator"
    assert legacy.arguments == {"expression": "6*7"}

    payload = parse_tool_call_payload(serialize_tool_call("calculator", {"expression": "6*7"}))
    assert payload.tool_name == "calculator"
    assert payload.arguments == {"expression": "6*7"}


def test_calculator_tool_supports_scientific_and_business_functions():
    tool = CalculatorTool()
    trig = tool.run({"expression": "round(sin(pi/2), 4)"})
    assert trig.success
    assert trig.output["value"] == 1.0

    emi = tool.run({"expression": "round(emi(500000,8,240), 2)"})
    assert emi.success
    assert emi.output["value"] == 4182.2


def test_web_search_tool_returns_mock_results():
    tool = WebSearchTool(search_backend=MockSearchBackend())
    result = tool.run({"query": "nanochat gpt2 speedrun", "top_k": 1})
    assert result.success
    assert result.output["results"][0]["url"] == "https://github.com/karpathy/nanochat"


def test_engine_executes_json_calculator_tool_call():
    tokenizer = ByteTokenizer()
    payload = serialize_tool_call("calculator", {"expression": "6*7"})
    sequence = [
        tokenizer.encode_special("<|python_start|>"),
        *tokenizer.encode(payload),
        tokenizer.encode_special("<|python_end|>"),
        tokenizer.encode_special("<|assistant_end|>"),
    ]
    model = SequencedModel(sequence)
    registry = ToolRegistry([CalculatorTool()])
    engine = Engine(model, tokenizer, tools=registry)
    prompt = [tokenizer.get_bos_token_id(), 72]
    results, _ = engine.generate_batch(prompt, num_samples=1, max_tokens=128, temperature=0.0)
    completion = tokenizer.decode(results[0][len(prompt):])
    assert "42" in completion


def test_tool_json_reward_and_eval(tmp_path):
    path = tmp_path / "tool_eval.jsonl"
    row = {
        "conversation": {
            "messages": [
                {"role": "user", "content": "What is 6 times 7?"},
                {"role": "assistant", "content": "42"},
            ]
        },
        "checks": {
            "must_call": "calculator",
            "answer_contains": ["42"],
        },
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    task = ToolJSON(filepath=str(path))
    conversation = task[0]
    response = '<|python_start|>{"tool":"calculator","arguments":{"expression":"6*7"}}<|python_end|>42'
    assert task.reward(conversation, response) == 1.0
    assert task.evaluate(conversation, response) == 1
