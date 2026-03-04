import runpy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import nanochat.engine as engine


def test_timeout_eval_and_calculator_paths(monkeypatch):
    original_timeout = engine.timeout
    with engine.timeout(1, "1+1"):
        assert 1 + 1 == 2

    assert engine.eval_with_timeout("1+2", max_time=1) == 3
    assert engine.eval_with_timeout("bad +", max_time=1) is None

    @contextmanager
    def boom(_duration, _formula):
        raise Exception("t")
        yield  # pragma: no cover

    monkeypatch.setattr(engine, "timeout", boom)
    assert engine.eval_with_timeout("1+1", max_time=1) is None
    monkeypatch.setattr(engine, "timeout", original_timeout)

    assert engine.use_calculator("1,000 + 2") == 1002
    assert engine.use_calculator("2 ** 8") is None
    assert engine.use_calculator("x$y") is None
    assert engine.use_calculator("__import__('os')") is None
    assert engine.use_calculator("'abc'") is None
    assert engine.use_calculator("'ababa'.count('a')") == 3


def test_timeout_handler_line(monkeypatch):
    captured = {}
    monkeypatch.setattr(engine.signal, "signal", lambda _sig, fn: captured.__setitem__("fn", fn))
    monkeypatch.setattr(engine.signal, "alarm", lambda _n: None)
    with pytest.raises(Exception):
        with engine.timeout(1, "slow_expr"):
            captured["fn"](0, None)


def test_sample_next_token_branches():
    logits = torch.tensor([[0.1, 0.2, 0.9]], dtype=torch.float32)
    rng = torch.Generator(device="cpu").manual_seed(123)

    out0 = engine.sample_next_token(logits, rng, temperature=0.0, top_k=None)
    assert out0.shape == (1, 1)
    assert out0.item() == 2

    out1 = engine.sample_next_token(logits, rng, temperature=1.0, top_k=2)
    assert out1.shape == (1, 1)
    assert out1.item() in {1, 2}

    out2 = engine.sample_next_token(logits, rng, temperature=1.0, top_k=None)
    assert out2.shape == (1, 1)

    with pytest.raises(AssertionError):
        engine.sample_next_token(logits, rng, temperature=-1.0)


def test_kv_cache_prefill_assertions():
    a = engine.KVCache(1, 1, 4, 2, 1, "cpu", torch.float32)
    b = engine.KVCache(1, 1, 2, 2, 1, "cpu", torch.float32)
    b.advance(1)
    a.advance(1)
    with pytest.raises(AssertionError):
        a.prefill(b)  # non-empty destination

    c = engine.KVCache(1, 2, 2, 2, 1, "cpu", torch.float32)
    d = engine.KVCache(1, 1, 2, 2, 1, "cpu", torch.float32)
    with pytest.raises(AssertionError):
        c.prefill(d)  # head mismatch

    e = engine.KVCache(1, 1, 1, 2, 1, "cpu", torch.float32)
    f = engine.KVCache(1, 1, 2, 2, 1, "cpu", torch.float32)
    with pytest.raises(AssertionError):
        e.prefill(f)  # seq len too small


class _TinyTok:
    def __init__(self):
        self.special = {
            "<|python_start|>": 100,
            "<|python_end|>": 101,
            "<|output_start|>": 102,
            "<|output_end|>": 103,
            "<|assistant_end|>": 104,
            "<|bos|>": 0,
        }

    def encode_special(self, s):
        return self.special[s]

    def get_bos_token_id(self):
        return 0

    def encode(self, s, prepend=None):
        out = [9] if s == "2" else [55]
        if prepend is not None:
            return [prepend] + out
        return out

    def decode(self, ids):
        return "1+1" if ids else ""


class _TinyModel:
    def __init__(self):
        self.config = SimpleNamespace(n_kv_head=1, n_head=1, n_embd=4, n_layer=1, sequence_len=32)
        self._device = torch.device("cpu")
        self.vocab = 200

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None):
        b, t = ids.shape
        if kv_cache is not None:
            kv_cache.advance(t)
        return torch.zeros((b, t, self.vocab), dtype=torch.float32)


def test_engine_generate_tool_forcing(monkeypatch):
    tok = _TinyTok()
    model = _TinyModel()
    eng = engine.Engine(model, tok)

    sampled = [
        [100],  # python_start
        [55],   # expression token
        [101],  # python_end -> enqueue forced output tokens
        [104],  # ignored while forced queue drains
        [104],
        [104],
        [104],  # finally consumed as assistant_end
    ]
    idx = {"i": 0}

    def fake_sample(_logits, _rng, _temperature, _top_k):
        row = sampled[min(idx["i"], len(sampled) - 1)]
        idx["i"] += 1
        return torch.tensor([[row[0]]], dtype=torch.long)

    calls = {"expr": None}
    monkeypatch.setattr(engine, "sample_next_token", fake_sample)
    monkeypatch.setattr(engine, "use_calculator", lambda expr: calls.__setitem__("expr", expr) or 2)

    rows = list(eng.generate([1, 2], num_samples=1, max_tokens=10, temperature=0.0))
    flat_tokens = [r[0][0] for r in rows]
    flat_masks = [r[1][0] for r in rows]
    assert 102 in flat_tokens and 103 in flat_tokens  # output_start/output_end forced
    assert 0 in flat_masks  # forced token mask
    assert calls["expr"] == "1+1"


def test_engine_generate_batch_and_input_validation(monkeypatch):
    tok = _TinyTok()
    model = _TinyModel()
    eng = engine.Engine(model, tok)

    # Stop immediately via BOS; also exercises max_tokens=None kv-length hint path.
    monkeypatch.setattr(
        engine,
        "sample_next_token",
        lambda *_a, **_k: torch.tensor([[tok.get_bos_token_id()]], dtype=torch.long),
    )
    rows = list(eng.generate([7], num_samples=1, max_tokens=None, temperature=0.0))
    assert len(rows) == 1

    results, masks = eng.generate_batch([7], num_samples=1, max_tokens=3, temperature=0.0)
    assert results == [[7]]
    assert masks == [[0]]

    with pytest.raises(AssertionError):
        list(eng.generate("bad", num_samples=1, max_tokens=1))  # type: ignore[arg-type]


def test_engine_main_block_runs(monkeypatch):
    # Patch external entrypoints used by the __main__ block.
    monkeypatch.setattr("nanochat.common.autodetect_device_type", lambda: "cpu")
    monkeypatch.setattr("nanochat.common.compute_init", lambda _d: (False, 0, 0, 1, torch.device("cpu")))

    class MainTok(_TinyTok):
        def encode(self, s, prepend=None):
            out = [11, 12]
            return ([prepend] + out) if prepend is not None else out

        def decode(self, ids):
            return "x"

    class MainModel(_TinyModel):
        def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
            del tokens, max_tokens, temperature, top_k, seed
            yield 11
            yield 12

    monkeypatch.setattr("nanochat.checkpoint_manager.load_model", lambda *a, **k: (MainModel(), MainTok(), {}))
    monkeypatch.setattr(engine.torch.cuda, "synchronize", lambda: None)
    runpy.run_module("nanochat.engine", run_name="__main__")
