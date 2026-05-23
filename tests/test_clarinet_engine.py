"""
Hermetic tests for clarinet's dual-pass IV engine.

Uses the existing MockModel/ByteTokenizer scaffolding pattern from
tests/test_engine.py, extended with a counting wrapper that lets us verify the
dual-cache mechanics. The IV combine formula is unit-tested directly via the
static method extracted from generate().

Run: python -m pytest tests/test_clarinet_engine.py -v
"""

from dataclasses import dataclass

import pytest
import torch

from clarinet.engine import ClarinetEngine


@dataclass
class MockConfig:
    n_kv_head: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_layer: int = 2
    sequence_len: int = 128


class CountingMockModel:
    """
    Returns uniform logits and records every forward call's kv_cache identity
    + input length. Lets us assert that the dual-pass engine does in fact
    issue two forward calls per decode step against two distinct KV caches.
    """

    def __init__(self, vocab_size=2048):
        self.vocab_size = vocab_size
        self.config = MockConfig()
        self._device = torch.device("cpu")
        self.forward_calls = []  # list of (id(kv_cache), T)

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None):
        B, T = ids.shape
        if kv_cache is not None:
            kv_cache.advance(T)
            self.forward_calls.append((id(kv_cache), T))
        else:
            self.forward_calls.append((None, T))
        return torch.zeros(B, T, self.vocab_size)


class ClarinetByteTokenizer:
    """Byte tokenizer with the clarinet source-marker specials wired in."""

    def __init__(self):
        self._special_tokens = {
            "<|python_start|>": 256,
            "<|python_end|>": 257,
            "<|output_start|>": 258,
            "<|output_end|>": 259,
            "<|assistant_end|>": 260,
            "<|bos|>": 261,
            "<|src_reasoning|>": 262,
            "<|src_general|>": 263,
            "<|src_unknown|>": 264,
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
        return bytes(t for t in tokens if t < 256).decode("utf-8", errors="replace")


# -----------------------------------------------------------------------------
# combine_logits: pure formula


def test_combine_w_zero_returns_uncond():
    lc = torch.tensor([1.0, 2.0, 3.0, 4.0])
    lu = torch.tensor([0.5, 1.0, 1.5, 2.0])
    out = ClarinetEngine.combine_logits(lc, lu, iv_weight=0.0, wald_scale=1.0)
    assert torch.allclose(out, lu)


def test_combine_w_one_s_one_returns_cond():
    lc = torch.tensor([1.0, 2.0, 3.0, 4.0])
    lu = torch.tensor([0.5, 1.0, 1.5, 2.0])
    out = ClarinetEngine.combine_logits(lc, lu, iv_weight=1.0, wald_scale=1.0)
    assert torch.allclose(out, lc)


def test_combine_overguided_extrapolates_past_cond():
    # w=2, s=1 should push 2x past cond in the cond-uncond direction
    lc = torch.tensor([1.0, 2.0, 3.0])
    lu = torch.tensor([0.5, 1.0, 1.5])
    out = ClarinetEngine.combine_logits(lc, lu, iv_weight=2.0, wald_scale=1.0)
    expected = lu + 2.0 * (lc - lu)
    assert torch.allclose(out, expected)
    # Sanity: each component is on the cond side of uncond and further away
    assert torch.all((out - lu).abs() >= (lc - lu).abs())


def test_combine_wald_scale_multiplies_step():
    lc = torch.tensor([1.0, 2.0])
    lu = torch.tensor([0.0, 0.0])
    # w=1, s=2 => uncond + 1*2*(cond-uncond) = 2*cond when uncond=0
    out = ClarinetEngine.combine_logits(lc, lu, iv_weight=1.0, wald_scale=2.0)
    assert torch.allclose(out, 2.0 * lc)


# -----------------------------------------------------------------------------
# _prefix_with_marker: prompt splicing


def _engine():
    return ClarinetEngine(CountingMockModel(), ClarinetByteTokenizer())


def test_prefix_with_marker_when_bos_already_present():
    eng = _engine()
    tokens = [261, 100, 101, 102]  # [BOS, ...]
    out = eng._prefix_with_marker(tokens, marker_id=262, bos_id=261)
    assert out == [261, 262, 100, 101, 102]


def test_prefix_with_marker_when_bos_missing():
    eng = _engine()
    tokens = [100, 101, 102]
    out = eng._prefix_with_marker(tokens, marker_id=263, bos_id=261)
    assert out == [261, 263, 100, 101, 102]


def test_prefix_with_marker_does_not_mutate_input():
    eng = _engine()
    tokens = [261, 50]
    snapshot = list(tokens)
    eng._prefix_with_marker(tokens, marker_id=264, bos_id=261)
    assert tokens == snapshot


# -----------------------------------------------------------------------------
# Dual-pass mechanics


def _consume(generator, n):
    for _ in range(n):
        try:
            next(generator)
        except StopIteration:
            break


def test_two_caches_used_during_prefill():
    model = CountingMockModel()
    eng = ClarinetEngine(model, ClarinetByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]  # <bos> Hello

    gen = eng.generate(prompt, num_samples=1, max_tokens=1, temperature=0.0,
                       iv_weight=1.0, wald_scale=1.0)
    _consume(gen, 1)

    # Prefill: 2 calls on the small prefill caches (one per condition).
    # Decode step 1: 2 calls on the decode caches (one per condition).
    # That's 4 forward calls total for max_tokens=1.
    assert len(model.forward_calls) == 4, (
        f"expected 4 forward calls (2 prefill + 2 decode), got {len(model.forward_calls)}"
    )

    # The 4 calls hit at most 4 distinct cache identities, and they come in
    # cond/uncond pairs by construction. Specifically: 2 prefill caches (held
    # briefly then deleted) and 2 decode caches.
    cache_ids = [cid for cid, _ in model.forward_calls]
    assert len(set(cache_ids)) >= 2, "engine should use more than one KV cache"


def test_decode_caches_advance_in_lockstep():
    model = CountingMockModel()
    eng = ClarinetEngine(model, ClarinetByteTokenizer())
    prompt = [261, 72, 105]  # <bos> Hi

    gen = eng.generate(prompt, num_samples=1, max_tokens=4, temperature=0.0,
                       iv_weight=0.5, wald_scale=1.0)
    _consume(gen, 4)

    # Per decode step there are exactly 2 forward calls (one cond, one uncond),
    # each consuming 1 token. After prefill (2 calls, each consuming len(prompt) tokens),
    # we should see 2*max_tokens=8 single-token forward calls.
    decode_calls = [(cid, T) for cid, T in model.forward_calls if T == 1]
    assert len(decode_calls) == 2 * 4, (
        f"expected 8 single-token decode forward calls (4 steps * 2 conditions), "
        f"got {len(decode_calls)}"
    )

    # The decode forward calls should split evenly between two cache identities
    from collections import Counter
    counts = Counter(cid for cid, _ in decode_calls)
    assert len(counts) == 2, f"expected exactly 2 distinct decode caches, got {len(counts)}"
    assert all(c == 4 for c in counts.values()), (
        f"each decode cache should see 4 advances, got {dict(counts)}"
    )


def test_generate_yields_max_tokens(monkeypatch):
    model = CountingMockModel()
    eng = ClarinetEngine(model, ClarinetByteTokenizer())
    prompt = [261, 72]

    yielded = []
    for token_col, _mask in eng.generate(prompt, num_samples=2, max_tokens=3,
                                          temperature=0.0, iv_weight=1.5, wald_scale=1.0):
        yielded.append(token_col)

    # max_tokens=3 with non-terminating uniform logits → exactly 3 yields.
    assert len(yielded) == 3
    # Each yield is one token per sample (num_samples=2).
    for col in yielded:
        assert len(col) == 2


def test_iv_weight_zero_matches_unconditional_only_sampling():
    """
    With iv_weight=0, the combined logits equal the unconditional logits.
    Since the mock model returns identical (uniform) logits regardless of
    cache, cond/uncond combine is a no-op and the run should complete cleanly.
    This is mostly a smoke test that w=0 doesn't crash.
    """
    model = CountingMockModel()
    eng = ClarinetEngine(model, ClarinetByteTokenizer())
    prompt = [261, 72, 105]
    out = list(eng.generate(prompt, num_samples=1, max_tokens=2, temperature=0.0,
                            iv_weight=0.0, wald_scale=1.0))
    assert len(out) == 2
