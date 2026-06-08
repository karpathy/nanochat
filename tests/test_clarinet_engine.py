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
# l1_adaptive_scale: content-adaptive guidance schedule


def test_l1_scale_constant_when_lo_equals_hi():
    # scale_lo == scale_hi short-circuits to a constant base_scale*scale_lo,
    # regardless of the logits (no softmax, exact vanilla-CFG behavior).
    lc = torch.tensor([[1.0, 5.0, -3.0, 0.0]])
    lu = torch.tensor([[-2.0, 0.0, 4.0, 1.0]])
    s = ClarinetEngine.l1_adaptive_scale(lc, lu, base_scale=1.5, scale_lo=1.0, scale_hi=1.0)
    assert s == 1.5


def test_l1_scale_zero_divergence_hits_floor():
    # Identical distributions => TV distance 0 => s == base_scale * scale_lo.
    lc = torch.tensor([[1.0, 2.0, 3.0]])
    lu = torch.tensor([[1.0, 2.0, 3.0]])
    s = ClarinetEngine.l1_adaptive_scale(lc, lu, base_scale=2.0, scale_lo=0.5, scale_hi=2.0)
    assert torch.allclose(s, torch.tensor([[1.0]]))  # 2.0 * 0.5


def test_l1_scale_max_divergence_hits_ceiling():
    # Disjoint support => TV distance ~1 => s ~ base_scale * scale_hi.
    big = 50.0
    lc = torch.tensor([[big, -big]])
    lu = torch.tensor([[-big, big]])
    s = ClarinetEngine.l1_adaptive_scale(lc, lu, base_scale=1.0, scale_lo=0.5, scale_hi=2.0)
    assert torch.allclose(s, torch.tensor([[2.0]]), atol=1e-4)


def test_l1_scale_monotone_and_bounded():
    # As divergence grows, s grows monotonically and stays within
    # [base*lo, base*hi]. Build three cond logits at increasing distance from a
    # fixed uncond.
    lu = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    near = torch.tensor([[0.1, 0.0, 0.0, 0.0]])
    mid = torch.tensor([[2.0, 0.0, 0.0, 0.0]])
    far = torch.tensor([[20.0, 0.0, 0.0, 0.0]])
    lo, hi, base = 0.5, 2.0, 1.0
    f = lambda lc: ClarinetEngine.l1_adaptive_scale(lc, lu, base, lo, hi).item()
    s_near, s_mid, s_far = f(near), f(mid), f(far)
    assert s_near < s_mid < s_far
    assert base * lo <= s_near and s_far <= base * hi


def test_l1_scale_is_per_row():
    # Row 0 identical (floor), row 1 disjoint (ceiling): scale is per-row.
    big = 50.0
    lc = torch.tensor([[1.0, 2.0], [big, -big]])
    lu = torch.tensor([[1.0, 2.0], [-big, big]])
    s = ClarinetEngine.l1_adaptive_scale(lc, lu, base_scale=1.0, scale_lo=0.5, scale_hi=2.0)
    assert s.shape == (2, 1)
    assert torch.allclose(s[0], torch.tensor([0.5]), atol=1e-4)
    assert torch.allclose(s[1], torch.tensor([2.0]), atol=1e-4)


def test_generate_with_adaptive_scale_smoke():
    # End-to-end: enabling the adaptive schedule doesn't crash and still yields
    # max_tokens tokens. (Mock model returns uniform logits so divergence is 0,
    # i.e. the schedule sits at its floor, but the code path is exercised.)
    model = CountingMockModel()
    eng = ClarinetEngine(model, ClarinetByteTokenizer())
    prompt = [261, 72, 105]
    out = list(eng.generate(prompt, num_samples=2, max_tokens=3, temperature=0.0,
                            iv_weight=2.0, wald_scale=1.0, scale_lo=0.5, scale_hi=2.0))
    assert len(out) == 3
    for col in out:
        assert len(col) == 2


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


def test_two_caches_used_during_prefill():
    model = CountingMockModel()
    eng = ClarinetEngine(model, ClarinetByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]  # <bos> Hello

    # list() drives the generator to StopIteration. After the single yield at
    # max_tokens=1, the engine still runs one post-yield decode forward (to
    # prepare logits for what would be the next step) before the loop breaks.
    list(eng.generate(prompt, num_samples=1, max_tokens=1, temperature=0.0,
                      iv_weight=1.0, wald_scale=1.0))

    # 2 prefill forwards (one per condition) + 2 post-yield decode forwards = 4.
    assert len(model.forward_calls) == 4, (
        f"expected 4 forward calls (2 prefill + 2 decode), got {len(model.forward_calls)}"
    )
    cache_ids = [cid for cid, _ in model.forward_calls]
    assert len(set(cache_ids)) >= 2, "engine should use more than one KV cache"


def test_decode_caches_advance_in_lockstep():
    model = CountingMockModel()
    eng = ClarinetEngine(model, ClarinetByteTokenizer())
    prompt = [261, 72, 105]  # <bos> Hi

    list(eng.generate(prompt, num_samples=1, max_tokens=4, temperature=0.0,
                      iv_weight=0.5, wald_scale=1.0))

    # 4 yields → 4 post-yield decode steps → 2*4 single-token forwards.
    decode_calls = [(cid, T) for cid, T in model.forward_calls if T == 1]
    assert len(decode_calls) == 2 * 4, (
        f"expected 8 single-token decode forward calls (4 steps * 2 conditions), "
        f"got {len(decode_calls)}"
    )

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


# ---------------------------------------------------------------------------
# categorical_logits_at: batch forward with IV combine for categorical eval


class PositionAwareMockModel:
    """
    Returns logits that depend on position AND the marker token at index 1,
    so we can verify both position extraction and dual-pass combination.

    - If marker at index 1 is <|src_reasoning|> (262): logit at pos t = t + 1.0
    - If marker at index 1 is <|src_unknown|> (264): logit at pos t = 0.0
    - Otherwise (base Engine path): logit at pos t = t + 1.0
    """

    def __init__(self, vocab_size=8):
        self.vocab_size = vocab_size
        self.config = MockConfig()
        self._device = torch.device("cpu")
        self.forward_count = 0

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None):
        B, T = ids.shape
        self.forward_count += 1
        logits = torch.zeros(B, T, self.vocab_size)
        for b in range(B):
            if T > 1 and ids[b, 1].item() == 262:  # reasoning marker
                for t in range(T):
                    logits[b, t, :] = float(t + 1)
            elif T > 1 and ids[b, 1].item() == 264:  # unknown marker
                pass  # zeros
            else:
                # Base engine path (no marker): position-based logits
                for t in range(T):
                    logits[b, t, :] = float(t + 1)
        return logits


def test_base_engine_categorical_logits_at():
    """Engine.categorical_logits_at extracts logits at the correct positions."""
    from nanochat.engine import Engine
    model = PositionAwareMockModel()
    eng = Engine(model, ClarinetByteTokenizer())
    # Two sequences of different lengths
    seq1 = [261, 10, 20, 30]       # answer at position 3
    seq2 = [261, 10, 20, 30, 40]   # answer at position 4
    logits = eng.categorical_logits_at([seq1, seq2], [3, 4])
    assert logits.shape == (2, model.vocab_size)
    # Position 3 -> value 4.0 (t+1), Position 4 -> value 5.0
    assert torch.allclose(logits[0], torch.full((model.vocab_size,), 4.0))
    assert torch.allclose(logits[1], torch.full((model.vocab_size,), 5.0))


def test_categorical_logits_at_dual_pass_combine():
    """ClarinetEngine.categorical_logits_at applies dual-pass IV combine."""
    model = PositionAwareMockModel()
    eng = ClarinetEngine(model, ClarinetByteTokenizer())
    # Input: [BOS, t1, t2] — BOS present, marker inserts at pos 1 -> offset +1
    # After marker: [BOS, marker, t1, t2] — answer originally at pos 2, now at 3
    seq = [261, 10, 20]  # answer at position 2 (last position)
    logits = eng.categorical_logits_at([seq], [2], iv_weight=2.0, wald_scale=1.0)
    assert logits.shape == (1, model.vocab_size)
    # Adjusted position 3: cond = 4.0 (3+1), uncond = 0.0
    # combine: 0.0 + 2.0 * 1.0 * (4.0 - 0.0) = 8.0
    assert torch.allclose(logits, torch.full((1, model.vocab_size), 8.0))


def test_categorical_logits_at_w_zero_recovers_uncond():
    """iv_weight=0 returns the unconditional logits (zeros from unknown marker)."""
    model = PositionAwareMockModel()
    eng = ClarinetEngine(model, ClarinetByteTokenizer())
    seq = [261, 10, 20]
    logits = eng.categorical_logits_at([seq], [2], iv_weight=0.0, wald_scale=1.0)
    assert torch.allclose(logits, torch.zeros(1, model.vocab_size))


def test_categorical_logits_at_w_one_recovers_cond():
    """iv_weight=1, wald_scale=1 returns the conditional logits exactly."""
    model = PositionAwareMockModel()
    eng = ClarinetEngine(model, ClarinetByteTokenizer())
    seq = [261, 10, 20]
    logits = eng.categorical_logits_at([seq], [2], iv_weight=1.0, wald_scale=1.0)
    # Adjusted position 3: cond = 4.0
    assert torch.allclose(logits, torch.full((1, model.vocab_size), 4.0))


def test_categorical_logits_at_issues_two_forwards():
    """Verifies exactly two forward calls (cond + uncond) per batch."""
    model = PositionAwareMockModel()
    eng = ClarinetEngine(model, ClarinetByteTokenizer())
    seq = [261, 10, 20]
    eng.categorical_logits_at([seq], [2], iv_weight=1.5, wald_scale=1.0)
    assert model.forward_count == 2


def test_categorical_logits_at_multi_element_batch():
    """Dual-pass works correctly with multiple sequences of different lengths."""
    model = PositionAwareMockModel()
    eng = ClarinetEngine(model, ClarinetByteTokenizer())
    seq1 = [261, 10, 20]           # answer at pos 2, adjusted to 3
    seq2 = [261, 10, 20, 30, 40]   # answer at pos 4, adjusted to 5
    logits = eng.categorical_logits_at(
        [seq1, seq2], [2, 4], iv_weight=1.5, wald_scale=1.0
    )
    assert logits.shape == (2, model.vocab_size)
    # seq1: cond@3 = 4.0, uncond@3 = 0.0 -> 0 + 1.5*(4-0) = 6.0
    # seq2: cond@5 = 6.0, uncond@5 = 0.0 -> 0 + 1.5*(6-0) = 9.0
    assert torch.allclose(logits[0], torch.full((model.vocab_size,), 6.0))
    assert torch.allclose(logits[1], torch.full((model.vocab_size,), 9.0))
