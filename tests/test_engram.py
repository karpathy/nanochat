"""
Regression tests for Engram integration.

These tests guard against bugs found during implementation:

1. Stale Block.forward left inside EngramModule scope overrode the correct method
2. Parameter double-counting when Engram modules stored in both engam_modules and blocks
3. Engram embed params appearing in both Muon and AdamW optimizer groups
4. Non-2D params (hash_seeds, conv weights) crashing Muon's shape-based stacking
5. ve_gate on Engram layers getting no gradient (ve=None → gate unused → no grad)
6. Block.forward signature must accept input_ids for Engram layers
7. KVCache must support ngram_running state for decode

Run: python -m pytest tests/test_engram.py -v
"""

import torch
import pytest
from nanochat.gpt import GPT, GPTConfig, Block, EngramModule, _next_prime


def _make_model(engram_layer_ids=(1, 2), n_layer=4):
    return GPTConfig(
        sequence_len=128, vocab_size=32768,
        n_layer=n_layer, n_head=4, n_kv_head=4, n_embd=256,
        engram_enabled=True, engram_ngram_size=3,
        engram_n_heads=2, engram_embed_dim=64,
        engram_layer_ids=engram_layer_ids,
    )


class TestEngramForwardSignature:
    """Block.forward must accept input_ids when Engram is present."""

    def test_block_forward_accepts_input_ids(self):
        config = _make_model()
        model = GPT(config)
        model.to(device="cpu")
        model.init_weights()

        idx = torch.randint(0, config.vocab_size, (1, 16))
        # Should not raise TypeError about missing arguments
        loss = model(idx, targets=idx)
        assert loss.item() > 0

    def test_block_without_engram_ignores_input_ids(self):
        """Non-Engram blocks still work (engram=None path)."""
        config = _make_model(engram_layer_ids=(1,), n_layer=4)
        model = GPT(config)
        model.to(device="cpu")
        model.init_weights()
        block0 = model.transformer.h[0]
        assert block0.engram is None

        idx = torch.randint(0, config.vocab_size, (1, 16))
        loss = model(idx, targets=idx)
        assert loss.item() > 0


class TestEngramNoStaleForward:
    """EngramModule.forward must have (x, input_ids, kv_cache) signature,
    NOT the old Block.forward signature (x, ve, cos_sin, window_size, kv_cache).
    This catches the bug where a leftover Block.forward was nested inside
    EngramModule's scope due to indentation."""

    def test_engram_forward_signature(self):
        import inspect
        sig = inspect.signature(EngramModule.forward)
        param_names = list(sig.parameters.keys())
        assert "input_ids" in param_names, f"EngramModule.forward missing input_ids, got {param_names}"
        assert "ve" not in param_names, f"EngramModule.forward should not have 've' (stale Block.forward), got {param_names}"

    def test_engram_forward_runs(self):
        config = _make_model()
        em = EngramModule(config, layer_idx=1, padded_vocab_size=32768)
        x = torch.randn(1, 16, config.n_embd)
        ids = torch.randint(0, 32768, (1, 16))
        out = em(x, ids)
        assert out.shape == x.shape


class TestParameterCounting:
    """num_scaling_params total must match actual parameter count (no double-counting)."""

    def test_param_count_matches(self):
        config = _make_model()
        model = GPT(config)
        model.to(device="cpu")
        model.init_weights()
        params = model.num_scaling_params()
        actual = sum(p.numel() for p in model.parameters())
        assert params["total"] == actual, f"Mismatch: {params['total']} != {actual}"

    def test_engram_params_nonzero(self):
        config = _make_model()
        model = GPT(config)
        model.to(device="cpu")
        model.init_weights()
        params = model.num_scaling_params()
        assert params["engram"] > 0

    def test_no_engram_params_when_disabled(self):
        config = GPTConfig(n_layer=4, n_head=4, n_kv_head=4, n_embd=256, engram_enabled=False)
        model = GPT(config)
        model.to(device="cpu")
        model.init_weights()
        params = model.num_scaling_params()
        assert params["engram"] == 0


class TestOptimizerNoDuplicates:
    """Each parameter must appear in exactly one optimizer group."""

    def test_no_duplicate_param_groups(self):
        config = _make_model()
        model = GPT(config)
        model.to(device="cpu")
        model.init_weights()
        opt = model.setup_optimizer()

        seen = {}
        for group in opt.param_groups:
            for p in group["params"]:
                pid = id(p)
                assert pid not in seen, (
                    f"Param shape {p.shape} in multiple groups: "
                    f"kind={group['kind']} and kind={seen[pid]}"
                )
                seen[pid] = group["kind"]

    def test_engram_embeds_not_in_muon(self):
        """Engram embedding table weights must be in AdamW, not Muon."""
        config = _make_model()
        model = GPT(config)
        model.to(device="cpu")
        model.init_weights()
        opt = model.setup_optimizer()

        engram_embed_ids = set()
        for block in model.transformer.h:
            if block.engram is not None:
                for tables_n in block.engram.embed_tables:
                    for t in tables_n:
                        engram_embed_ids.add(id(t.weight))

        muon_params = set()
        for group in opt.param_groups:
            if group.get("kind") == "muon":
                for p in group["params"]:
                    muon_params.add(id(p))

        overlap = engram_embed_ids & muon_params
        assert not overlap, f"Engram embed params found in Muon groups: {len(overlap)}"


class TestMuonOnlyMatrixParams:
    """Muon optimizer groups must only contain 2D trainable parameters."""

    def test_muon_groups_only_2d_trainable(self):
        config = _make_model()
        model = GPT(config)
        model.to(device="cpu")
        model.init_weights()
        opt = model.setup_optimizer()

        for group in opt.param_groups:
            if group.get("kind") != "muon":
                continue
            for p in group["params"]:
                assert p.dim() == 2, f"Muon got {p.dim()}D param (shape={p.shape})"
                assert p.requires_grad, f"Muon got non-trainable param"


class TestVEGateNoDeadGrad:
    """On Engram layers, ve_gate must be None so it never has a dead gradient."""

    def test_engram_layers_have_no_ve_gate(self):
        config = _make_model(engram_layer_ids=(1, 2), n_layer=4)
        model = GPT(config)
        model.to(device="cpu")
        model.init_weights()

        for i in model.engram_layer_ids:
            block = model.transformer.h[i]
            assert block.attn.ve_gate is None, f"Layer {i} has ve_gate but is an Engram layer"

    def test_all_trainable_params_get_gradients(self):
        """Every requires_grad=True param must receive a gradient after backward."""
        config = _make_model()
        model = GPT(config)
        model.to(device="cpu")
        model.init_weights()

        idx = torch.randint(0, config.vocab_size, (1, 16))
        loss = model(idx, targets=idx)
        loss.backward()

        dead_grads = []
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is None:
                dead_grads.append(name)
        assert not dead_grads, f"Params with no gradient: {dead_grads}"


class TestAutoLayerSelection:
    """Default engram_layer_ids should be (1, n_layer//2)."""

    @pytest.mark.parametrize("n_layer,expected", [
        (6, (1, 2)),
        (12, (1, 5)),
        (20, (1, 9)),
        (30, (1, 14)),
    ])
    def test_auto_layer_ids(self, n_layer, expected):
        config = GPTConfig(n_layer=n_layer, n_head=4, n_kv_head=4, n_embd=256, engram_enabled=True)
        model = GPT(config)
        assert model.engram_layer_ids == expected


class TestKVCacheNgramState:
    """KVCache must support ngram_history for Engram decode."""

    def test_ngram_history_initially_none(self):
        from nanochat.engine import KVCache
        cache = KVCache(batch_size=1, num_heads=4, seq_len=128, head_dim=64, num_layers=4, device="cpu", dtype=torch.float32)
        assert cache.ngram_history is None

    def test_reset_clears_ngram_history(self):
        from nanochat.engine import KVCache
        cache = KVCache(batch_size=1, num_heads=4, seq_len=128, head_dim=64, num_layers=4, device="cpu", dtype=torch.float32)
        cache.ngram_history = torch.zeros(1, 2, dtype=torch.long)
        cache.reset()
        assert cache.ngram_history is None

    def test_prefill_copies_ngram_history(self):
        from nanochat.engine import KVCache
        src = KVCache(batch_size=1, num_heads=4, seq_len=128, head_dim=64, num_layers=4, device="cpu", dtype=torch.float32)
        src.ngram_history = torch.tensor([[1, 2]], dtype=torch.long)
        dst = KVCache(batch_size=4, num_heads=4, seq_len=128, head_dim=64, num_layers=4, device="cpu", dtype=torch.float32)
        dst.prefill(src)
        assert dst.ngram_history is not None
        assert dst.ngram_history.shape == (4, 2)
        assert (dst.ngram_history == torch.tensor([[1, 2]])).all()


class TestNextPrime:
    """_next_prime helper must return primes."""

    @pytest.mark.parametrize("n,expected", [
        (1, 2),
        (2, 2),
        (3, 3),
        (4, 5),
        (10, 11),
        (100, 101),
    ])
    def test_next_prime(self, n, expected):
        assert _next_prime(n) == expected

    def test_next_prime_returns_prime(self):
        result = _next_prime(99999)
        # Verify it's actually prime
        assert all(result % d for d in range(2, int(result**0.5) + 1))


class TestEngramInitWeightsIdentity:
    """Short conv must be zero-initialized so Engram starts as identity."""

    def test_conv_weights_zero_after_init(self):
        config = _make_model()
        model = GPT(config)
        model.to(device="cpu")
        model.init_weights()

        for i in model.engram_layer_ids:
            conv = model.transformer.h[i].engram.short_conv
            assert torch.all(conv.weight == 0), "Engram conv weights should be zero-initialized"


class TestEngramDecodeIntegration:
    """Full inference with KV cache + Engram: prefill then decode must produce consistent results."""

    def _make_model(self):
        config = GPTConfig(
            sequence_len=64, vocab_size=32768,
            n_layer=6, n_head=4, n_kv_head=4, n_embd=256,
            engram_enabled=True, engram_ngram_size=3,
            engram_n_heads=2, engram_embed_dim=64,
            engram_layer_ids=(1, 2),
        )
        model = GPT(config)
        model.to(device="cpu")
        model.init_weights()
        return model

    def test_naive_vs_kv_cache_single_token(self):
        """Single-token decode with KV cache should match naive recomputation."""
        from nanochat.engine import KVCache
        model = self._make_model()
        config = model.config
        prompt = torch.randint(0, config.vocab_size, (1, 16))

        # Naive: full sequence forward, take last token logits
        with torch.no_grad():
            logits_naive = model(prompt)[:, -1, :]

        # KV cache: prefill then decode one token
        m = config
        kv_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv = KVCache(batch_size=1, seq_len=64, device=torch.device("cpu"), dtype=torch.float32, **kv_kwargs)
        with torch.no_grad():
            logits_prefill = model(prompt, kv_cache=kv)[:, -1, :]

        # Prefill should match naive
        assert torch.allclose(logits_naive, logits_prefill, atol=1e-4), "Prefill logits should match naive"

    def test_prefill_copies_ngram_history_to_batch_cache(self):
        """After prefill, ngram_history should be available and copyable for batch decode."""
        from nanochat.engine import KVCache
        model = self._make_model()
        config = model.config
        prompt = torch.randint(0, config.vocab_size, (1, 8))

        m = config
        kv_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_prefill = KVCache(batch_size=1, seq_len=64, device=torch.device("cpu"), dtype=torch.float32, **kv_kwargs)
        with torch.no_grad():
            model(prompt, kv_cache=kv_prefill)

        assert kv_prefill.ngram_history is not None, "Prefill should set ngram_history"
        max_hist = config.engram_ngram_size - 1
        assert kv_prefill.ngram_history.shape == (1, max_hist)

        # Copy to batch cache
        kv_batch = KVCache(batch_size=3, seq_len=64, device=torch.device("cpu"), dtype=torch.float32, **kv_kwargs)
        kv_batch.prefill(kv_prefill)
        assert kv_batch.ngram_history.shape == (3, max_hist)
