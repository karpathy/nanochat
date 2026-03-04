from types import SimpleNamespace

import pytest
import torch

import nanochat.gpt as gpt


def _cfg(**overrides):
    base = dict(sequence_len=8, vocab_size=13, n_layer=2, n_head=2, n_kv_head=1, n_embd=4, window_pattern="SL")
    base.update(overrides)
    return gpt.GPTConfig(**base)


def test_norm_has_ve_and_rotary():
    x = torch.randn(2, 3, 4)
    y = gpt.norm(x)
    assert y.shape == x.shape
    assert gpt.has_ve(1, 4) is True
    assert gpt.has_ve(0, 4) is False

    h = torch.randn(1, 2, 2, 4)
    cos = torch.ones(1, 2, 1, 2)
    sin = torch.zeros(1, 2, 1, 2)
    out = gpt.apply_rotary_emb(h, cos, sin)
    assert out.shape == h.shape


def test_attention_forward_paths(monkeypatch):
    cfg = _cfg(n_layer=1)
    attn = gpt.CausalSelfAttention(cfg, layer_idx=0)

    calls = {"func": 0, "kvcache": 0}

    def fake_func(q, k, v, causal, window_size):
        del k, v, causal, window_size
        calls["func"] += 1
        return q

    def fake_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None, causal=False, window_size=(-1, -1)):
        del k_cache, v_cache, k, v, cache_seqlens, causal, window_size
        calls["kvcache"] += 1
        return q

    monkeypatch.setattr(gpt.flash_attn, "flash_attn_func", fake_func)
    monkeypatch.setattr(gpt.flash_attn, "flash_attn_with_kvcache", fake_kvcache)

    x = torch.randn(2, 3, cfg.n_embd)
    cos = torch.ones(1, 3, 1, cfg.n_embd // cfg.n_head // 2)
    sin = torch.zeros_like(cos)
    y = attn(x, ve=None, cos_sin=(cos, sin), window_size=(cfg.sequence_len, 0), kv_cache=None)
    assert y.shape == x.shape
    assert calls["func"] == 1

    class Cache:
        def __init__(self):
            self.n_layers = 1
            self.cache_seqlens = torch.zeros(2, dtype=torch.int32)
            self.k = torch.zeros(2, 10, cfg.n_kv_head, cfg.n_embd // cfg.n_head)
            self.v = torch.zeros(2, 10, cfg.n_kv_head, cfg.n_embd // cfg.n_head)
            self.advanced = 0

        def get_layer_cache(self, _idx):
            return self.k, self.v

        def advance(self, t):
            self.advanced += t

    ve = torch.randn(2, 3, cfg.n_kv_head * (cfg.n_embd // cfg.n_head))
    attn.ve_gate_channels = cfg.n_embd
    attn.ve_gate = torch.nn.Linear(cfg.n_embd, cfg.n_kv_head, bias=False)
    cache = Cache()
    y2 = attn(x, ve=ve, cos_sin=(cos, sin), window_size=(cfg.sequence_len, 0), kv_cache=cache)
    assert y2.shape == x.shape
    assert calls["kvcache"] == 1
    assert cache.advanced == 3

    with pytest.raises(AssertionError):
        gpt.CausalSelfAttention(_cfg(n_embd=5, n_head=2), layer_idx=0)
    with pytest.raises(AssertionError):
        gpt.CausalSelfAttention(_cfg(n_head=3, n_kv_head=2), layer_idx=0)


def test_mlp_and_block(monkeypatch):
    cfg = _cfg(n_layer=1)
    mlp = gpt.MLP(cfg)
    x = torch.randn(2, 3, cfg.n_embd)
    y = mlp(x)
    assert y.shape == x.shape

    monkeypatch.setattr(gpt.flash_attn, "flash_attn_func", lambda q, k, v, causal, window_size: q)
    block = gpt.Block(cfg, layer_idx=0)
    cos = torch.ones(1, 3, 1, cfg.n_embd // cfg.n_head // 2)
    sin = torch.zeros_like(cos)
    out = block(x, ve=None, cos_sin=(cos, sin), window_size=(cfg.sequence_len, 0), kv_cache=None)
    assert out.shape == x.shape


def test_gpt_core_helpers_and_forward(monkeypatch):
    monkeypatch.setattr(gpt, "print0", lambda *a, **k: None)
    monkeypatch.setattr(gpt.flash_attn, "flash_attn_func", lambda q, k, v, causal, window_size: q)
    monkeypatch.setattr(gpt.flash_attn, "flash_attn_with_kvcache", lambda q, k_cache, v_cache, **kw: q)

    model = gpt.GPT(_cfg(), pad_vocab_size_to=8)
    model.init_weights()
    for block in model.transformer.h:
        if block.attn.ve_gate is not None:
            block.attn.ve_gate_channels = model.config.n_embd
            block.attn.ve_gate = torch.nn.Linear(model.config.n_embd, model.config.n_kv_head, bias=False)

    ws = model._compute_window_sizes(_cfg(window_pattern="SLS", n_layer=4))
    assert len(ws) == 4
    assert ws[-1][0] == 8
    with pytest.raises(AssertionError):
        model._compute_window_sizes(_cfg(window_pattern="Q"))

    cos, sin = model._precompute_rotary_embeddings(4, 2, base=1000, device=torch.device("cpu"))
    assert cos.shape == sin.shape

    assert model.get_device().type == "cpu"
    assert model.estimate_flops() > 0
    counts = model.num_scaling_params()
    assert counts["total"] == sum(p.numel() for p in model.parameters())

    idx = torch.randint(0, model.config.vocab_size, (2, 4), dtype=torch.long)
    logits = model.forward(idx)
    assert logits.shape == (2, 4, model.config.vocab_size)

    targets = torch.randint(0, model.config.vocab_size, (2, 4), dtype=torch.long)
    loss = model.forward(idx, targets=targets, loss_reduction="mean")
    assert loss.ndim == 0

    with pytest.raises(AssertionError):
        model.forward(torch.randint(0, model.config.vocab_size, (1, model.cos.size(1) + 1)))


def test_setup_optimizer_paths(monkeypatch):
    model = gpt.GPT(_cfg(), pad_vocab_size_to=8)

    class FakeOpt:
        def __init__(self, groups):
            self.param_groups = groups

    monkeypatch.setattr(gpt, "print0", lambda *a, **k: None)
    monkeypatch.setattr(gpt, "MuonAdamW", FakeOpt)
    monkeypatch.setattr(gpt, "DistMuonAdamW", FakeOpt)

    monkeypatch.setattr(gpt, "get_dist_info", lambda: (False, 0, 0, 1))
    opt1 = model.setup_optimizer()
    assert isinstance(opt1, FakeOpt)
    assert all("initial_lr" in g for g in opt1.param_groups)

    monkeypatch.setattr(gpt, "get_dist_info", lambda: (True, 0, 0, 2))
    opt2 = model.setup_optimizer()
    assert isinstance(opt2, FakeOpt)


def test_generate_paths(monkeypatch):
    model = gpt.GPT(_cfg(), pad_vocab_size_to=8)

    # Force predictable logits.
    def fake_forward(ids, *args, **kwargs):
        b, t = ids.shape
        v = model.config.vocab_size
        logits = torch.zeros((b, t, v), dtype=torch.float32)
        logits[..., 2] = 2.0
        logits[..., 3] = 1.0
        return logits

    monkeypatch.setattr(model, "forward", fake_forward)
    tokens = [1, 2]
    out_temp0 = list(model.generate(tokens, max_tokens=3, temperature=0.0, top_k=None, seed=42))
    assert out_temp0 == [2, 2, 2]

    out_topk = list(model.generate(tokens, max_tokens=2, temperature=1.0, top_k=1, seed=42))
    assert out_topk == [2, 2]

    with pytest.raises(AssertionError):
        list(model.generate("bad", max_tokens=1))  # type: ignore[arg-type]


def test_init_weights_cuda_cast_branch_with_fake_self(monkeypatch):
    # This executes the CUDA-only cast lines in init_weights without requiring a real CUDA device.
    class FakeTensor:
        def __init__(self):
            self.device = SimpleNamespace(type="cuda")

    class FakeParam:
        def __init__(self):
            self.weight = FakeTensor()
            self.to_calls = 0

        def to(self, dtype):
            del dtype
            self.to_calls += 1

    class FakeAttn:
        def __init__(self):
            self.c_q = SimpleNamespace(weight=object())
            self.c_k = SimpleNamespace(weight=object())
            self.c_v = SimpleNamespace(weight=object())
            self.c_proj = SimpleNamespace(weight=object())
            self.ve_gate = SimpleNamespace(weight=object())

    class FakeMLP:
        def __init__(self):
            self.c_fc = SimpleNamespace(weight=object())
            self.c_proj = SimpleNamespace(weight=object())

    class FakeBlock:
        def __init__(self):
            self.attn = FakeAttn()
            self.mlp = FakeMLP()

    class Fillable:
        def fill_(self, _v):
            return self

        def numel(self):
            return 1

    fake_self = SimpleNamespace()
    fake_self.config = _cfg()
    fake_self.rotary_seq_len = 8
    fake_self.transformer = SimpleNamespace(
        wte=FakeParam(),
        h=[FakeBlock(), FakeBlock()],
    )
    fake_self.lm_head = SimpleNamespace(weight=object())
    fake_self.resid_lambdas = Fillable()
    fake_self.x0_lambdas = Fillable()
    ve0 = FakeParam()
    fake_self.value_embeds = {"0": ve0}
    fake_self._precompute_rotary_embeddings = lambda *a, **k: (torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16), torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16))

    monkeypatch.setattr(gpt.torch.nn.init, "normal_", lambda *a, **k: None)
    monkeypatch.setattr(gpt.torch.nn.init, "uniform_", lambda *a, **k: None)
    monkeypatch.setattr(gpt.torch.nn.init, "zeros_", lambda *a, **k: None)

    gpt.GPT.init_weights(fake_self)
    assert fake_self.transformer.wte.to_calls == 1
    assert ve0.to_calls == 1
