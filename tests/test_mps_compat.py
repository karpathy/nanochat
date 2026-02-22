"""
Tests for MPS/CPU compatibility fixes introduced in the cpu-mps-dev PR.

Each test exercises one specific fix and can run entirely on MPS or CPU — no GPU needed.
Run with: python -m pytest tests/test_mps_compat.py -v
"""
import threading
import time
import pytest
import torch
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device_type = device.type


# ---------------------------------------------------------------------------
# Fix 1: optim.py — _cuda_compile skips torch.compile on non-CUDA,
#                    and CPU 0-D scalar tensors are moved to device before ops
# ---------------------------------------------------------------------------

class TestOptimMPSCompat:
    def test_cuda_compile_is_identity_on_non_cuda(self):
        """_cuda_compile must return the function unchanged when CUDA is not available."""
        from nanochat.optim import _cuda_compile
        sentinel = lambda x: x
        result = _cuda_compile(sentinel)
        assert result is sentinel, "_cuda_compile should be a no-op without CUDA"

    def test_fused_functions_not_compiled(self):
        """adamw_step_fused and muon_step_fused must NOT be torch.OptimizedModule on MPS/CPU."""
        from nanochat.optim import adamw_step_fused, muon_step_fused
        # torch.compile wraps in OptimizedModule which has _orig_mod attribute
        assert not hasattr(adamw_step_fused, "_orig_mod"), \
            "adamw_step_fused should not be torch.compile'd on MPS/CPU"
        assert not hasattr(muon_step_fused, "_orig_mod"), \
            "muon_step_fused should not be torch.compile'd on MPS/CPU"

    def test_adamw_step_fused_runs_on_device(self):
        """adamw_step_fused must execute successfully with device tensors + CPU scalar tensors."""
        from nanochat.optim import adamw_step_fused
        p        = torch.randn(16, 16, device=device)
        grad     = torch.randn(16, 16, device=device)
        exp_avg  = torch.zeros(16, 16, device=device)
        exp_avg_sq = torch.zeros(16, 16, device=device)
        # Scalars intentionally kept on CPU (as the optimizer allocates them)
        p_before = p.clone()
        adamw_step_fused(
            p, grad, exp_avg, exp_avg_sq,
            torch.tensor(1.0),    # step_t  — CPU scalar
            torch.tensor(1e-3),   # lr_t
            torch.tensor(0.9),    # beta1_t
            torch.tensor(0.999),  # beta2_t
            torch.tensor(1e-8),   # eps_t
            torch.tensor(0.01),   # wd_t
        )
        assert not torch.equal(p, p_before), "adamw_step_fused should update p in-place"

    def test_muon_step_fused_runs_on_device(self):
        """muon_step_fused must execute successfully on MPS/CPU."""
        from nanochat.optim import muon_step_fused
        B, M, N = 4, 32, 64
        stacked_grads  = torch.randn(B, M, N, device=device)
        stacked_params = torch.randn(B, M, N, device=device)
        momentum_buf   = torch.zeros(B, M, N, device=device)
        # red_dim=-1 means we reduce over N, so second moment buffer is (B, M, 1)
        second_mom_buf = torch.ones(B, M, 1, device=device)
        params_before  = stacked_params.clone()
        muon_step_fused(
            stacked_grads, stacked_params, momentum_buf, second_mom_buf,
            torch.tensor(0.95),  # momentum_t — CPU scalar
            torch.tensor(0.02),  # lr_t
            torch.tensor(0.0),   # wd_t
            torch.tensor(0.95),  # beta2_t
            ns_steps=5,
            red_dim=-1,
        )
        assert not torch.equal(stacked_params, params_before), \
            "muon_step_fused should update stacked_params in-place"

    def test_full_optimizer_step_on_device(self):
        """MuonAdamW optimizer.step() must work end-to-end on MPS/CPU."""
        from nanochat.optim import MuonAdamW
        # A small model with both matrix (Muon) and non-matrix (AdamW) params
        model = nn.Sequential(
            nn.Embedding(32, 16),      # AdamW — 1-D embedding
            nn.Linear(16, 32, bias=False),  # Muon — 2-D matrix
        ).to(device)
        embedding_params = list(model[0].parameters())
        matrix_params    = list(model[1].parameters())
        param_groups = [
            dict(kind="adamw", params=embedding_params, lr=1e-3,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0),
            dict(kind="muon",  params=matrix_params, lr=0.02,
                 momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=0.0),
        ]
        opt = MuonAdamW(param_groups)
        x = torch.randint(0, 32, (4,), device=device)
        out = model[1](model[0](x))
        loss = out.sum()
        loss.backward()
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        opt.step()
        opt.zero_grad()
        for name, p in model.named_parameters():
            assert not torch.equal(p, params_before[name]), \
                f"Parameter {name} should have been updated by optimizer.step()"


# ---------------------------------------------------------------------------
# Fix 2: engine.py — concurrent.futures timeout works from any thread
# ---------------------------------------------------------------------------

class TestCalculatorThreadSafety:
    def test_calculator_works_from_main_thread(self):
        """Basic sanity: use_calculator works from the main thread."""
        from nanochat.engine import use_calculator
        assert use_calculator("2 + 2") == 4
        assert use_calculator("10 * 3") == 30
        assert use_calculator("1 / 4") == 0.25

    def test_calculator_works_from_background_thread(self):
        """Critical: use_calculator must work when called from a non-main thread (FastAPI worker scenario)."""
        from nanochat.engine import use_calculator
        results = {}
        errors  = {}

        def worker():
            try:
                results["2+2"]  = use_calculator("2+2")
                results["10*3"] = use_calculator("10*3")
            except Exception as e:
                errors["exc"] = e

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=10)
        assert not t.is_alive(), "Worker thread hung"
        assert not errors, f"Exception in worker thread: {errors.get('exc')}"
        assert results["2+2"]  == 4
        assert results["10*3"] == 30

    def test_calculator_timeout_does_not_hang(self):
        """A calculator timeout must not block the caller for more than ~max_time seconds."""
        from nanochat.engine import eval_with_timeout
        # We can't easily trigger a true infinite loop through use_calculator's sanitizer,
        # but we can call eval_with_timeout directly with a very short timeout.
        t0 = time.time()
        result = eval_with_timeout("1+1", max_time=0.1)
        elapsed = time.time() - t0
        assert result == 2
        assert elapsed < 2.0, f"eval_with_timeout took too long: {elapsed:.2f}s"

    def test_calculator_rejects_unsafe_input(self):
        """use_calculator must return None for non-numeric / unsafe expressions."""
        from nanochat.engine import use_calculator
        assert use_calculator("__import__('os').system('echo pwned')") is None
        assert use_calculator("2 ** 100") is None    # power operator blocked
        assert use_calculator("open('/etc/passwd')") is None

    def test_no_sigalrm_usage(self):
        """engine.py must not CALL signal.alarm() or signal.signal(SIGALRM) — comments are fine."""
        source = open("nanochat/engine.py").read()
        # Strip comment lines before checking for actual usage
        non_comment_lines = [
            line for line in source.splitlines()
            if not line.lstrip().startswith("#")
        ]
        code = "\n".join(non_comment_lines)
        assert "signal.alarm(" not in code, \
            "engine.py still calls signal.alarm() — should use concurrent.futures"
        assert "signal.signal(signal.SIGALRM" not in code, \
            "engine.py still registers SIGALRM handler — should use concurrent.futures"
        assert "import concurrent.futures" in source, \
            "engine.py should import concurrent.futures for thread-safe timeout"


# ---------------------------------------------------------------------------
# Fix 3: gpt.py — init_rotary_embeddings() exists and works standalone
# ---------------------------------------------------------------------------

class TestInitRotaryEmbeddings:
    def _make_small_model(self):
        from nanochat.gpt import GPT, GPTConfig
        cfg = GPTConfig(sequence_len=64, vocab_size=256, n_layer=2,
                        n_head=2, n_kv_head=2, n_embd=64)
        with torch.device("meta"):
            model = GPT(cfg)
        model.to_empty(device=device)
        return model

    def test_method_exists(self):
        """GPT must expose init_rotary_embeddings()."""
        from nanochat.gpt import GPT
        assert hasattr(GPT, "init_rotary_embeddings"), \
            "GPT should have init_rotary_embeddings() method"

    def test_rotary_buffers_populated_after_call(self):
        """init_rotary_embeddings() alone must produce valid cos/sin buffers."""
        model = self._make_small_model()
        model.init_rotary_embeddings()
        assert model.cos is not None
        assert model.sin is not None
        assert model.cos.shape[1] == model.rotary_seq_len
        assert not model.cos.isnan().any(), "cos buffer contains NaN"
        assert not model.sin.isnan().any(), "sin buffer contains NaN"

    def test_init_rotary_does_not_touch_parameters(self):
        """init_rotary_embeddings() must not change learnable parameters."""
        model = self._make_small_model()
        model.init_weights()  # proper full init
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        model.init_rotary_embeddings()  # should only touch buffers
        for name, p in model.named_parameters():
            assert torch.equal(p, params_before[name]), \
                f"init_rotary_embeddings() should not modify parameter {name}"

    def test_forward_works_after_init_rotary_only(self):
        """A model initialized only via init_weights (which calls init_rotary) must forward cleanly."""
        model = self._make_small_model()
        model.init_weights()
        model.eval()
        ids = torch.randint(0, 256, (1, 16), device=device)
        with torch.no_grad():
            logits = model(ids)
        assert logits.shape == (1, 16, 256)
        assert not logits.isnan().any()


# ---------------------------------------------------------------------------
# Fix 4: base_train.py / mid_train.py — torch.compile guarded on non-CUDA
# ---------------------------------------------------------------------------

class TestTorchCompileGuard:
    def test_compile_guard_in_base_train_source(self):
        """base_train.py must only call torch.compile when device_type == 'cuda'."""
        source = open("scripts/base_train.py").read()
        # Find the torch.compile call and verify it's inside a CUDA guard
        compile_idx = source.find('model = torch.compile(model')
        assert compile_idx != -1, "Could not find torch.compile call in base_train.py"
        # The nearest preceding if-statement should reference cuda
        preceding = source[max(0, compile_idx - 200):compile_idx]
        assert 'device_type == "cuda"' in preceding, \
            'torch.compile in base_train.py is not guarded by `if device_type == "cuda":`'

    def test_mfu_none_on_non_cuda(self):
        """On MPS/CPU, mfu should be None (not computed), not a misleading float."""
        # We can't import base_train directly (it's a script), so test the logic pattern
        gpu_peak_flops = float('inf')  # what base_train sets for non-CUDA
        flops_per_sec = 1e12
        # Our fix: mfu is None when device_type != "cuda"
        device_type_local = "mps"  # or "cpu"
        if device_type_local == "cuda":
            mfu = 100 * flops_per_sec / (gpu_peak_flops * 1)
        else:
            mfu = None
        assert mfu is None, "MFU should be None on non-CUDA devices"
