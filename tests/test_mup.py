"""
Test muP implementation: coordinate check and transfer check.

Verifies that:
1. Activation magnitudes are width-independent under muP (coord check)
2. Optimal learning rates transfer across widths under muP (transfer check)

python -m pytest tests/test_mup.py -v
"""

import pytest
import torch
import torch._dynamo
torch._dynamo.config.disable = True
import numpy as np
from collections import defaultdict

from nanochat.gpt import GPT, GPTConfig


def create_model(width, seq_len=64, n_layer=2, vocab_size=256, mup_base_width=0):
    """Create a small model at the given width."""
    head_dim = 64
    n_head = max(1, width // head_dim)
    actual_width = n_head * head_dim
    config = GPTConfig(
        sequence_len=seq_len, vocab_size=vocab_size,
        n_layer=n_layer, n_head=n_head, n_kv_head=n_head,
        n_embd=actual_width, window_pattern="L",
        mup_base_width=mup_base_width,
    )
    with torch.device('meta'):
        model = GPT(config)
    model.to_empty(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.init_weights()
    return model, config


def get_activation_stats(model, x, y):
    """Run one forward pass and return mean |activation| for key layers."""
    stats = {}

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if output is not None and isinstance(output, torch.Tensor):
                stats[name] = output.float().abs().mean().item()
        return hook

    hooks = []
    hooks.append(model.transformer.wte.register_forward_hook(make_hook('embedding')))
    for i, block in enumerate(model.transformer.h):
        hooks.append(block.attn.c_proj.register_forward_hook(make_hook(f'attn.{i}')))
        hooks.append(block.mlp.c_proj.register_forward_hook(make_hook(f'ffn.{i}')))

    # Output logits with muP scaling applied
    mup_base = model.config.mup_base_width
    n_embd = model.config.n_embd
    def logit_hook(module, input, output):
        if output is not None and isinstance(output, torch.Tensor):
            scaled = output * (mup_base / n_embd) if mup_base > 0 else output
            stats['logits'] = scaled.float().abs().mean().item()
    hooks.append(model.lm_head.register_forward_hook(logit_hook))

    model.eval()
    with torch.no_grad():
        model(x, y)

    for h in hooks:
        h.remove()
    return stats


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestMuPCoordCheck:
    """Test that muP activations are width-independent."""

    WIDTHS = [128, 256, 512]
    BASE_WIDTH = 128

    def _compute_slopes(self, use_mup):
        """Run coord check and return log-log slopes for each layer."""
        device = torch.device('cuda')
        seq_len, vocab_size = 64, 256
        torch.manual_seed(42)
        x = torch.randint(0, vocab_size, (4, seq_len), device=device)
        y = torch.roll(x, -1, dims=1)

        all_stats = {}
        for width in self.WIDTHS:
            torch.manual_seed(42)
            mup_base = self.BASE_WIDTH if use_mup else 0
            model, _ = create_model(width, seq_len, vocab_size=vocab_size, mup_base_width=mup_base)
            all_stats[width] = get_activation_stats(model, x, y)
            del model
            torch.cuda.empty_cache()

        # Compute slopes on log-log plot
        log_widths = np.log2(np.array(self.WIDTHS, dtype=float))
        slopes = {}
        for layer in all_stats[self.WIDTHS[0]]:
            values = [all_stats[w][layer] for w in self.WIDTHS]
            log_values = np.log2(np.array(values) + 1e-10)
            slope, _ = np.polyfit(log_widths, log_values, 1)
            slopes[layer] = slope

        return slopes

    def test_mup_activations_width_independent(self):
        """Under muP, internal activation slopes should be near zero.

        Note: output logits are excluded because at init (no training steps),
        muP logits are expected to decrease with width — the init scaling
        (std * sqrt(base/width)) combined with forward scaling (logits * base/width)
        gives O(1/width) initial logits. muP preserves update magnitude, not init magnitude.
        """
        slopes = self._compute_slopes(use_mup=True)
        for layer, slope in slopes.items():
            if layer == 'logits':
                continue  # skip — output logits have expected width dependence at init
            assert abs(slope) < 0.2, \
                f"muP activation slope for '{layer}' is {slope:.4f}, expected near 0"

    def test_sp_activations_width_dependent(self):
        """Under SP, at least some activation slopes should be nonzero (sanity check)."""
        slopes = self._compute_slopes(use_mup=False)
        max_slope = max(abs(s) for s in slopes.values())
        assert max_slope > 0.1, \
            f"SP max slope is only {max_slope:.4f}, expected SP to show width dependence"


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestMuPTransfer:
    """Test that optimal LR transfers across widths under muP."""

    WIDTHS = [128, 256]
    BASE_WIDTH = 128
    LR_MULTS = [0.25, 1.0, 4.0, 16.0]
    NUM_STEPS = 50

    def _run_sweep(self, use_mup):
        """Sweep LR multipliers across widths, return optimal mult per width."""
        device = torch.device('cuda')
        seq_len, vocab_size = 64, 256
        matrix_lr = 0.12
        embedding_lr = 6.0
        unembedding_lr = 0.12

        torch.manual_seed(42)
        x = torch.randint(0, vocab_size, (8, seq_len), device=device)
        y = torch.roll(x, -1, dims=1)

        optimal_mults = {}
        for width in self.WIDTHS:
            best_loss, best_mult = float('inf'), None
            for lr_mult in self.LR_MULTS:
                torch.manual_seed(42)
                mup_base = self.BASE_WIDTH if use_mup else 0
                model, _ = create_model(width, seq_len, vocab_size=vocab_size, mup_base_width=mup_base)
                optimizer = model.setup_optimizer(
                    matrix_lr=matrix_lr * lr_mult,
                    embedding_lr=embedding_lr * lr_mult,
                    unembedding_lr=unembedding_lr * lr_mult,
                    weight_decay=0.0,
                    use_mup=use_mup,
                    base_width=self.BASE_WIDTH,
                )
                model.train()
                for _ in range(self.NUM_STEPS):
                    loss = model(x, y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                final_loss = loss.item()
                if final_loss < best_loss:
                    best_loss = final_loss
                    best_mult = lr_mult

                del model, optimizer
                torch.cuda.empty_cache()

            optimal_mults[width] = best_mult

        return optimal_mults

    def test_mup_lr_transfer(self):
        """Under muP, optimal LR multiplier should be similar across widths."""
        optimal = self._run_sweep(use_mup=True)
        mults = list(optimal.values())
        spread = np.log2(max(mults)) - np.log2(min(mults))
        assert spread <= 2.0, \
            f"muP LR spread is {spread:.1f} log2 (optimal mults: {optimal}), expected <= 2.0"
