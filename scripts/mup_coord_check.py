"""
muP Coordinate Check for nanochat

This script validates muP implementation by checking that activation magnitudes
are independent of model width. Based on EleutherAI's nanoGPT-mup and Microsoft's
mup library.

Reference: https://blog.eleuther.ai/mutransfer/
Reference: Yang et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot
           Hyperparameter Transfer" (arXiv:2203.03466), Sections B.1 and F.

Usage:
    python -m scripts.mup_coord_check --widths 128,256,512,1024 --steps 10
    python -m scripts.mup_coord_check --use-mup --widths 128,256,512,1024
    python -m scripts.mup_coord_check --compare --detailed
    python -m scripts.mup_coord_check --compare --muon-lr-exponent 0.5
"""

import argparse
import os
os.environ["NANOCHAT_DTYPE"] = "float32"
import torch
import torch._dynamo
torch._dynamo.config.disable = True
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os

from nanochat.gpt import GPT, GPTConfig


def load_batch(batch_size: int, seq_len: int, device: torch.device):
    """Load a single batch from the nanochat training pipeline.
    Falls back to random data if the tokenizer/dataset isn't available."""
    try:
        from nanochat.tokenizer import get_tokenizer
        from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
        tokenizer = get_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, batch_size, seq_len, split="train", device=device,
        )
        x, y = next(loader)
        print(f"Loaded real training data (vocab_size={vocab_size})")
        return x, y, vocab_size
    except Exception as e:
        print(f"Could not load training data ({e}), using random tokens")
        vocab_size = 32768
        rng = torch.Generator(device=device)
        rng.manual_seed(42)
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, generator=rng)
        y = torch.roll(x, -1, dims=1)
        y[:, -1] = -1
        return x, y, vocab_size


@dataclass
class CoordCheckConfig:
    widths: List[int]
    num_steps: int = 10
    batch_size: int = 4
    seq_len: int = 128
    vocab_size: int = 32768
    n_layer: int = 2
    seed: int = 42
    use_mup: bool = False
    base_width: int = 128
    # Learning rates (tuned at base_width=128)
    matrix_lr: float = 0.12
    embedding_lr: float = 6.0
    unembedding_lr: float = 0.12
    # Detailed diagnostics
    detailed: bool = False
    # Muon LR exponent: 1.0 = base/width (standard muP), 0.5 = sqrt(base/width)
    # Paper Section C.1: Frobenius-normalizing optimizers may need exponent 0.5
    muon_lr_exponent: float = 0.0


class ActivationRecorder:
    """Records activation statistics during forward pass using hooks."""

    def __init__(self, detailed: bool = False):
        self.stats: Dict[str, List[float]] = defaultdict(list)
        self.hooks = []
        self.detailed = detailed

    def _get_stat(self, tensor: torch.Tensor) -> float:
        """Compute mean absolute value (l1 norm per element)."""
        if tensor is None:
            return 0.0
        if tensor.dtype == torch.bool:
            return tensor.float().abs().mean().item()
        return tensor.float().abs().mean().item()

    def _make_hook(self, name: str):
        """Create a forward hook that records output statistics."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if output is not None and isinstance(output, torch.Tensor):
                self.stats[name].append(self._get_stat(output))
        return hook

    def _make_attn_logit_hook(self, name: str, n_head: int, n_kv_head: int, head_dim: int):
        """Create a hook on c_k that computes pre-softmax attention logit magnitudes.

        We hook onto c_k's forward, then use the most recent c_q output to compute
        q @ k^T / sqrt(d) for a single batch element to measure attention logit scale.
        """
        # We'll store q output and compute logits when k is available
        self._last_q = None

        def q_hook(module, input, output):
            self._last_q = output.detach()

        def k_hook(module, input, output):
            if self._last_q is None:
                return
            q = self._last_q
            k = output.detach()
            B, T, _ = q.shape
            q = q[0:1].view(1, T, n_head, head_dim)
            k = k[0:1].view(1, T, n_kv_head, head_dim)
            # Apply QK norm (same as model)
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
            # Expand k for GQA
            if n_head != n_kv_head:
                k = k.repeat_interleave(n_head // n_kv_head, dim=2)
            # Compute logits: q @ k^T / sqrt(d) — just for first few positions
            T_sub = min(T, 32)
            q_sub = q[:, :T_sub].transpose(1, 2)  # (1, H, T_sub, D)
            k_sub = k[:, :T_sub].transpose(1, 2)  # (1, H, T_sub, D)
            logits = torch.matmul(q_sub, k_sub.transpose(-2, -1)) / (head_dim ** 0.5)
            self.stats[name].append(logits.float().abs().mean().item())
            self._last_q = None

        return q_hook, k_hook

    def register_hooks(self, model: GPT) -> None:
        """Register forward hooks on key layers."""
        # Embedding
        h = model.transformer.wte.register_forward_hook(self._make_hook('word embedding'))
        self.hooks.append(h)

        # Each transformer block
        for i, block in enumerate(model.transformer.h):
            # Attention output
            h = block.attn.c_proj.register_forward_hook(self._make_hook(f'attn output.{i}'))
            self.hooks.append(h)
            # MLP output
            h = block.mlp.c_proj.register_forward_hook(self._make_hook(f'FFN output.{i}'))
            self.hooks.append(h)

            # Detailed: attention logit magnitudes
            if self.detailed:
                n_head = block.attn.n_head
                n_kv_head = block.attn.n_kv_head
                head_dim = block.attn.head_dim
                q_hook, k_hook = self._make_attn_logit_hook(
                    f'attn logits.{i}', n_head, n_kv_head, head_dim)
                h1 = block.attn.c_q.register_forward_hook(q_hook)
                h2 = block.attn.c_k.register_forward_hook(k_hook)
                self.hooks.extend([h1, h2])

        # Output logits: hook on lm_head, but apply muP scaling to match what forward() does
        mup_base = model.config.mup_base_width
        n_embd = model.config.n_embd
        def logit_hook(module, input, output):
            if output is not None and isinstance(output, torch.Tensor):
                scaled = output
                if mup_base > 0:
                    scaled = output * (mup_base / n_embd)
                self.stats['output logits'].append(self._get_stat(scaled))
        h = model.lm_head.register_forward_hook(logit_hook)
        self.hooks.append(h)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_step_stats(self) -> Dict[str, float]:
        """Get mean stats for the current step and reset."""
        step_stats = {}
        for name, values in self.stats.items():
            if values:
                step_stats[name] = np.mean(values)
        self.stats = defaultdict(list)
        return step_stats


def create_model(width: int, config: CoordCheckConfig, device: torch.device, mup_base_width: int = 0) -> Tuple[GPT, GPTConfig]:
    """Create a model with the specified width."""
    head_dim = 64
    n_head = max(1, width // head_dim)
    actual_width = n_head * head_dim

    gpt_config = GPTConfig(
        sequence_len=config.seq_len,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=n_head,
        n_kv_head=n_head,
        n_embd=actual_width,
        window_pattern="L",
        mup_base_width=mup_base_width,
    )

    with torch.device('meta'):
        model = GPT(gpt_config)
    model.to_empty(device=device)
    model.init_weights()

    return model, gpt_config


def setup_optimizer_mup(model: GPT, config: CoordCheckConfig, width: int):
    """Set up optimizer with muP scaling using the native use_mup flag."""
    optimizer = model.setup_optimizer(
        unembedding_lr=config.unembedding_lr,
        embedding_lr=config.embedding_lr,
        matrix_lr=config.matrix_lr,
        weight_decay=0.0,
        use_mup=True,
        base_width=config.base_width,
        muon_lr_exponent=config.muon_lr_exponent,
    )
    return optimizer


def setup_optimizer_sp(model: GPT, config: CoordCheckConfig, width: int):
    """Set up optimizer with standard parameterization (current nanochat)."""
    optimizer = model.setup_optimizer(
        unembedding_lr=config.unembedding_lr,
        embedding_lr=config.embedding_lr,
        matrix_lr=config.matrix_lr,
        weight_decay=0.0,
        use_mup=False,
    )
    return optimizer


def record_detailed_stats(model: GPT, results: Dict, width: int, step: int):
    """Record weight update norms and gradient norms per parameter group."""
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        # Simplify name for display
        short_name = name.replace('transformer.', '').replace('.weight', '')
        # Gradient norm
        grad_norm = p.grad.float().norm().item()
        results['detailed_stats'][width][f'grad norm: {short_name}'].append(grad_norm)


def record_weight_update_norms(model: GPT, params_before: Dict[str, torch.Tensor],
                                results: Dict, width: int):
    """Record ||delta_W|| for each parameter after optimizer step."""
    for name, p in model.named_parameters():
        if name not in params_before:
            continue
        short_name = name.replace('transformer.', '').replace('.weight', '')
        delta = (p.data.float() - params_before[name]).norm().item()
        results['detailed_stats'][width][f'update norm: {short_name}'].append(delta)


def run_coord_check(config: CoordCheckConfig, device: torch.device,
                    x: torch.Tensor, y: torch.Tensor) -> Dict:
    """Run coordinate check across all widths."""
    results = {
        'widths': [],
        'steps': list(range(config.num_steps)),
        'stats': defaultdict(lambda: defaultdict(list)),
        'losses': defaultdict(list),
        'detailed_stats': defaultdict(lambda: defaultdict(list)),
    }

    for width in config.widths:
        print(f"\nTraining width={width}...")

        torch.manual_seed(config.seed)

        mup_base_width = config.base_width if config.use_mup else 0
        model, gpt_config = create_model(width, config, device, mup_base_width=mup_base_width)
        actual_width = gpt_config.n_embd
        results['widths'].append(actual_width)

        if config.use_mup:
            optimizer = setup_optimizer_mup(model, config, actual_width)
        else:
            optimizer = setup_optimizer_sp(model, config, actual_width)

        recorder = ActivationRecorder(detailed=config.detailed)
        recorder.register_hooks(model)

        model.train()

        for step in range(config.num_steps):
            with torch.amp.autocast(device_type='cuda', dtype=torch.float32, enabled=False):
                loss = model(x, y)

            results['losses'][actual_width].append(loss.item())

            step_stats = recorder.get_step_stats()
            for layer, value in step_stats.items():
                results['stats'][actual_width][layer].append(value)

            if step == 0:
                print(f"  Step {step}: loss={loss.item():.4f}, layers={list(step_stats.keys())}")

            # Record gradient norms before step (detailed mode)
            loss.backward()

            if config.detailed:
                record_detailed_stats(model, results, actual_width, step)
                # Snapshot params before optimizer step to compute update norms
                params_before = {name: p.data.float().clone()
                                 for name, p in model.named_parameters()
                                 if p.grad is not None}

            optimizer.step()

            if config.detailed:
                record_weight_update_norms(model, params_before, results, actual_width)

            optimizer.zero_grad(set_to_none=True)

        print(f"  Final loss: {loss.item():.4f}")

        recorder.remove_hooks()
        del model, optimizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results


def plot_coord_check(results: Dict, config: CoordCheckConfig, save_path: Optional[str] = None):
    """Plot coordinate check: one subplot per layer, x=width (log2), y=mean |activation|, lines=steps."""
    widths = results['widths']
    steps = results['steps']
    stats = results['stats']

    layer_names = list(stats[widths[0]].keys())
    n_layers = len(layer_names)
    n_cols = 4
    n_rows = (n_layers + n_cols - 1) // n_cols

    param_type = "muP" if config.use_mup else "SP"
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.array(axes).flatten()

    step_colors = plt.cm.plasma(np.linspace(0, 1, len(steps)))

    for i, layer in enumerate(layer_names):
        ax = axes[i]
        for s, step in enumerate(steps):
            values = [stats[w][layer][s] for w in widths]
            ax.plot(widths, values, 'o-', color=step_colors[s], linewidth=1.5,
                    label=f'step {step}' if i == 0 else None)
        ax.set_xscale('log', base=2)
        ax.set_xticks(widths)
        ax.set_xticklabels(widths, fontsize=7)
        ax.set_title(layer, fontsize=9)
        ax.set_xlabel('Width')
        ax.set_ylabel('Mean |activation|')
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=7, loc='best')

    for i in range(n_layers, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f'Coordinate Check ({param_type}): Activation Magnitude vs Width', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.show()


def plot_loss_curves(results: Dict, config: CoordCheckConfig, title: str = "", save_path: Optional[str] = None):
    """Plot loss curves across widths to verify HP transfer."""
    widths = results['widths']
    steps = results['steps']
    losses = results['losses']

    fig, ax = plt.subplots(figsize=(5 * 2, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(widths)))

    for i, w in enumerate(widths):
        ax.plot(steps, losses[w], label=f'width={w}', color=colors[i], linewidth=2)

    ax.set_yscale('log')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss Curves Across Widths{" - " + title if title else ""}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation for final loss spread
    final_losses = [losses[w][-1] for w in widths]
    spread = max(final_losses) - min(final_losses)
    ax.annotate(f'Final loss spread: {spread:.4f}', xy=(0.7, 0.95), xycoords='axes fraction', fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved loss curves to {save_path}")

    plt.show()


def plot_comparison(results_sp: Dict, results_mup: Dict, config: CoordCheckConfig, save_path: Optional[str] = None):
    """Plot SP vs muP: one subplot per layer (left=SP, right=muP), x=width (log2), y=mean |activation|, lines=steps."""
    widths = results_sp['widths']
    steps = results_sp['steps']

    layer_names = list(results_sp['stats'][widths[0]].keys())
    n_layers = len(layer_names)

    # n_layers activation rows + 1 loss row, 2 cols (SP | muP)
    n_rows, n_cols = n_layers + 1, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))

    step_colors = plt.cm.plasma(np.linspace(0, 1, len(steps)))
    width_colors = plt.cm.viridis(np.linspace(0, 1, len(widths)))

    for row, layer in enumerate(layer_names):
        # Shared y-axis range across SP and muP for this layer
        all_vals = [results_sp['stats'][w][layer][s] for w in widths for s in range(len(steps))] + \
                   [results_mup['stats'][w][layer][s] for w in widths for s in range(len(steps))]
        y_min, y_max = min(all_vals) * 0.9, max(all_vals) * 1.1

        for col, (results, label) in enumerate([(results_sp, 'SP'), (results_mup, 'muP')]):
            ax = axes[row, col]
            for s, step in enumerate(steps):
                values = [results['stats'][w][layer][s] for w in widths]
                ax.plot(widths, values, 'o-', color=step_colors[s], linewidth=1.5,
                        label=f'step {step}' if (row == 0 and col == 0) else None)
            ax.set_xscale('log', base=2)
            ax.set_xticks(widths)
            ax.set_xticklabels(widths, fontsize=7)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f'{label}: {layer}', fontsize=9)
            ax.set_xlabel('Width')
            ax.set_ylabel('Mean |activation|')
            ax.grid(True, alpha=0.3)

    axes[0, 0].legend(fontsize=7, loc='best')

    # Loss curves row (log scale so low-loss detail is visible)
    all_losses = [v for r in (results_sp, results_mup) for w in widths for v in r['losses'][w]]
    loss_min, loss_max = min(all_losses) * 0.9, max(all_losses) * 1.1

    for col, (results, label) in enumerate([(results_sp, 'SP'), (results_mup, 'muP')]):
        ax = axes[n_layers, col]
        for j, w in enumerate(widths):
            ax.plot(steps, results['losses'][w], label=f'w={w}', color=width_colors[j], linewidth=2)
        ax.set_yscale('log')
        ax.set_ylim(loss_min, loss_max)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title(f'{label}: Loss Curves')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        final_losses = [results['losses'][w][-1] for w in widths]
        spread = max(final_losses) - min(final_losses)
        ax.annotate(f'Spread: {spread:.4f}', xy=(0.65, 0.95), xycoords='axes fraction', fontsize=9)

    fig.suptitle('Coordinate Check: SP vs muP', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    plt.show()


def plot_detailed(results: Dict, config: CoordCheckConfig, save_path: Optional[str] = None):
    """Plot detailed diagnostics: gradient norms, weight update norms, attention logits."""
    widths = results['widths']
    detailed = results['detailed_stats']
    if not detailed or not detailed[widths[0]]:
        print("No detailed stats recorded. Use --detailed flag.")
        return

    # Collect all detailed metric names
    metric_names = sorted(detailed[widths[0]].keys())

    # Group by category
    categories = defaultdict(list)
    for name in metric_names:
        if name.startswith('grad norm:'):
            categories['Gradient Norms'].append(name)
        elif name.startswith('update norm:'):
            categories['Weight Update Norms'].append(name)
        elif name.startswith('attn logits'):
            categories['Attention Logit Magnitudes'].append(name)

    for cat_name, names in categories.items():
        n = len(names)
        n_cols = min(4, n)
        n_rows = (n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n == 1:
            axes = np.array([axes])
        axes = np.array(axes).flatten()

        steps = results['steps']
        width_colors = plt.cm.viridis(np.linspace(0, 1, len(widths)))

        for i, name in enumerate(names):
            ax = axes[i]
            for j, w in enumerate(widths):
                values = detailed[w].get(name, [])
                if values:
                    ax.plot(range(len(values)), values, color=width_colors[j],
                            linewidth=1.5, label=f'w={w}' if i == 0 else None)
            ax.set_title(name.split(': ', 1)[-1] if ': ' in name else name, fontsize=8)
            ax.set_xlabel('Step')
            ax.set_ylabel('Norm')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        for i in range(n, len(axes)):
            axes[i].set_visible(False)

        axes[0].legend(fontsize=7, loc='best')
        param_type = "muP" if config.use_mup else "SP"
        fig.suptitle(f'{cat_name} ({param_type})', fontsize=14)
        plt.tight_layout()

        if save_path:
            cat_slug = cat_name.lower().replace(' ', '_')
            path = save_path.replace('.png', f'_{cat_slug}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved {cat_name} plot to {path}")

        plt.show()


def compute_width_dependence(results: Dict) -> Dict[str, float]:
    """Compute how much activations scale with width (slope on log-log plot)."""
    widths = np.array(results['widths'])
    log_widths = np.log2(widths)
    final_step = len(results['steps']) - 1

    slopes = {}
    for layer in results['stats'][widths[0]].keys():
        values = [results['stats'][w][layer][final_step] for w in widths]
        log_values = np.log2(np.array(values) + 1e-10)
        slope, _ = np.polyfit(log_widths, log_values, 1)
        slopes[layer] = slope

    return slopes


def main():
    parser = argparse.ArgumentParser(description='muP Coordinate Check')
    parser.add_argument('--widths', type=str, default='128,256,512,1024',
                        help='Comma-separated list of widths to test')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--seq-len', type=int, default=128,
                        help='Sequence length')
    parser.add_argument('--n-layer', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--use-mup', action='store_true',
                        help='Use muP learning rate scaling')
    parser.add_argument('--base-width', type=int, default=128,
                        help='Base width for muP scaling')
    parser.add_argument('--compare', action='store_true',
                        help='Run both SP and muP and compare')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--detailed', action='store_true',
                        help='Record detailed diagnostics: gradient norms, weight update norms, '
                             'attention logit magnitudes')
    parser.add_argument('--muon-lr-exponent', type=float, default=0.0,
                        help='Muon LR exponent for muP: 1.0 = (base/width)^1 (standard muP), '
                             '0.5 = (base/width)^0.5 (for Frobenius-normalizing optimizers, '
                             'see Yang et al. Section C.1)')

    args = parser.parse_args()

    # Parse widths
    widths = [int(w) for w in args.widths.split(',')]

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load a single batch of real training data (reused every step)
    x, y, vocab_size = load_batch(args.batch_size, args.seq_len, device)

    # Create config
    config = CoordCheckConfig(
        widths=widths,
        num_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        seed=args.seed,
        use_mup=args.use_mup,
        base_width=args.base_width,
        detailed=args.detailed,
        muon_lr_exponent=args.muon_lr_exponent,
    )

    if args.compare:
        # Run both SP and muP
        print("\n" + "="*60)
        print("Running Standard Parameterization (SP)")
        print("="*60)
        config.use_mup = False
        results_sp = run_coord_check(config, device, x, y)

        print("\n" + "="*60)
        print("Running muP")
        if config.muon_lr_exponent != 1.0:
            print(f"  (Muon LR exponent: {config.muon_lr_exponent})")
        print("="*60)
        config.use_mup = True
        results_mup = run_coord_check(config, device, x, y)

        # Compute slopes
        print("\n" + "="*60)
        print("Width Dependence (slope on log-log plot)")
        print("Expected: ~0 for width-independent, positive = grows with width")
        print("="*60)

        slopes_sp = compute_width_dependence(results_sp)
        slopes_mup = compute_width_dependence(results_mup)

        print(f"\n{'Layer':<20} {'SP Slope':>12} {'muP Slope':>12}")
        print("-"*46)
        for layer in slopes_sp:
            print(f"{layer:<20} {slopes_sp[layer]:>12.4f} {slopes_mup[layer]:>12.4f}")

        # Plot comparison
        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, 'coord_check_comparison.png')
        plot_comparison(results_sp, results_mup, config, save_path)

        # Plot detailed diagnostics if requested
        if config.detailed:
            for results, label in [(results_sp, 'SP'), (results_mup, 'muP')]:
                config.use_mup = (label == 'muP')
                detail_save = None
                if args.save_dir:
                    detail_save = os.path.join(args.save_dir, f'detailed_{label.lower()}.png')
                plot_detailed(results, config, detail_save)

    else:
        # Run single mode
        param_type = "muP" if config.use_mup else "SP"
        print(f"\n{'='*60}")
        print(f"Running Coordinate Check ({param_type})")
        print(f"{'='*60}")
        print(f"Widths: {widths}")
        print(f"Steps: {config.num_steps}")
        print(f"Base width: {config.base_width}")
        if config.use_mup and config.muon_lr_exponent != 1.0:
            print(f"Muon LR exponent: {config.muon_lr_exponent}")

        results = run_coord_check(config, device, x, y)

        # Compute slopes
        slopes = compute_width_dependence(results)
        print("\n" + "="*60)
        print("Width Dependence (slope on log-log plot)")
        print("Expected for muP: ~0 (width-independent)")
        print("="*60)
        for layer, slope in slopes.items():
            status = "OK" if abs(slope) < 0.1 else "WARN"
            print(f"  {layer}: {slope:+.4f} [{status}]")

        # Loss curve analysis
        final_losses = [results['losses'][w][-1] for w in results['widths']]
        loss_spread = max(final_losses) - min(final_losses)
        print(f"\nFinal loss spread across widths: {loss_spread:.4f}")
        print(f"Expected for muP: low spread (similar losses across widths)")

        # Plot activations
        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f'coord_check_{param_type.lower()}.png')
        plot_coord_check(results, config, save_path)

        # Plot loss curves
        loss_save_path = None
        if args.save_dir:
            loss_save_path = os.path.join(args.save_dir, f'loss_curves_{param_type.lower()}.png')
        plot_loss_curves(results, config, title=param_type, save_path=loss_save_path)

        # Plot detailed diagnostics if requested
        if config.detailed:
            detail_save = None
            if args.save_dir:
                detail_save = os.path.join(args.save_dir, f'detailed_{param_type.lower()}.png')
            plot_detailed(results, config, detail_save)


if __name__ == '__main__':
    main()
