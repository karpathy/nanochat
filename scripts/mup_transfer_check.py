"""
muP Transfer Check for nanochat

Validates that optimal learning rates transfer across model widths under muP.
For each width, sweeps over LR multipliers and records final loss. Under correct
muP, the optimal LR multiplier should be ~1.0 at all widths (i.e., the same LR
works everywhere). Under SP, the optimal LR typically shifts with width.

Reference: https://blog.eleuther.ai/mutransfer/
Reference: Yang et al., "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot
           Hyperparameter Transfer" (arXiv:2203.03466), Section F.

Usage:
    # Quick check (~2 min on GPU)
    python -m scripts.mup_transfer_check

    # Compare SP vs muP side-by-side
    python -m scripts.mup_transfer_check --compare

    # Wide LR sweep (paper-style, ~1000x range)
    python -m scripts.mup_transfer_check --compare --widths 128,256,512,1024 --steps 200

    # Random log-uniform LR trials (paper-style methodology)
    python -m scripts.mup_transfer_check --compare --num-random-trials 20

    # Multi-HP sweep (init scale + output multiplier)
    python -m scripts.mup_transfer_check --compare --sweep-init-scale --sweep-output-mult

    # Save plots
    python -m scripts.mup_transfer_check --compare --save-dir plots/
"""

import argparse
import os
os.environ["NANOCHAT_DTYPE"] = "float32"
import torch
import torch._dynamo
torch._dynamo.config.disable = True
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os

from nanochat.gpt import GPT, GPTConfig


@dataclass
class TransferCheckConfig:
    widths: List[int]
    lr_multipliers: List[float]
    num_steps: int = 200
    batch_size: int = 8
    seq_len: int = 128
    vocab_size: int = 32768
    n_layer: int = 2
    seed: int = 42
    use_mup: bool = False
    base_width: int = 128
    # Base learning rates (tuned at base_width=128)
    matrix_lr: float = 0.12
    embedding_lr: float = 6.0
    unembedding_lr: float = 0.12
    # Multi-HP sweeps
    sweep_init_scale: bool = False
    sweep_output_mult: bool = False
    # Data diversity
    num_batches: int = 1
    # Muon LR exponent for muP (1.0=standard, 0.5=Frobenius-norm optimizers)
    muon_lr_exponent: float = 0.0
    # Sweep mode: which optimizer groups the LR multiplier applies to
    # "all" = multiply all LRs (default), "muon-only" = only matrix_lr,
    # "adamw-only" = only embedding_lr/unembedding_lr
    sweep_mode: str = "all"


def load_batches(num_batches: int, batch_size: int, seq_len: int, device: torch.device):
    """Load multiple batches from the nanochat training pipeline.
    Falls back to random data if the tokenizer/dataset isn't available."""
    try:
        from nanochat.tokenizer import get_tokenizer
        from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
        tokenizer = get_tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        loader = tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, batch_size, seq_len, split="train", device=device,
        )
        batches = []
        for i, (x, y) in enumerate(loader):
            batches.append((x, y))
            if len(batches) >= num_batches:
                break
        print(f"Loaded {len(batches)} real training batch(es) (vocab_size={vocab_size})")
        return batches, vocab_size
    except Exception as e:
        print(f"Could not load training data ({e}), using random tokens")
        vocab_size = 32768
        batches = []
        for i in range(num_batches):
            rng = torch.Generator(device=device)
            rng.manual_seed(42 + i)
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, generator=rng)
            y = torch.roll(x, -1, dims=1)
            y[:, -1] = -1
            batches.append((x, y))
        return batches, vocab_size


def create_model(width: int, config: TransferCheckConfig, device: torch.device,
                 mup_base_width: int = 0, init_scale: float = 1.0,
                 output_mult: float = 1.0):
    """Create a model with the specified width and optional HP overrides."""
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

    # Apply init_scale: multiply all parameter inits by scalar
    if init_scale != 1.0:
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(init_scale)

    return model, gpt_config


def train_model(width: int, lr_mult: float, config: TransferCheckConfig,
                device: torch.device, batches: List,
                init_scale: float = 1.0, output_mult: float = 1.0):
    """Train a model at given width and LR multiplier, return loss history."""
    torch.manual_seed(config.seed)

    mup_base_width = config.base_width if config.use_mup else 0
    model, gpt_config = create_model(width, config, device, mup_base_width=mup_base_width,
                                      init_scale=init_scale, output_mult=output_mult)
    actual_width = gpt_config.n_embd

    # Scale the learning rates by the multiplier, respecting sweep_mode
    if config.sweep_mode == "muon-only":
        muon_mult, adamw_mult = lr_mult, 1.0
    elif config.sweep_mode == "adamw-only":
        muon_mult, adamw_mult = 1.0, lr_mult
    else:  # "all"
        muon_mult, adamw_mult = lr_mult, lr_mult

    optimizer = model.setup_optimizer(
        unembedding_lr=config.unembedding_lr * adamw_mult,
        embedding_lr=config.embedding_lr * adamw_mult,
        matrix_lr=config.matrix_lr * muon_mult,
        weight_decay=0.0,
        use_mup=config.use_mup,
        base_width=config.base_width,
        muon_lr_exponent=config.muon_lr_exponent,
    )

    model.train()
    losses = []
    num_batches = len(batches)

    for step in range(config.num_steps):
        x, y = batches[step % num_batches]
        with torch.amp.autocast(device_type='cuda', dtype=torch.float32, enabled=False):
            loss = model(x, y)

        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return losses, actual_width


def run_transfer_check(config: TransferCheckConfig, device: torch.device,
                       batches: List) -> Dict:
    """Run LR sweep across all widths."""
    results = {
        'widths': [],
        'lr_multipliers': config.lr_multipliers,
        'losses': {},  # losses[(width, lr_mult)] = [loss_step0, ...]
        'final_losses': defaultdict(dict),  # final_losses[width][lr_mult] = final_loss
    }

    for width in config.widths:
        actual_width = None
        for lr_mult in config.lr_multipliers:
            print(f"  width={width}, lr_mult={lr_mult:.4f}...", end=" ", flush=True)

            losses, actual_width = train_model(width, lr_mult, config, device, batches)
            results['losses'][(actual_width, lr_mult)] = losses
            results['final_losses'][actual_width][lr_mult] = losses[-1]
            print(f"final_loss={losses[-1]:.4f}")

        if actual_width not in results['widths']:
            results['widths'].append(actual_width)

    return results


def run_hp_sweep(config: TransferCheckConfig, device: torch.device,
                 batches: List, hp_name: str, hp_values: List[float]) -> Dict:
    """Run a sweep over a single HP (init_scale or output_mult) at fixed LR."""
    results = {
        'widths': [],
        'hp_values': hp_values,
        'hp_name': hp_name,
        'final_losses': defaultdict(dict),
    }

    for width in config.widths:
        actual_width = None
        for hp_val in hp_values:
            init_scale = hp_val if hp_name == 'init_scale' else 1.0
            output_mult = hp_val if hp_name == 'output_mult' else 1.0
            print(f"  width={width}, {hp_name}={hp_val:.4f}...", end=" ", flush=True)

            losses, actual_width = train_model(
                width, 1.0, config, device, batches,
                init_scale=init_scale, output_mult=output_mult)
            results['final_losses'][actual_width][hp_val] = losses[-1]
            print(f"final_loss={losses[-1]:.4f}")

        if actual_width not in results['widths']:
            results['widths'].append(actual_width)

    return results


def find_optimal_lr(final_losses: Dict[float, float]) -> float:
    """Find the LR multiplier with the lowest final loss."""
    return min(final_losses, key=final_losses.get)


def plot_lr_sweep(results: Dict, config: TransferCheckConfig, title: str = "", save_path: Optional[str] = None):
    """Plot LR sweep: final loss vs LR multiplier for each width."""
    widths = results['widths']
    lr_mults = results['lr_multipliers']
    final_losses = results['final_losses']

    n_cols = 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(widths)))

    # Left: final loss vs LR multiplier
    ax = axes[0]
    for i, w in enumerate(widths):
        losses = [final_losses[w][m] for m in lr_mults]
        ax.plot(lr_mults, losses, 'o-', color=colors[i], linewidth=2, label=f'width={w}')
        opt_mult = find_optimal_lr(final_losses[w])
        opt_loss = final_losses[w][opt_mult]
        ax.plot(opt_mult, opt_loss, '*', color=colors[i], markersize=15, zorder=5)

    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('LR Multiplier')
    ax.set_ylabel('Final Loss')
    ax.set_title(f'LR Sweep{" - " + title if title else ""}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: optimal LR multiplier vs width
    ax = axes[1]
    opt_mults = [find_optimal_lr(final_losses[w]) for w in widths]
    ax.plot(widths, opt_mults, 'o-', linewidth=2, markersize=8, color='tab:blue')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='target (1.0)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels(widths)
    ax.set_xlabel('Width')
    ax.set_ylabel('Optimal LR Multiplier')
    ax.set_title(f'Optimal LR vs Width{" - " + title if title else ""}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_comparison(results_sp: Dict, results_mup: Dict, config: TransferCheckConfig, save_path: Optional[str] = None):
    """Plot SP vs muP comparison side by side."""
    widths = results_sp['widths']
    lr_mults = results_sp['lr_multipliers']

    n_rows, n_cols = 2, 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(widths)))

    # Top row: LR sweep curves (log scale for loss detail)
    for col, (results, label) in enumerate([(results_sp, 'SP'), (results_mup, 'muP')]):
        ax = axes[0, col]
        for i, w in enumerate(widths):
            losses = [results['final_losses'][w][m] for m in lr_mults]
            ax.plot(lr_mults, losses, 'o-', color=colors[i], linewidth=2, label=f'w={w}')
            opt_mult = find_optimal_lr(results['final_losses'][w])
            opt_loss = results['final_losses'][w][opt_mult]
            ax.plot(opt_mult, opt_loss, '*', color=colors[i], markersize=15, zorder=5)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('LR Multiplier')
        ax.set_ylabel('Final Loss')
        ax.set_title(f'{label}: Final Loss vs LR Multiplier')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Shared y-axis for top row
    all_losses_flat = [results_sp['final_losses'][w][m] for m in lr_mults for w in widths] + \
                      [results_mup['final_losses'][w][m] for m in lr_mults for w in widths]
    y_min, y_max = min(all_losses_flat) * 0.9, max(all_losses_flat) * 1.1
    axes[0, 0].set_ylim(y_min, y_max)
    axes[0, 1].set_ylim(y_min, y_max)

    # Bottom left: optimal LR vs width for both
    ax = axes[1, 0]
    for results, label, color in [(results_sp, 'SP', 'tab:red'), (results_mup, 'muP', 'tab:blue')]:
        opt_mults = [find_optimal_lr(results['final_losses'][w]) for w in widths]
        ax.plot(widths, opt_mults, 'o-', linewidth=2, markersize=8, color=color, label=label)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='target')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels(widths)
    ax.set_xlabel('Width')
    ax.set_ylabel('Optimal LR Multiplier')
    ax.set_title('Optimal LR Multiplier vs Width')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom right: loss at optimal LR vs width
    ax = axes[1, 1]
    for results, label, color in [(results_sp, 'SP', 'tab:red'), (results_mup, 'muP', 'tab:blue')]:
        opt_losses = [results['final_losses'][w][find_optimal_lr(results['final_losses'][w])] for w in widths]
        ax.plot(widths, opt_losses, 'o-', linewidth=2, markersize=8, color=color, label=label)
    ax.set_xscale('log', base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels(widths)
    ax.set_xlabel('Width')
    ax.set_ylabel('Best Final Loss')
    ax.set_title('Best Achievable Loss vs Width')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('muP Transfer Check: SP vs muP', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    plt.show()


def plot_hp_sweep(results: Dict, config: TransferCheckConfig, title: str = "", save_path: Optional[str] = None):
    """Plot HP sweep: final loss vs HP value for each width."""
    widths = results['widths']
    hp_values = results['hp_values']
    hp_name = results['hp_name']
    final_losses = results['final_losses']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(widths)))

    ax = axes[0]
    for i, w in enumerate(widths):
        losses = [final_losses[w][v] for v in hp_values]
        ax.plot(hp_values, losses, 'o-', color=colors[i], linewidth=2, label=f'w={w}')
        opt_v = min(final_losses[w], key=final_losses[w].get)
        ax.plot(opt_v, final_losses[w][opt_v], '*', color=colors[i], markersize=15, zorder=5)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel(hp_name)
    ax.set_ylabel('Final Loss')
    ax.set_title(f'{hp_name} Sweep{" - " + title if title else ""}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    opt_vals = [min(final_losses[w], key=final_losses[w].get) for w in widths]
    ax.plot(widths, opt_vals, 'o-', linewidth=2, markersize=8, color='tab:blue')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='target (1.0)')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xlabel('Width')
    ax.set_ylabel(f'Optimal {hp_name}')
    ax.set_title(f'Optimal {hp_name} vs Width')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def plot_loss_curves_at_optimal(results: Dict, config: TransferCheckConfig, title: str = "", save_path: Optional[str] = None):
    """Plot full loss curves at the optimal LR for each width."""
    widths = results['widths']

    fig, ax = plt.subplots(figsize=(5 * 2, 4))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(widths)))

    for i, w in enumerate(widths):
        opt_mult = find_optimal_lr(results['final_losses'][w])
        losses = results['losses'][(w, opt_mult)]
        ax.plot(losses, color=colors[i], linewidth=2, label=f'w={w} (lr_mult={opt_mult})')

    ax.set_yscale('log')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss Curves at Optimal LR{" - " + title if title else ""}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()


def print_summary(results: Dict, label: str):
    """Print a summary table of the LR sweep results."""
    widths = results['widths']
    lr_mults = results['lr_multipliers']
    final_losses = results['final_losses']

    print(f"\n{'='*70}")
    print(f"  {label}: LR Sweep Results")
    print(f"{'='*70}")

    # Header
    header = f"{'Width':>8}"
    for m in lr_mults:
        header += f" | {m:>7.3f}"
    header += f" | {'Best':>7} | {'Opt LR':>7}"
    print(header)
    print("-" * len(header))

    opt_mults = []
    for w in widths:
        row = f"{w:>8}"
        for m in lr_mults:
            loss = final_losses[w][m]
            row += f" | {loss:>7.4f}"
        opt_m = find_optimal_lr(final_losses[w])
        opt_mults.append(opt_m)
        opt_loss = final_losses[w][opt_m]
        row += f" | {opt_loss:>7.4f} | {opt_m:>7.3f}"
        print(row)

    # Transfer quality metric: how much does the optimal LR shift?
    opt_mults_arr = np.array(opt_mults)
    log_opt = np.log2(opt_mults_arr)
    spread = log_opt.max() - log_opt.min()  # spread in log2 space
    print(f"\nOptimal LR spread (log2): {spread:.3f}")
    print(f"  (0 = perfect transfer, >1 = poor transfer)")


def main():
    parser = argparse.ArgumentParser(description='muP Transfer Check')
    parser.add_argument('--widths', type=str, default='128,256,512,1024',
                        help='Comma-separated list of widths to test')
    # Paper-style default: ~1000x range, 11 log-spaced points
    parser.add_argument('--lr-mults', type=str,
                        default='0.03125,0.0625,0.125,0.25,0.5,1.0,2.0,4.0,8.0,16.0,32.0',
                        help='Comma-separated LR multipliers to sweep (default: 1024x range, 11 points)')
    parser.add_argument('--num-random-trials', type=int, default=0,
                        help='If >0, use N log-uniform random LR multipliers from 10^Uniform(-1.5,1.5) '
                             'instead of the grid. Paper-style methodology (Section F).')
    parser.add_argument('--steps', type=int, default=200,
                        help='Number of training steps per run')
    parser.add_argument('--batch-size', type=int, default=8,
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
    parser.add_argument('--num-batches', type=int, default=1,
                        help='Number of data batches to cycle through (default 1 for backward compat, '
                             'recommend 8 for thorough checks)')
    # Multi-HP sweeps
    parser.add_argument('--sweep-init-scale', action='store_true',
                        help='Also sweep init scale multiplier (sampled from 10^Uniform(-1,1))')
    parser.add_argument('--sweep-output-mult', action='store_true',
                        help='Also sweep output logit multiplier (sampled from 4^Uniform(-1,1))')
    parser.add_argument('--muon-lr-exponent', type=float, default=0.0,
                        help='Muon LR exponent for muP: 1.0 = (base/width)^1 (standard), '
                             '0.5 = (base/width)^0.5 (for Frobenius-normalizing optimizers like Muon)')
    parser.add_argument('--sweep-mode', type=str, default='all',
                        choices=['all', 'muon-only', 'adamw-only'],
                        help='Which optimizer groups the LR multiplier applies to: '
                             '"all" = all LRs (default), "muon-only" = only Muon/matrix LR, '
                             '"adamw-only" = only AdamW/embedding/output LR')

    args = parser.parse_args()

    widths = [int(w) for w in args.widths.split(',')]

    # Generate LR multipliers
    if args.num_random_trials > 0:
        # Log-uniform random sampling: 10^Uniform(-1.5, 1.5)
        rng = np.random.RandomState(args.seed)
        lr_mults = sorted(10 ** rng.uniform(-1.5, 1.5, args.num_random_trials))
        lr_mults = [round(float(m), 6) for m in lr_mults]
        print(f"Using {args.num_random_trials} random log-uniform LR multipliers: "
              f"[{lr_mults[0]:.4f}, ..., {lr_mults[-1]:.4f}]")
    else:
        lr_mults = sorted(float(m) for m in args.lr_mults.split(','))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load batches
    batches, vocab_size = load_batches(args.num_batches, args.batch_size, args.seq_len, device)

    config = TransferCheckConfig(
        widths=widths,
        lr_multipliers=lr_mults,
        num_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        n_layer=args.n_layer,
        seed=args.seed,
        use_mup=args.use_mup,
        base_width=args.base_width,
        sweep_init_scale=args.sweep_init_scale,
        sweep_output_mult=args.sweep_output_mult,
        num_batches=args.num_batches,
        muon_lr_exponent=args.muon_lr_exponent,
        sweep_mode=args.sweep_mode,
    )

    if args.compare:
        # Run SP
        print("\n" + "=" * 60)
        print("Running Standard Parameterization (SP)")
        print("=" * 60)
        config.use_mup = False
        results_sp = run_transfer_check(config, device, batches)
        print_summary(results_sp, "SP")

        # Run muP
        print("\n" + "=" * 60)
        print("Running muP")
        print("=" * 60)
        config.use_mup = True
        results_mup = run_transfer_check(config, device, batches)
        print_summary(results_mup, "muP")

        # Compare
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        sp_opts = [find_optimal_lr(results_sp['final_losses'][w]) for w in results_sp['widths']]
        mup_opts = [find_optimal_lr(results_mup['final_losses'][w]) for w in results_mup['widths']]
        sp_spread = np.log2(max(sp_opts)) - np.log2(min(sp_opts))
        mup_spread = np.log2(max(mup_opts)) - np.log2(min(mup_opts))
        print(f"SP  optimal LR spread (log2): {sp_spread:.3f}")
        print(f"muP optimal LR spread (log2): {mup_spread:.3f}")
        if mup_spread < sp_spread:
            print(f"muP shows {sp_spread/max(mup_spread, 0.001):.1f}x better LR transfer!")
        else:
            print("muP does NOT show better LR transfer (check implementation)")

        # Plot
        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, 'transfer_check_comparison.png')
        plot_comparison(results_sp, results_mup, config, save_path)

        # Also plot loss curves at optimal LR
        for results, label in [(results_sp, 'SP'), (results_mup, 'muP')]:
            lc_save = None
            if args.save_dir:
                lc_save = os.path.join(args.save_dir, f'optimal_loss_curves_{label.lower()}.png')
            plot_loss_curves_at_optimal(results, config, title=label, save_path=lc_save)

        # Multi-HP sweeps (only for muP, to demonstrate transfer)
        if args.sweep_init_scale or args.sweep_output_mult:
            config.use_mup = True

        if args.sweep_init_scale:
            print("\n" + "=" * 60)
            print("muP: Init Scale Sweep")
            print("=" * 60)
            # 10^Uniform(-1, 1) => range [0.1, 10]
            init_scales = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
            init_results = run_hp_sweep(config, device, batches, 'init_scale', init_scales)
            save_hp = None
            if args.save_dir:
                save_hp = os.path.join(args.save_dir, 'init_scale_sweep.png')
            plot_hp_sweep(init_results, config, title="muP", save_path=save_hp)

        if args.sweep_output_mult:
            print("\n" + "=" * 60)
            print("muP: Output Multiplier Sweep")
            print("=" * 60)
            # 4^Uniform(-1, 1) => range [0.25, 4]
            output_mults = [0.25, 0.5, 1.0, 2.0, 4.0]
            output_results = run_hp_sweep(config, device, batches, 'output_mult', output_mults)
            save_hp = None
            if args.save_dir:
                save_hp = os.path.join(args.save_dir, 'output_mult_sweep.png')
            plot_hp_sweep(output_results, config, title="muP", save_path=save_hp)

    else:
        param_type = "muP" if config.use_mup else "SP"
        print(f"\n{'='*60}")
        print(f"Running Transfer Check ({param_type})")
        print(f"{'='*60}")
        print(f"Widths: {widths}")
        print(f"LR multipliers: {lr_mults}")
        print(f"Steps: {config.num_steps}")
        if config.sweep_mode != "all":
            print(f"Sweep mode: {config.sweep_mode}")

        results = run_transfer_check(config, device, batches)
        print_summary(results, param_type)

        # Plot LR sweep
        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f'transfer_check_{param_type.lower()}.png')
        plot_lr_sweep(results, config, title=param_type, save_path=save_path)

        # Plot loss curves at optimal LR
        lc_save = None
        if args.save_dir:
            lc_save = os.path.join(args.save_dir, f'optimal_loss_curves_{param_type.lower()}.png')
        plot_loss_curves_at_optimal(results, config, title=param_type, save_path=lc_save)


if __name__ == '__main__':
    main()
