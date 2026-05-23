"""
Clarinet pretraining entry point.

Wraps scripts.base_train by substituting clarinet's source-marker-aware
dataloader in place of the upstream BOS-bestfit one. Two clarinet-specific
flags are parsed first and stripped from sys.argv so base_train's argparse
doesn't see them; everything else passes through:

    torchrun --nproc_per_node=8 -m scripts.clarinet_train \\
        --depth=20 --reasoning-mix-ratio=0.3 --p-uncond=0.1

This shim keeps scripts/base_train.py and nanochat/ entirely untouched so
upstream rebases stay clean.
"""

import argparse
import sys

import nanochat.dataloader as _upstream_dataloader

from clarinet.dataloader import clarinet_data_loader


def _parse_and_strip_clarinet_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--reasoning-mix-ratio", type=float, default=0.3,
                     help="Fraction of training docs sampled from proof-pile-2 (vs climbmix).")
    pre.add_argument("--p-uncond", type=float, default=0.1,
                     help="Probability of overriding the true source marker with <|src_unknown|> during training.")
    pre.add_argument("--clarinet-seed", type=int, default=0,
                     help="Seed for the per-doc p_uncond dropout RNG (seeded per rank).")
    clarinet_args, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return clarinet_args


def _install_clarinet_dataloader(clarinet_args):
    def train_loader(tokenizer, B, T, split, **kwargs):
        return clarinet_data_loader(
            tokenizer, B, T, split,
            reasoning_mix_ratio=clarinet_args.reasoning_mix_ratio,
            p_uncond=clarinet_args.p_uncond,
            seed=clarinet_args.clarinet_seed,
            **kwargs,
        )

    def val_loader(tokenizer, B, T, split, **kwargs):
        # Eval is deterministic — always use the true marker (no dropout) and
        # drop the resume state_dict from the yields to match the upstream
        # non-stateful loader's contract.
        for inputs, targets, _state in clarinet_data_loader(
            tokenizer, B, T, split,
            reasoning_mix_ratio=clarinet_args.reasoning_mix_ratio,
            p_uncond=0.0,
            seed=clarinet_args.clarinet_seed,
            **kwargs,
        ):
            yield inputs, targets

    _upstream_dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit = train_loader
    _upstream_dataloader.tokenizing_distributed_data_loader_bos_bestfit = val_loader


if __name__ == "__main__":
    clarinet_args = _parse_and_strip_clarinet_args()
    _install_clarinet_dataloader(clarinet_args)
    # base_train.py runs at import time (argparse + training loop are
    # module-level), so importing it after the patch is what kicks off training.
    import scripts.base_train  # noqa: F401
