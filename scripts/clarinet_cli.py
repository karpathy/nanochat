"""
Clarinet interactive chat CLI — wraps scripts.chat_cli with the dual-pass
IV engine and adds CLI flags for the guidance weight and Wald scale:

    python -m scripts.clarinet_cli --iv-weight 2.0 --wald-scale 1.0

All other flags (--prompt, --temperature, --top-k, --model-tag, etc.) are
forwarded to the upstream chat_cli unchanged.
"""

import argparse
import sys

import nanochat.engine

from clarinet.engine import ClarinetEngine


def _parse_and_strip_clarinet_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--iv-weight", type=float, default=1.5,
                     help="IV guidance weight w. w=0 -> unconditional; w=1 -> conditional only; w>1 -> guided.")
    pre.add_argument("--wald-scale", type=float, default=1.0,
                     help="Wald-style scale factor s. s=1 matches vanilla CFG.")
    args, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def _install_clarinet_engine(iv_weight, wald_scale):
    class _BoundClarinetEngine(ClarinetEngine):
        def generate(self, *args, **kwargs):
            kwargs.setdefault("iv_weight", iv_weight)
            kwargs.setdefault("wald_scale", wald_scale)
            yield from super().generate(*args, **kwargs)

    nanochat.engine.Engine = _BoundClarinetEngine


if __name__ == "__main__":
    cli_args = _parse_and_strip_clarinet_args()
    _install_clarinet_engine(cli_args.iv_weight, cli_args.wald_scale)
    import scripts.chat_cli  # noqa: F401
