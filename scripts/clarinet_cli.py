"""
Clarinet interactive chat CLI — wraps scripts.chat_cli with the dual-pass
IV engine and adds CLI flags for the guidance weight and scale:

    python -m scripts.clarinet_cli --iv-weight 2.0 --wald-scale 1.0

To enable the L1 content-adaptive scale schedule (spend guidance budget where
the source marker actually moves the prediction):

    python -m scripts.clarinet_cli --iv-weight 2.0 --scale-lo 0.5 --scale-hi 2.0

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
                     help="Base scale factor s. s=1 matches vanilla CFG.")
    pre.add_argument("--scale-lo", type=float, default=1.0,
                     help="L1-adaptive scale floor (applied at zero cond/uncond "
                          "divergence). scale-lo == scale-hi == 1.0 -> constant CFG.")
    pre.add_argument("--scale-hi", type=float, default=1.0,
                     help="L1-adaptive scale ceiling (applied at max divergence). "
                          "Try --scale-lo 0.5 --scale-hi 2.0 to enable adaptation.")
    args, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


def _install_clarinet_engine(iv_weight, wald_scale, scale_lo, scale_hi):
    class _BoundClarinetEngine(ClarinetEngine):
        def generate(self, *args, **kwargs):
            kwargs.setdefault("iv_weight", iv_weight)
            kwargs.setdefault("wald_scale", wald_scale)
            kwargs.setdefault("scale_lo", scale_lo)
            kwargs.setdefault("scale_hi", scale_hi)
            yield from super().generate(*args, **kwargs)

    nanochat.engine.Engine = _BoundClarinetEngine


if __name__ == "__main__":
    cli_args = _parse_and_strip_clarinet_args()
    _install_clarinet_engine(cli_args.iv_weight, cli_args.wald_scale,
                             cli_args.scale_lo, cli_args.scale_hi)
    import scripts.chat_cli  # noqa: F401
