"""
Shared run knobs (SSOT).

Minimal stub: only the constants nanochat/memory.py needs. The full version of this module
(profiles, argparse defaults, shell emission) is owned by the defaults stream of this slice
and wins on merge - do not grow this file here.
"""
import os

VOCAB_SIZE = 32768
DEFAULT_SEQ_LEN = 2048

# Fraction of available device memory a training step is allowed to be estimated at.
# 0.6 leaves room for fragmentation, compile caches and the OS on unified memory.
MEMORY_FIT_MARGIN = float(os.environ.get("NANOCHAT_MEMORY_FIT_MARGIN", 0.6))

# Extra full-size copies of the logits tensor to budget for on top of the logits themselves.
MEMORY_ACTIVATION_FUDGE = 1.5
