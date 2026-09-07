"""
nanochat/ is the load-bearing code: what a model is and how it trains. It must never
import the harness around it (harness/, evals/, scripts/); those import it.

python -m pytest tests/test_layering.py -v
"""

import os
import re
import glob


def test_nanochat_does_not_import_the_harness():
    root = os.path.join(os.path.dirname(__file__), "..")
    forbidden = re.compile(r"^\s*(?:from|import) (?:harness|evals|scripts)\b", re.M)
    offenders = []
    for path in sorted(glob.glob(os.path.join(root, "nanochat", "*.py"))):
        if forbidden.search(open(path).read()):
            offenders.append(os.path.basename(path))
    assert offenders == [], f"nanochat/ imports the harness: {offenders}"
