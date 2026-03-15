"""Wandb stubs for when wandb logging is disabled."""

import json
import os
from pathlib import Path


class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures."""

    def __init__(self):
        pass

    def log(self, *args: object, **kwargs: object) -> None:
        pass

    def finish(self):
        pass


class LocalWandb:
    """Logs metrics to a JSONL file for offline runs."""

    def __init__(self, run_name: str, base_dir: str | None = None):
        from nanochat.common.io import get_base_dir
        root = base_dir if base_dir else get_base_dir()
        log_dir = Path(root) / "runs" / run_name
        os.makedirs(log_dir, exist_ok=True)
        self._f = open(log_dir / "wandb.jsonl", "a")

    def log(self, data: dict, *args: object, **kwargs: object) -> None:
        self._f.write(json.dumps(data) + "\n")
        self._f.flush()

    def finish(self):
        self._f.close()
