"""Wandb integration: stubs and unified init helper."""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from nanochat.config import CommonConfig


class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures."""

    def __init__(self):
        pass

    def log(self, *args: object, **kwargs: object) -> None:
        pass

    def finish(self):
        pass


class LocalWandb(DummyWandb):
    """Logs metrics to a JSONL file for offline runs."""

    def __init__(self, base_dir: str, run_name: str, project: str = "nanochat"):
        self.project = project
        self.run_name = run_name
        log_dir = Path(base_dir) / "runs" / project / run_name
        os.makedirs(log_dir, exist_ok=True)
        self._f = open(log_dir / "wandb.jsonl", "a")

    def log(self, data: dict, *args: object, **kwargs: object) -> None:
        self._f.write(json.dumps(data) + "\n")
        self._f.flush()

    def finish(self):
        self._f.close()


def init_wandb(
    config: "CommonConfig",
    user_config: dict,
    master_process: bool,
    project_suffix: str | None = None,
) -> Union[DummyWandb, LocalWandb, object]:
    """Unified wandb initialisation.

    Resolution order:
    1. Non-master ranks always get DummyWandb (no logging from worker processes).
    2. wandb="disabled", WANDB_MODE=disabled env var, or run="dummy" legacy magic → DummyWandb.
    3. wandb="local" → LocalWandb (JSONL file under <base_dir>/runs/<run>/).
    4. wandb="online" → wandb.init(...) with project and run name from config.
    """
    if not master_process:
        return DummyWandb()

    mode = config.wandb
    # honour legacy --run=dummy magic and WANDB_MODE env var
    if config.run == "dummy" or os.environ.get("WANDB_MODE") == "disabled":
        mode = "disabled"

    project = config.wandb_project if project_suffix is None else f"{config.wandb_project}-{project_suffix}"

    match mode:
        case "disabled":
            return DummyWandb()
        case "local":
            assert config.base_dir is not None, "base_dir must be set for local wandb mode"
            return LocalWandb(base_dir=config.base_dir, run_name=config.run, project=project)
        case "online":
            import wandb  # type: ignore[import-untyped]

            return wandb.init(project=project, name=config.run, config=user_config)
        case _:
            raise ValueError(f"Unknown wandb mode: {mode!r}. Expected: online | local | disabled")
