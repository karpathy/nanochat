"""ConfigLoader: builds Config from TOML + CLI args with a single active section."""
from __future__ import annotations

import argparse
import tomllib
from argparse import Namespace
from pathlib import Path
import os
from nanochat.config.common import CommonConfig
from nanochat.config.config import Config, SECTION_CLS


def _get_base_dir() -> str:
    """Return the nanochat base directory: --base-dir CLI > NANOCHAT_BASE_DIR env > ~/.cache/nanochat."""
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ["NANOCHAT_BASE_DIR"]
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

class ConfigLoader:

    def __init__(self) -> None:
        self._sections: set[str] = {"common"}

    def _add_section(self, name: str) -> ConfigLoader:
        if self._sections - {"common"}:
            raise RuntimeError("ConfigLoader only supports one section per instance")
        self._sections.add(name)
        return self

    def add_training(self) -> ConfigLoader:
        return self._add_section("training")

    def add_sft(self) -> ConfigLoader:
        return self._add_section("sft")

    def add_rl(self) -> ConfigLoader:
        return self._add_section("rl")

    def add_evaluation(self) -> ConfigLoader:
        return self._add_section("evaluation")

    def parse(self, args: list[str] | None = None) -> Config:
        """Build a parser, parse args, and resolve. Used by tests; production path goes through resolve()."""
        parser = argparse.ArgumentParser()
        CommonConfig.update_parser(parser)
        for section in self._sections - {"common"}:
            SECTION_CLS[section].update_parser(parser)
        return self.resolve(parser.parse_args(args))

    def resolve(self, ns: Namespace) -> Config:
        """Resolve Config from an already-parsed argparse Namespace + optional TOML."""
        
        cli = vars(ns)

        toml_path: Path | None = None
        base_dir :str | None = cli.get("base_dir")

        if base_dir is not None:
            candidate = Path(base_dir) / "config.toml"
            if candidate.exists():
                toml_path = candidate
        if cli.get("config") is not None:
            toml_path = Path(cli["config"])
            base_dir = str(toml_path.parent)
        if base_dir is None:
            base_dir = _get_base_dir()
            candidate = Path(base_dir) / "config.toml"
            if candidate.exists():
                toml_path = candidate
            else:
                raise RuntimeError(f"Could not determine config.toml path. Please specify with --config, --base-dir or set NANOCHAT_BASE_DIR env var.")

        toml_data: dict = {}
        if toml_path is not None:
            with open(toml_path, "rb") as f:
                toml_data = tomllib.load(f)

        cfg = Config()
        for section in self._sections:
            cls = SECTION_CLS[section]
            valid = cls.__dataclass_fields__
            merged = {**toml_data.get(section, {}), **{k: v for k, v in cli.items() if k in valid}}
            setattr(cfg, section, cls(**merged))
        
        if cfg.common.base_dir is None:
            cfg.common.base_dir = base_dir

        return cfg


