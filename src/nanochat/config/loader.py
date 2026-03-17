"""ConfigLoader: builds Config from TOML + CLI args with a single active section."""

from __future__ import annotations

import argparse
import tomllib
from argparse import Namespace
from pathlib import Path
from typing import Any

from nanochat.common import get_default_base_dir
from nanochat.config.common import CommonConfig
from nanochat.config.config import SECTION_CLS, Config


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

    def add_tokenizer(self) -> ConfigLoader:
        return self._add_section("tokenizer")

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
        base_dir: str | None = cli.get("base_dir")

        if base_dir is not None:
            candidate = Path(base_dir) / "config.toml"
            if candidate.exists():
                toml_path = candidate
        if cli.get("config") is not None:
            toml_path = Path(cli["config"])
            base_dir = str(toml_path.parent)
        if base_dir is None:
            base_dir = get_default_base_dir()
            candidate = Path(base_dir) / "config.toml"
            if candidate.exists():
                toml_path = candidate
            else:
                raise RuntimeError(
                    "Could not determine config.toml path. Please specify with --config, --base-dir or set NANOCHAT_BASE_DIR env var."
                )

        toml_data: dict[str, Any] = {}
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
