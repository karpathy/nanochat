"""CLI handlers for `nanochat config` subcommands."""
import argparse
import tomli_w
from dataclasses import asdict
from pathlib import Path

from nanochat.config.config import Config
from nanochat.config.loader import ConfigLoader

def config_init(args: argparse.Namespace) -> None:
    out = Path(args.output)
    if out.exists():
        raise SystemExit(f"error: {out} already exists")
    out.write_text(Config.generate_default(), encoding="utf-8")
    print(f"wrote {out}")


def config_show(args: argparse.Namespace) -> None:
    cfg = ConfigLoader().resolve(args)
    print(tomli_w.dumps(asdict(cfg)).rstrip())
