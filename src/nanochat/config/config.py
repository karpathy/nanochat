"""Top-level Config dataclass and SECTION_CLS registry mapping section names to their types."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import tomli_w

from nanochat.config.common import CommonConfig
from nanochat.config.evaluation import EvaluationConfig
from nanochat.config.rl import RLConfig
from nanochat.config.sft import SFTConfig
from nanochat.config.tokenizer import TokenizerConfig
from nanochat.config.training import TrainingConfig

SECTION_CLS: dict[str, type] = {
    "common": CommonConfig,
    "training": TrainingConfig,
    "sft": SFTConfig,
    "rl": RLConfig,
    "evaluation": EvaluationConfig,
    "tokenizer": TokenizerConfig,
}


@dataclass
class Config:
    common: CommonConfig = field(default_factory=CommonConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)

    def save(self, path: Path) -> None:
        data = {k: {fk: fv for fk, fv in v.items() if fv is not None} for k, v in asdict(self).items()}
        with open(path, "wb") as f:
            tomli_w.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> Config:
        import tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)
        cfg = cls()
        for section, values in data.items():
            section_cls = SECTION_CLS.get(section)
            if section_cls is None:
                raise ValueError(f"Unknown config section: {section!r}")
            setattr(cfg, section, section_cls(**values))
        return cfg

    @classmethod
    def generate_default(cls) -> str:
        return (
            "[common]\n" + CommonConfig.generate_default() + "\n"
            "[training]\n" + TrainingConfig.generate_default() + "\n"
            "[sft]\n" + SFTConfig.generate_default() + "\n"
            "[rl]\n" + RLConfig.generate_default() + "\n"
            "[evaluation]\n" + EvaluationConfig.generate_default() + "\n"
            "[tokenizer]\n" + TokenizerConfig.generate_default()
        )
