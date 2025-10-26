"""Shared helpers for agent construction and prompt loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Type

import textprompts
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

from src.synth_data_pipeline.config import APIConfig

PROMPTS_DIR = Path(__file__).with_name("prompts")


def get_prompt_path(name: str) -> Path:
    """Return the absolute path to an agent prompt file."""
    return PROMPTS_DIR / f"{name}.txt"


def load_prompt_template(name: str):
    """Load a textprompts template by agent name."""
    return textprompts.load_prompt(str(get_prompt_path(name)))


def build_google_agent(
    api_config: APIConfig,
    *,
    system_prompt: str,
    output_type: Type,
    api_key: str | None = None,
) -> Agent:
    """Construct a Google Gemini-backed agent for a stage."""
    api_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")

    model_settings = GoogleModelSettings(
        gemini_thinking_config={"thinking_budget": api_config.thinking_budget},
        temperature=api_config.temperature,
        timeout=api_config.timeout,
    )

    model = GoogleModel(
        api_config.model,
        provider=GoogleProvider(api_key=api_key),
    )

    return Agent(
        model,
        system_prompt=system_prompt,
        model_settings=model_settings,
        output_type=output_type,
    )
