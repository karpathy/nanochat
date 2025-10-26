"""Agent definition for Q&A validation."""

from src.synth_data_pipeline.agents.base import build_google_agent, load_prompt_template
from src.synth_data_pipeline.config import APIConfig
from src.synth_data_pipeline.models import QAValidation

PROMPT_NAME = "qa_validator"
SYSTEM_PROMPT = "You are an expert validator ensuring high-quality training data."


def build_agent(config: APIConfig, *, api_key: str | None = None):
    """Return a configured Q&A validation agent."""
    return build_google_agent(
        config,
        system_prompt=SYSTEM_PROMPT,
        output_type=QAValidation,
        api_key=api_key,
    )


def get_prompt_template():
    """Load the Q&A validation prompt template."""
    return load_prompt_template(PROMPT_NAME)
