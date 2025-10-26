"""Agent definition for Q&A extraction."""

from src.synth_data_pipeline.agents.base import build_google_agent, load_prompt_template
from src.synth_data_pipeline.config import APIConfig
from src.synth_data_pipeline.models import QAPairBatch

PROMPT_NAME = "qa_extractor"
SYSTEM_PROMPT = (
    "You are an expert at creating high-quality, diverse Q&A pairs from documentation."
)


def build_agent(config: APIConfig, *, api_key: str | None = None):
    """Return a configured Q&A extraction agent."""
    return build_google_agent(
        config,
        system_prompt=SYSTEM_PROMPT,
        output_type=QAPairBatch,
        api_key=api_key,
    )


def get_prompt_template():
    """Load the Q&A extraction prompt template."""
    return load_prompt_template(PROMPT_NAME)
