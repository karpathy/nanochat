"""Agent definition for conversation quality judging."""

from src.synth_data_pipeline.agents.base import build_google_agent, load_prompt_template
from src.synth_data_pipeline.config import APIConfig
from src.synth_data_pipeline.models import JudgmentScore

PROMPT_NAME = "conversation_judge"
SYSTEM_PROMPT = "You are an expert evaluator of training data quality for language models."


def build_agent(config: APIConfig, *, api_key: str | None = None):
    """Return a configured judging agent."""
    return build_google_agent(
        config,
        system_prompt=SYSTEM_PROMPT,
        output_type=JudgmentScore,
        api_key=api_key,
    )


def get_prompt_template():
    """Load the conversation judging prompt template."""
    return load_prompt_template(PROMPT_NAME)
