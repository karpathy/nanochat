"""Agent definition for conversation generation."""

from src.synth_data_pipeline.agents.base import build_google_agent, load_prompt_template
from src.synth_data_pipeline.config import APIConfig
from src.synth_data_pipeline.models import Conversation

PROMPT_NAME = "conversation_generator"
SYSTEM_PROMPT = "You are an expert at creating natural, realistic conversations."


def build_agent(config: APIConfig, *, api_key: str | None = None):
    """Return a configured conversation generation agent."""
    return build_google_agent(
        config,
        system_prompt=SYSTEM_PROMPT,
        output_type=Conversation,
        api_key=api_key,
    )


def get_prompt_template():
    """Load the conversation generation prompt template."""
    return load_prompt_template(PROMPT_NAME)
