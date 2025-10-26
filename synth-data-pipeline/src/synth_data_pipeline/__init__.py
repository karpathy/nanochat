"""Synthetic Data Pipeline for NanoChat Fine-Tuning."""

import textprompts

# Core models
from .models import (
    QAPair,
    QAPairBatch,
    QAValidation,
    ValidatedQAPair,
    Message,
    Conversation,
    ConversationMetadata,
    JudgmentScore,
    JudgedConversation,
    EmbeddedConversation,
    UniqueConversation,
    NanoChatMessage,
    NanoChatConversation,
)

# Configuration
from .config import (
    APIConfig,
    FilePaths,
    PipelineParams,
    Persona,
    SystemPromptTemplate,
    PATHS,
    FULL_PARAMS,
    STAGE_CONFIGS,
)

# Utilities
from .utils import (
    load_jsonl,
    save_jsonl,
    parse_markdown_chunks,
    process_with_concurrency,
    calculate_overall_score,
    print_sample,
    print_statistics,
)

# Agents (as submodule)
from . import agents

__version__ = "0.1.0"

# Set strict metadata requirement for all prompts globally
textprompts.set_metadata("strict")

__all__ = [
    # Version
    "__version__",
    # Models
    "QAPair",
    "QAPairBatch",
    "QAValidation",
    "ValidatedQAPair",
    "Message",
    "Conversation",
    "ConversationMetadata",
    "JudgmentScore",
    "JudgedConversation",
    "EmbeddedConversation",
    "UniqueConversation",
    "NanoChatMessage",
    "NanoChatConversation",
    # Config classes
    "APIConfig",
    "FilePaths",
    "PipelineParams",
    "Persona",
    "SystemPromptTemplate",
    # Config instances
    "PATHS",
    "FULL_PARAMS",
    "STAGE_CONFIGS",
    # Utils
    "load_jsonl",
    "save_jsonl",
    "parse_markdown_chunks",
    "process_with_concurrency",
    "calculate_overall_score",
    "print_sample",
    "print_statistics",
    # Submodules
    "agents",
]
