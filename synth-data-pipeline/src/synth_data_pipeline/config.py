"""
Configuration and constants for the synthetic data pipeline.

This module centralizes all variations, tags, and configuration options
for generating diverse training data.
"""

from typing import Dict, List, Literal
from dataclasses import dataclass, field


# ============================================================================
# Difficulty Levels
# ============================================================================

DIFFICULTY_LEVELS = ["basic", "intermediate", "advanced"]

DIFFICULTY_DESCRIPTIONS = {
    "basic": "Simple factual questions requiring basic recall",
    "intermediate": "Questions requiring understanding and reasoning",
    "advanced": "Complex technical or multi-faceted questions"
}


# ============================================================================
# Conversation Styles
# ============================================================================

CONVERSATION_STYLES = ["formal", "casual", "technical"]

STYLE_DESCRIPTIONS = {
    "formal": "Professional language, complete sentences, no slang",
    "casual": "Friendly tone, can use contractions, conversational",
    "technical": "Uses technical terminology, assumes some expertise"
}


# ============================================================================
# User Personas
# ============================================================================

@dataclass
class Persona:
    """Definition of a user persona."""
    name: str
    description: str
    typical_questions: List[str] = field(default_factory=list)
    formality: Literal["formal", "casual", "neutral"] = "neutral"


PERSONAS = {
    "developer": Persona(
        name="developer",
        description="Software developer or engineer evaluating SWAP Commerce's APIs and technical implementation",
        typical_questions=[
            "API integration details",
            "Technical specifications",
            "SDK usage",
            "Error handling"
        ],
        formality="technical"
    ),
    "product_manager": Persona(
        name="product_manager",
        description="Product manager researching SWAP Commerce features, capabilities, and business value",
        typical_questions=[
            "Feature comparisons",
            "Roadmap questions",
            "Use cases",
            "ROI analysis"
        ],
        formality="formal"
    ),
    "cs_agent": Persona(
        name="cs_agent",
        description="Customer success or support agent learning about SWAP Commerce to help customers",
        typical_questions=[
            "Setup instructions",
            "Troubleshooting",
            "Configuration options",
            "Best practices"
        ],
        formality="neutral"
    ),
    "executive": Persona(
        name="executive",
        description="Business executive or decision-maker evaluating SWAP Commerce for strategic fit and ROI",
        typical_questions=[
            "Business value",
            "Competitive advantages",
            "Pricing strategy",
            "Scalability"
        ],
        formality="formal"
    ),
    "operations": Persona(
        name="operations",
        description="Operations or logistics manager interested in SWAP Commerce's operational features and integrations",
        typical_questions=[
            "Integration capabilities",
            "Workflow automation",
            "Performance metrics",
            "SLA guarantees"
        ],
        formality="neutral"
    ),
    "finance": Persona(
        name="finance",
        description="Finance or accounting professional interested in tax compliance, pricing, and financial aspects",
        typical_questions=[
            "Tax compliance",
            "Financial reporting",
            "Audit trails",
            "Cost structure"
        ],
        formality="formal"
    ),
}


# ============================================================================
# System Prompt Templates
# ============================================================================

@dataclass
class SystemPromptTemplate:
    """Definition of a system prompt template."""
    name: str
    description: str
    template: str
    verbosity: Literal["concise", "balanced", "detailed"] = "balanced"
    use_case: str = "general"


SYSTEM_PROMPT_TEMPLATES = {
    "helpful": SystemPromptTemplate(
        name="helpful",
        description="Helpful and friendly assistant",
        template="You are a helpful AI assistant with expertise in SWAP Commerce's e-commerce platform and services. You provide accurate, friendly, and detailed answers to questions about SWAP Commerce's products, features, integrations, and pricing.",
        verbosity="detailed",
        use_case="general"
    ),
    "concise": SystemPromptTemplate(
        name="concise",
        description="Brief and to-the-point responses",
        template="You are a SWAP Commerce expert providing clear, concise answers. Focus on key information without unnecessary detail.",
        verbosity="concise",
        use_case="quick_reference"
    ),
    "technical": SystemPromptTemplate(
        name="technical",
        description="Technical expert for developers",
        template="You are a technical expert on SWAP Commerce's platform. You provide detailed technical information about APIs, integrations, implementation, and system architecture. You assume the user has technical knowledge.",
        verbosity="detailed",
        use_case="developer"
    ),
    "detailed": SystemPromptTemplate(
        name="detailed",
        description="Comprehensive explanations",
        template="You are a comprehensive SWAP Commerce expert who provides thorough, well-explained answers with examples, context, and relevant details. You ensure users fully understand the topic.",
        verbosity="detailed",
        use_case="learning"
    ),
    "sales": SystemPromptTemplate(
        name="sales",
        description="Sales and solutions focused",
        template="You are a SWAP Commerce solutions consultant helping potential customers understand how SWAP Commerce can solve their e-commerce challenges. You're knowledgeable about features, benefits, and competitive advantages.",
        verbosity="balanced",
        use_case="sales"
    ),
}


# ============================================================================
# Topic Categories
# ============================================================================

TOPIC_CATEGORIES = [
    "pricing",
    "features",
    "integrations",
    "api",
    "compliance",
    "tax",
    "shipping",
    "returns",
    "tracking",
    "inventory",
    "operations",
    "company_info",
    "funding",
    "customers",
    "partnerships",
    "security",
]


# ============================================================================
# Conversation Length Distribution
# ============================================================================

# Default distribution of conversation lengths (num_turns)
DEFAULT_TURN_DISTRIBUTION = {
    1: 0.20,  # 20% single-turn
    2: 0.35,  # 35% two-turn
    3: 0.35,  # 35% three-turn
    4: 0.10,  # 10% four-turn
}

# Alternative distributions for different use cases
TURN_DISTRIBUTIONS = {
    "default": DEFAULT_TURN_DISTRIBUTION,
    "short": {1: 0.6, 2: 0.3, 3: 0.1},  # Mostly short conversations
    "long": {2: 0.2, 3: 0.4, 4: 0.4},   # Longer conversations
    "balanced": {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25},  # Equal distribution
}


# ============================================================================
# User Emotions
# ============================================================================

USER_EMOTIONS = ["professional", "happy", "frustrated", "impatient", "confused"]

USER_EMOTION_DISTRIBUTION = {
    "professional": 0.50,  # Most common - neutral, business-like
    "happy": 0.15,         # Positive, enthusiastic
    "frustrated": 0.15,    # Having issues, needs help
    "impatient": 0.10,     # Wants quick answers
    "confused": 0.10,      # Unclear about something
}

EMOTION_DESCRIPTIONS = {
    "professional": "Neutral, business-like tone. Formal language, clear questions.",
    "happy": "Positive, enthusiastic. May express excitement about features or capabilities.",
    "frustrated": "Experiencing issues or challenges. May express mild annoyance or urgency.",
    "impatient": "Wants quick, direct answers. Brief messages, may skip pleasantries.",
    "confused": "Unclear about concepts or features. May ask for clarification or examples."
}


# ============================================================================
# Input Modalities
# ============================================================================

INPUT_MODALITIES = ["standard", "typed_on_phone", "voice_dictated"]

INPUT_MODALITY_DISTRIBUTION = {
    "standard": 0.70,         # Normal typing on computer
    "typed_on_phone": 0.20,   # Mobile typing - autocorrect errors, brevity
    "voice_dictated": 0.10,   # Voice-to-text - filler words, natural speech
}

MODALITY_DESCRIPTIONS = {
    "standard": "Standard computer typing. Clean text, proper formatting.",
    "typed_on_phone": "Mobile device typing. May have autocorrect errors, abbreviations, shorter messages.",
    "voice_dictated": "Voice-to-text transcription. May include 'um', 'uh', natural speech patterns, occasional transcription errors."
}


# ============================================================================
# Text Variations
# ============================================================================

TEXT_VARIATIONS = ["standard", "all_lowercase", "no_punctuation"]

TEXT_VARIATION_DISTRIBUTION = {
    "standard": 0.80,           # Normal capitalization and punctuation
    "all_lowercase": 0.15,      # all lowercase (casual/mobile)
    "no_punctuation": 0.05,     # missing punctuation (rushed/mobile)
}

VARIATION_DESCRIPTIONS = {
    "standard": "Standard capitalization and punctuation.",
    "all_lowercase": "All lowercase letters (common in casual or mobile communication).",
    "no_punctuation": "Missing or minimal punctuation (rushed typing or informal style)."
}


# ============================================================================
# Quality Scoring Weights
# ============================================================================

QUALITY_WEIGHTS = {
    "factual_accuracy": 0.35,
    "naturalness": 0.25,
    "relevance": 0.25,
    "diversity": 0.15,
}


# ============================================================================
# API Configuration
# ============================================================================

@dataclass
class APIConfig:
    """Configuration for API calls."""
    model: str = "gemini-2.5-flash-lite"
    max_concurrent: int = 10
    temperature: float = 0.9  # Higher for generation, lower for judging
    thinking_budget: int = 0
    timeout: int = 60


# Default configurations for each stage
STAGE_CONFIGS = {
    "stage1_qa_extraction": APIConfig(
        model="gemini-2.5-flash",
        temperature=0.8,
        max_concurrent=30,
    ),
    "stage2_qa_validation": APIConfig(
        model="gemini-2.5-flash-lite",
        temperature=0.0,  # Low for validation consistency
        max_concurrent=50,  # Can be faster since lighter model
    ),
    "stage3_conversation_generation": APIConfig(
        model="gemini-2.5-flash",
        temperature=0.9,  # High diversity
        max_concurrent=30,
    ),
    "stage4_judging": APIConfig(
        model="gemini-2.5-flash",
        temperature=0.0,  # Low for consistency
        max_concurrent=50,
    ),
    "stage5_embedding": APIConfig(
        model="text-embedding-3-small",  # OpenAI model
        temperature=0,  # Not applicable to embeddings
        max_concurrent=20,  # High concurrency for batch processing
    ),
}


# ============================================================================
# File Paths
# ============================================================================

@dataclass
class FilePaths:
    """Standard file paths for the pipeline."""
    data_dir: str = "data"
    prompts_dir: str = "prompts"
    output_dir: str = "output"

    # Input files
    source_facts: str = "data/swap_facts.md"

    # Stage outputs (full pipeline)
    stage1_qa_pairs: str = "output/qa_pairs.jsonl"
    stage2_qa_validated: str = "output/qa_pairs_validated.jsonl"
    stage2_qa_validated_passed: str = "output/qa_pairs_validated_passed.jsonl"
    stage3_conversations_raw: str = "output/conversations_raw.jsonl"
    stage4_conversations_judged: str = "output/conversations_judged.jsonl"
    stage5_conversations_embedded: str = "output/conversations_embedded.jsonl"
    stage6_conversations_unique: str = "output/conversations_unique.jsonl"
    stage7_conversations_final: str = "output/conversations_final.jsonl"

    # Trial outputs
    trial_qa_pairs: str = "output/trial_qa_pairs.jsonl"
    trial_qa_validated: str = "output/trial_qa_validated.jsonl"
    trial_conversations_raw: str = "output/trial_conversations_raw.jsonl"
    trial_conversations_judged: str = "output/trial_conversations_judged.jsonl"
    trial_conversations_embedded: str = "output/trial_conversations_embedded.jsonl"
    trial_conversations_unique: str = "output/trial_conversations_unique.jsonl"
    trial_conversations_final: str = "output/trial_conversations_final.jsonl"


PATHS = FilePaths()


# ============================================================================
# Pipeline Parameters
# ============================================================================

@dataclass
class PipelineParams:
    """Parameters for the full pipeline run."""

    # Stage 1: Q&A Extraction
    qa_chunk_context_lines: int = 3
    qa_pairs_per_chunk: int = 3  # Generate 3 Q&A pairs per chunk
    qa_chunk_limit: int | None = None  # None = no limit on chunks

    # Stage 2: Q&A Validation
    qa_validation_enabled: bool = True

    # Stage 3: Conversation Generation
    num_conversations: int = 2000
    conversations_per_qa: int = 10
    turn_distribution: Dict[int, float] = field(default_factory=lambda: DEFAULT_TURN_DISTRIBUTION)
    emotion_distribution: Dict[str, float] = field(default_factory=lambda: USER_EMOTION_DISTRIBUTION)
    modality_distribution: Dict[str, float] = field(default_factory=lambda: INPUT_MODALITY_DISTRIBUTION)
    variation_distribution: Dict[str, float] = field(default_factory=lambda: TEXT_VARIATION_DISTRIBUTION)

    # Stage 4: Judging
    min_quality_score: float = 5.0  # Minimum acceptable score

    # Stage 5: Embedding
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 100
    embedding_max_chars: int = 24000

    # Stage 6: Deduplication
    dedup_enabled: bool = True
    dedup_similarity_threshold: float = 0.95  # 95% similarity

    # Stage 7: Selection
    top_k: int = 1000

    # General
    max_concurrent: int = 10
    random_seed: int | None = 42  # For reproducibility


# Default parameters for full runs
FULL_PARAMS = PipelineParams(
    qa_chunk_limit=None,
    qa_pairs_per_chunk=3,
    num_conversations=2000,
    top_k=1000,
    max_concurrent=10,
    dedup_enabled=True,
)
