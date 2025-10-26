"""
Pydantic models for the synthetic data generation pipeline.
"""

from typing import Literal
from pydantic import BaseModel, Field


# ============================================================================
# Stage 1: Q&A Extraction Models
# ============================================================================


class QAPair(BaseModel):
    """A question-answer pair extracted from source documentation."""

    question: str = Field(
        description="A natural question that could be asked about this topic"
    )
    answer: str = Field(
        description="The accurate answer grounded in the source text"
    )
    source_text: str = Field(
        description="The specific text chunk this Q&A was generated from"
    )
    context_before: str = Field(
        default="",
        description="Preceding lines for context"
    )
    context_after: str = Field(
        default="",
        description="Following lines for context"
    )
    difficulty: Literal["basic", "intermediate", "advanced"] = Field(
        description="The difficulty level of this question"
    )
    categories: list[str] = Field(
        default_factory=list,
        description="Topic categories (e.g., 'pricing', 'features', 'integrations')"
    )


class QAPairBatch(BaseModel):
    """A batch of 3 Q&A pairs generated from a single chunk."""

    qa_pairs: list[QAPair] = Field(
        description="Exactly 3 diverse Q&A pairs from the same source chunk"
    )


# ============================================================================
# Stage 2: Q&A Validation Models
# ============================================================================


class QAValidation(BaseModel):
    """Validation result for a Q&A pair."""

    uses_source_fact: bool = Field(
        description="Does the Q&A correctly use facts from the source text (no hallucinations)?"
    )
    realistic_question: bool = Field(
        description="Is this a question a real person would ask?"
    )
    sensible_answer: bool = Field(
        description="Is the answer appropriate and sensible for the question?"
    )
    passed: bool = Field(
        description="Overall pass (all three bools must be True)"
    )
    feedback: str = Field(
        description="Brief explanation of validation result"
    )


class ValidatedQAPair(BaseModel):
    """A Q&A pair with its validation result."""

    qa_pair: QAPair = Field(
        description="The Q&A pair being validated"
    )
    validation: QAValidation = Field(
        description="The validation result"
    )


# ============================================================================
# Stage 3: Conversation Generation Models
# ============================================================================


class Message(BaseModel):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant"] = Field(
        description="The role of the message sender"
    )
    content: str = Field(
        description="The message content"
    )


class ConversationMetadata(BaseModel):
    """Metadata about how a conversation was generated."""

    num_turns: int = Field(
        description="Number of user-assistant turns (not counting system message)"
    )
    style: Literal["formal", "casual", "technical"] = Field(
        description="The conversation style"
    )
    user_persona: str = Field(
        description="The persona/role of the user (e.g., 'developer', 'business owner')"
    )
    user_emotion: Literal["professional", "happy", "frustrated", "impatient", "confused"] = Field(
        default="professional",
        description="The emotional state of the user"
    )
    input_modality: Literal["standard", "typed_on_phone", "voice_dictated"] = Field(
        default="standard",
        description="How the user is inputting their messages"
    )
    text_variation: Literal["standard", "all_lowercase", "no_punctuation"] = Field(
        default="standard",
        description="Text formatting variation applied to user messages"
    )
    source_qa_ids: list[int] = Field(
        default_factory=list,
        description="Indices of Q&A pairs used to generate this conversation"
    )
    difficulty: str = Field(
        description="Overall difficulty level"
    )
    categories: list[str] = Field(
        default_factory=list,
        description="Topic categories covered in this conversation"
    )


class Conversation(BaseModel):
    """A complete conversation with metadata."""

    messages: list[Message] = Field(
        description="The conversation messages (system, user, assistant)"
    )
    metadata: ConversationMetadata = Field(
        description="Metadata about this conversation"
    )
    source_qa_pairs: list[QAPair] = Field(
        default_factory=list,
        description="The Q&A pairs used to generate this conversation (for fact-checking)"
    )


# ============================================================================
# Stage 4: Judging Models
# ============================================================================


class JudgmentScore(BaseModel):
    """Quality judgment for a conversation using clear YES/NO rubrics."""

    factually_accurate: bool = Field(
        description="PASS: All facts match source Q&A, no hallucinations or invented details"
    )
    natural_conversation: bool = Field(
        description="PASS: Sounds human, flows naturally, realistic interaction"
    )
    on_topic: bool = Field(
        description="PASS: Relevant to SWAP Commerce, would be useful for training"
    )
    adds_value: bool = Field(
        description="PASS: Not generic/repetitive, covers topic in specific/interesting way"
    )
    overall_pass: bool = Field(
        description="TRUE only if ALL four criteria above are TRUE"
    )
    feedback: str = Field(
        description="Brief explanation of judgment (1-2 sentences)"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Specific problems found (if any)"
    )


class JudgedConversation(BaseModel):
    """A conversation with its quality judgment."""

    conversation: Conversation = Field(
        description="The conversation being judged"
    )
    judgment: JudgmentScore = Field(
        description="The quality judgment scores"
    )


# ============================================================================
# Stage 5: Embedding Models
# ============================================================================


class EmbeddedConversation(BaseModel):
    """A judged conversation with its embedding."""

    conversation: Conversation = Field(
        description="The conversation"
    )
    judgment: JudgmentScore = Field(
        description="The quality judgment"
    )
    embedding: list[float] = Field(
        description="Conversation embedding (1024 dimensions)"
    )
    text_preview: str = Field(
        description="First 200 characters for debugging"
    )


# ============================================================================
# Stage 6: Deduplication Models
# ============================================================================


class UniqueConversation(BaseModel):
    """A conversation marked as unique after deduplication."""

    conversation: Conversation = Field(
        description="The conversation"
    )
    judgment: JudgmentScore = Field(
        description="The quality judgment"
    )
    # Note: embedding removed to save space after dedup


# ============================================================================
# Stage 7: Final Output Format (NanoChat compatible)
# ============================================================================


class NanoChatMessage(BaseModel):
    """A message in NanoChat format."""

    role: Literal["system", "user", "assistant"]
    content: str


class NanoChatConversation(BaseModel):
    """NanoChat training format - just the messages array."""

    messages: list[NanoChatMessage]


# ============================================================================
# Prompt Generation Models (for structured LLM outputs)
# ============================================================================


class QAGenerationRequest(BaseModel):
    """Input for Q&A generation from a text chunk."""

    chunk: str
    context_before: str = ""
    context_after: str = ""


class ConversationGenerationRequest(BaseModel):
    """Input for conversation generation."""

    qa_pairs: list[QAPair]
    num_turns: int = Field(ge=1, le=4, description="Number of conversation turns")
    style: Literal["formal", "casual", "technical"]
    user_persona: str
    system_prompt_template: str
