"""
Stage 4: Judge conversations and save top candidates.

This script:
1. Loads raw conversations from output/conversations_raw.jsonl
2. Uses Gemini to judge quality of each conversation
3. Ranks by quality score
4. Saves all judged conversations and top 1000 in NanoChat format
"""

import asyncio

import logfire
from dotenv import load_dotenv

from src.synth_data_pipeline.agents import conversation_judge

from src.synth_data_pipeline.models import (
    Conversation,
    JudgedConversation,
    JudgmentScore,
    NanoChatConversation,
    NanoChatMessage,
)
from src.synth_data_pipeline.config import (
    PATHS,
    STAGE_CONFIGS,
    FULL_PARAMS,
)
from src.synth_data_pipeline.utils import (
    load_jsonl,
    save_jsonl,
    process_with_concurrency,
    print_statistics,
)

# Load environment variables
load_dotenv()

# Configure logging
logfire.configure(scrubbing=False)
logfire.instrument_pydantic_ai()

# Get configuration for this stage
config = STAGE_CONFIGS["stage4_judging"]

# Load judging agent definition
judge_prompt_template = conversation_judge.get_prompt_template()
judge_agent = conversation_judge.build_agent(config)


async def judge_conversation(conversation: Conversation) -> JudgedConversation:
    """
    Judge the quality of a conversation.

    Args:
        conversation: Conversation object to judge

    Returns:
        JudgedConversation with quality scores
    """
    # Format conversation for judging
    conv_text = "\n\n".join([
        f"{msg.role.upper()}: {msg.content}"
        for msg in conversation.messages
    ])

    # Format source Q&A pairs for fact-checking
    source_qa_text = "\n\n".join([
        f"Q: {qa.question}\nA: {qa.answer}"
        for qa in conversation.source_qa_pairs
    ])

    # Format the prompt
    prompt_text = judge_prompt_template.prompt.format(
        conversation=conv_text,
        source_qa=source_qa_text if source_qa_text else "No source Q&A available"
    )

    # Judge using the agent
    result = await judge_agent.run(prompt_text)
    judgment = result.output

    return JudgedConversation(
        conversation=conversation,
        judgment=judgment
    )


def conversation_to_nanochat(conversation: Conversation) -> NanoChatConversation:
    """
    Convert a Conversation to NanoChat format.

    Args:
        conversation: Conversation object

    Returns:
        NanoChatConversation (just messages array)
    """
    messages = [
        NanoChatMessage(role=msg.role, content=msg.content)
        for msg in conversation.messages
    ]
    return NanoChatConversation(messages=messages)


def save_top_conversations_nanochat(
    judged_conversations: list[JudgedConversation],
    output_path: str,
    top_k: int = 1000,
    min_score: float = None
):
    """
    Save top K conversations in NanoChat format.

    Args:
        judged_conversations: List of judged conversations
        output_path: Path to output JSONL file
        top_k: Number of top conversations to save
        min_score: Minimum score threshold (optional)
    """
    # Filter to only passing conversations
    passing_conversations = [
        jc for jc in judged_conversations
        if jc.judgment.overall_pass
    ]

    # Sort by number of criteria passed (for ordering within passing conversations)
    def count_passes(jc):
        return sum([
            jc.judgment.factually_accurate,
            jc.judgment.natural_conversation,
            jc.judgment.on_topic,
            jc.judgment.adds_value
        ])

    sorted_conversations = sorted(
        passing_conversations,
        key=count_passes,
        reverse=True
    )

    # Note: min_score parameter is ignored with bool-only system

    # Take top K
    top_conversations = sorted_conversations[:top_k]

    # Convert to NanoChat format and save
    nanochat_convs = [
        conversation_to_nanochat(jc.conversation)
        for jc in top_conversations
    ]
    save_jsonl(nanochat_convs, output_path)

    # Log statistics
    print(f"\nTop {len(top_conversations)} passing conversations selected")
    print(f"  All passed: factually_accurate AND natural AND on_topic AND adds_value")


def print_quality_statistics(judged_conversations: list[JudgedConversation]):
    """Print quality statistics for all judged conversations."""
    if not judged_conversations:
        return

    total = len(judged_conversations)
    passing = sum(1 for jc in judged_conversations if jc.judgment.overall_pass)
    factual_pass = sum(1 for jc in judged_conversations if jc.judgment.factually_accurate)
    natural_pass = sum(1 for jc in judged_conversations if jc.judgment.natural_conversation)
    ontopic_pass = sum(1 for jc in judged_conversations if jc.judgment.on_topic)
    value_pass = sum(1 for jc in judged_conversations if jc.judgment.adds_value)

    print("\n" + "="*80)
    print("QUALITY STATISTICS (All Conversations)")
    print("="*80)
    print(f"Total conversations judged: {total}")
    print(f"Overall PASS (all 4 criteria): {passing} ({passing/total*100:.1f}%)")
    print(f"\nIndividual criteria:")
    print(f"  Factually accurate : {factual_pass}/{total} ({factual_pass/total*100:.1f}%)")
    print(f"  Natural conversation: {natural_pass}/{total} ({natural_pass/total*100:.1f}%)")
    print(f"  On topic           : {ontopic_pass}/{total} ({ontopic_pass/total*100:.1f}%)")
    print(f"  Adds value         : {value_pass}/{total} ({value_pass/total*100:.1f}%)")
    print("="*80 + "\n")


async def main(
    input_file: str = None,
    judged_output: str = None,
    nanochat_output: str = None,
    max_concurrent: int = None,
    top_k: int = None,
    min_score: float = None
):
    """
    Main function to judge conversations and save top K.

    Args:
        input_file: Path to raw conversations JSONL file (default from config)
        judged_output: Path to save all judged conversations (default from config)
        nanochat_output: Path to save top K in NanoChat format (default from config)
        max_concurrent: Maximum concurrent API calls (default from config)
        top_k: Number of top conversations to save (default from config)
        min_score: Minimum quality score threshold (default from config)
    """
    # Use defaults from config if not specified
    input_file = input_file or PATHS.stage3_conversations_raw
    judged_output = judged_output or PATHS.stage4_conversations_judged
    nanochat_output = nanochat_output or PATHS.stage7_conversations_final
    max_concurrent = max_concurrent or config.max_concurrent
    top_k = top_k or FULL_PARAMS.top_k
    min_score = min_score or FULL_PARAMS.min_quality_score

    logfire.info("Starting conversation judging", input_file=input_file)

    # Load conversations
    conversations = load_jsonl(input_file, model_class=Conversation)
    logfire.info(f"Loaded {len(conversations)} conversations")

    # Judge conversations
    with logfire.span("judge_conversations"):
        judged_conversations = await process_with_concurrency(
            conversations,
            judge_conversation,
            max_concurrent=max_concurrent,
            desc="Judging conversations"
        )

    logfire.info(f"Judged {len(judged_conversations)} conversations")

    # Save all judged conversations
    save_jsonl(judged_conversations, judged_output)

    # Print statistics
    print_quality_statistics(judged_conversations)

    # Save top K in NanoChat format
    save_top_conversations_nanochat(
        judged_conversations,
        nanochat_output,
        top_k,
        min_score
    )

    # Print sample of a passing conversation
    passing_convs = [jc for jc in judged_conversations if jc.judgment.overall_pass]
    if passing_convs:
        print("\n" + "="*80)
        print("SAMPLE PASSING CONVERSATION:")
        print("="*80)
        sample = passing_convs[0]
        print(f"Overall: PASS (all 4 criteria met)")
        print(f"Feedback: {sample.judgment.feedback}")
        print("\nConversation:")
        for msg in sample.conversation.messages:
            print(f"\n{msg.role.upper()}: {msg.content[:200]}...")
        print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
