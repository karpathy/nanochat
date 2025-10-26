"""
Stage 7: Select top K conversations and convert to NanoChat format.

This script:
1. Loads unique conversations from Stage 6
2. Sorts by quality score
3. Selects top K conversations
4. Converts to NanoChat format (messages only)
5. Saves final dataset
"""

import asyncio

import logfire

from src.synth_data_pipeline.models import UniqueConversation, NanoChatConversation, NanoChatMessage
from src.synth_data_pipeline.config import PATHS, FULL_PARAMS
from src.synth_data_pipeline.utils import load_jsonl, save_jsonl

# Configure logging
logfire.configure(scrubbing=False)


def conversation_to_nanochat(unique_conv: UniqueConversation) -> NanoChatConversation:
    """
    Convert a UniqueConversation to NanoChat format.

    Args:
        unique_conv: UniqueConversation object

    Returns:
        NanoChatConversation (messages only)
    """
    nanochat_messages = [
        NanoChatMessage(role=msg.role, content=msg.content)
        for msg in unique_conv.conversation.messages
    ]

    return NanoChatConversation(messages=nanochat_messages)


async def main(
    input_file: str = None,
    output_file: str = None,
    top_k: int = None,
    min_score: float = None
):
    """
    Main function to select top K conversations.

    Args:
        input_file: Path to input JSONL file (default from config)
        output_file: Path to output JSONL file (default from config)
        top_k: Number of top conversations to select (default from config)
        min_score: Minimum quality score threshold (default from config)
    """
    # Use defaults from config if not specified
    input_file = input_file or PATHS.stage6_conversations_unique
    output_file = output_file or PATHS.stage7_conversations_final
    top_k = top_k or FULL_PARAMS.top_k
    min_score = min_score or FULL_PARAMS.min_quality_score

    logfire.info(
        "Starting top-K selection",
        input_file=input_file,
        top_k=top_k,
        min_score=min_score
    )

    # Load unique conversations
    unique_convs = load_jsonl(input_file, model_class=UniqueConversation)
    logfire.info(f"Loaded {len(unique_convs)} unique conversations")

    # Filter by minimum score if specified
    if min_score is not None:
        filtered_convs = [
            uc for uc in unique_convs
            if uc.judgment.overall_score >= min_score
        ]
        logfire.info(
            f"Filtered to {len(filtered_convs)} conversations with score >= {min_score}"
        )
    else:
        filtered_convs = unique_convs

    # Sort by quality score (descending)
    sorted_convs = sorted(
        filtered_convs,
        key=lambda uc: uc.judgment.overall_score,
        reverse=True
    )

    # Select top K
    top_convs = sorted_convs[:top_k]
    logfire.info(f"Selected top {len(top_convs)} conversations")

    # Convert to NanoChat format
    nanochat_convs = [conversation_to_nanochat(uc) for uc in top_convs]

    # Save results
    save_jsonl(nanochat_convs, output_file)
    logfire.info(f"Saved {len(nanochat_convs)} conversations in NanoChat format")

    # Print statistics
    print("\n" + "="*80)
    print("TOP-K SELECTION STATISTICS:")
    print("="*80)
    print(f"Total unique conversations: {len(unique_convs)}")
    print(f"After minimum score filter: {len(filtered_convs)}")
    print(f"Top K selected: {len(top_convs)}")
    print("="*80 + "\n")

    if top_convs:
        scores = [uc.judgment.overall_score for uc in top_convs]
        print("Selected conversation scores:")
        print(f"  Average: {sum(scores) / len(scores):.2f}")
        print(f"  Min: {min(scores):.2f}")
        print(f"  Max: {max(scores):.2f}")
        print("="*80 + "\n")

        # Show best conversation
        best = top_convs[0]
        print("BEST CONVERSATION:")
        print("="*80)
        print(f"Score: {best.judgment.overall_score:.2f}")
        print(f"Feedback: {best.judgment.feedback}")
        print("\nMessages:")
        for msg in best.conversation.messages:
            print(f"  {msg.role.upper()}: {msg.content[:100]}...")
        print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
