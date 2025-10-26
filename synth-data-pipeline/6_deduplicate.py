"""
Stage 6: Deduplicate conversations based on embedding similarity.

This script:
1. Loads embedded conversations from Stage 5
2. L2-normalizes embeddings
3. Computes pairwise cosine similarity
4. Removes duplicates above similarity threshold
5. Saves unique conversations
"""

import asyncio
import numpy as np

import logfire

from src.synth_data_pipeline.models import EmbeddedConversation, UniqueConversation
from src.synth_data_pipeline.config import PATHS, FULL_PARAMS
from src.synth_data_pipeline.utils import load_jsonl, save_jsonl
from src.synth_data_pipeline.embedding_utils import (
    l2_normalize,
    greedy_deduplicate,
)

# Configure logging
logfire.configure(scrubbing=False)


async def main(
    input_file: str = None,
    output_file: str = None,
    similarity_threshold: float = None
):
    """
    Main function to deduplicate conversations.

    Args:
        input_file: Path to input JSONL file (default from config)
        output_file: Path to output JSONL file (default from config)
        similarity_threshold: Similarity threshold for deduplication (default from config)
    """
    # Use defaults from config if not specified
    input_file = input_file or PATHS.stage5_conversations_embedded
    output_file = output_file or PATHS.stage6_conversations_unique
    similarity_threshold = similarity_threshold or FULL_PARAMS.dedup_similarity_threshold

    logfire.info(
        "Starting deduplication",
        input_file=input_file,
        similarity_threshold=similarity_threshold
    )

    # Load embedded conversations
    embedded_convs = load_jsonl(input_file, model_class=EmbeddedConversation)
    logfire.info(f"Loaded {len(embedded_convs)} embedded conversations")

    # Extract embeddings and scores
    embeddings = [np.array(ec.embedding, dtype=np.float32) for ec in embedded_convs]
    scores = [ec.judgment.overall_score for ec in embedded_convs]

    # L2-normalize embeddings for cosine similarity
    with logfire.span("normalize_embeddings"):
        normalized_embeddings = l2_normalize(embeddings)

    logfire.info("Normalized embeddings for cosine similarity")

    # Deduplicate
    with logfire.span("deduplicate"):
        kept_indices = greedy_deduplicate(
            normalized_embeddings,
            scores,
            similarity_threshold=similarity_threshold
        )

    # Create unique conversations (without embeddings to save space)
    unique_convs = []
    for idx in kept_indices:
        ec = embedded_convs[idx]
        unique_conv = UniqueConversation(
            conversation=ec.conversation,
            judgment=ec.judgment
        )
        unique_convs.append(unique_conv)

    # Save results
    save_jsonl(unique_convs, output_file)

    # Statistics
    total = len(embedded_convs)
    kept = len(unique_convs)
    removed = total - kept
    removal_rate = 100 * removed / total if total > 0 else 0

    logfire.info(
        f"Deduplication complete: {kept} kept, {removed} removed ({removal_rate:.1f}%)"
    )

    # Print statistics
    print("\n" + "="*80)
    print("DEDUPLICATION STATISTICS:")
    print("="*80)
    print(f"Total conversations: {total}")
    print(f"Unique conversations: {kept}")
    print(f"Duplicates removed: {removed} ({removal_rate:.1f}%)")
    print(f"Similarity threshold: {similarity_threshold}")
    print("="*80 + "\n")

    # Print score statistics for unique conversations
    unique_scores = [uc.judgment.overall_score for uc in unique_convs]
    if unique_scores:
        print("Unique conversation scores:")
        print(f"  Average: {np.mean(unique_scores):.2f}")
        print(f"  Min: {np.min(unique_scores):.2f}")
        print(f"  Max: {np.max(unique_scores):.2f}")
        print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
