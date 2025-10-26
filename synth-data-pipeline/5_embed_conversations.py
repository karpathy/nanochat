"""
Stage 5: Embed conversations using OpenAI embeddings.

This script:
1. Loads judged conversations from Stage 4
2. Converts each conversation to text
3. Generates embeddings using OpenAI API
4. Saves conversations with embeddings
"""

import asyncio
import os

import logfire
from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.synth_data_pipeline.models import JudgedConversation, EmbeddedConversation
from src.synth_data_pipeline.config import PATHS, FULL_PARAMS
from src.synth_data_pipeline.utils import load_jsonl, save_jsonl
from src.synth_data_pipeline.embedding_utils import (
    batch_embed,
    conversation_to_text,
)

# Load environment variables
load_dotenv()

# Configure logging
logfire.configure(scrubbing=False)


async def main(
    input_file: str = None,
    output_file: str = None
):
    """
    Main function to embed conversations.

    Args:
        input_file: Path to input JSONL file (default from config)
        output_file: Path to output JSONL file (default from config)
    """
    # Use defaults from config if not specified
    input_file = input_file or PATHS.stage4_conversations_judged
    output_file = output_file or PATHS.stage5_conversations_embedded

    logfire.info("Starting conversation embedding", input_file=input_file)

    # Load judged conversations
    judged_convs = load_jsonl(input_file, model_class=JudgedConversation)
    logfire.info(f"Loaded {len(judged_convs)} judged conversations")

    # Convert conversations to text
    texts = []
    for jc in judged_convs:
        # Convert messages to text
        text = conversation_to_text(
            [msg.model_dump() for msg in jc.conversation.messages],
            max_chars=FULL_PARAMS.embedding_max_chars
        )
        texts.append(text)

    # Initialize OpenAI client
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Generate embeddings
    with logfire.span("generate_embeddings"):
        embeddings = await batch_embed(
            texts,
            client,
            model=FULL_PARAMS.embedding_model,
            dimensions=FULL_PARAMS.embedding_dimensions,
            batch_size=FULL_PARAMS.embedding_batch_size,
            max_concurrent=20
        )

    logfire.info(f"Generated {len(embeddings)} embeddings")

    # Create embedded conversations
    embedded_convs = []
    for jc, emb, text in zip(judged_convs, embeddings, texts):
        embedded_conv = EmbeddedConversation(
            conversation=jc.conversation,
            judgment=jc.judgment,
            embedding=emb.tolist(),  # Convert numpy array to list
            text_preview=text[:200]  # First 200 chars for debugging
        )
        embedded_convs.append(embedded_conv)

    # Save results
    save_jsonl(embedded_convs, output_file)
    logfire.info(f"Saved {len(embedded_convs)} embedded conversations")

    # Print sample
    if embedded_convs:
        print("\n" + "="*80)
        print("EMBEDDING SAMPLE:")
        print("="*80)
        sample = embedded_convs[0]
        print(f"Conversation preview: {sample.text_preview}...")
        print(f"Embedding dimensions: {len(sample.embedding)}")
        print(f"Quality score: {sample.judgment.overall_score:.2f}")
        print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
