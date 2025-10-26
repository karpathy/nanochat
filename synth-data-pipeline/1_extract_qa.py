"""
Stage 1: Extract Q&A pairs from SWAP Commerce documentation.

This script:
1. Parses swap_facts.md and chunks by bullet points/sections
2. Uses Gemini 2.5 Flash to generate Q&A pairs with context
3. Saves results to output/qa_pairs.jsonl
"""

import asyncio

import logfire
from dotenv import load_dotenv

from src.synth_data_pipeline.agents import qa_extractor

from src.synth_data_pipeline.models import QAPair, QAPairBatch
from src.synth_data_pipeline.config import PATHS, STAGE_CONFIGS, FULL_PARAMS
from src.synth_data_pipeline.utils import (
    parse_markdown_chunks,
    process_with_concurrency,
    save_jsonl,
)

# Load environment variables
load_dotenv()

# Configure logging
logfire.configure(scrubbing=False)
logfire.instrument_pydantic_ai()

# Get configuration for this stage
config = STAGE_CONFIGS["stage1_qa_extraction"]

# Load Q&A extraction agent definition
qa_prompt_template = qa_extractor.get_prompt_template()
qa_agent = qa_extractor.build_agent(config)


async def generate_qa_batch(chunk: dict) -> list[QAPair]:
    """
    Generate 3 Q&A pairs from a text chunk using Gemini.

    Args:
        chunk: Dict with source_text, context_before, context_after

    Returns:
        List of 3 QAPair objects
    """
    # Format the prompt
    prompt_text = qa_prompt_template.prompt.format(
        source_text=chunk['source_text'],
        context_before=chunk['context_before'],
        context_after=chunk['context_after'],
    )

    # Generate batch of 3 Q&A pairs using the agent
    result = await qa_agent.run(prompt_text)
    batch = result.output

    # Return the list of QAPair objects
    return batch.qa_pairs


async def main(
    input_file: str = None,
    output_file: str = None,
    max_concurrent: int = None,
    limit: int = None
):
    """
    Main function to extract Q&A pairs from documentation.

    Args:
        input_file: Path to input markdown file (default from config)
        output_file: Path to output JSONL file (default from config)
        max_concurrent: Maximum concurrent API calls (default from config)
        limit: Limit number of chunks to process (None = no limit)
    """
    # Use defaults from config if not specified
    input_file = input_file or PATHS.source_facts
    output_file = output_file or PATHS.stage1_qa_pairs
    max_concurrent = max_concurrent or config.max_concurrent
    limit = limit or FULL_PARAMS.qa_chunk_limit

    logfire.info("Starting Q&A extraction", input_file=input_file)

    # Parse the markdown file into chunks
    chunks = parse_markdown_chunks(input_file, FULL_PARAMS.qa_chunk_context_lines)
    logfire.info(f"Parsed {len(chunks)} chunks from {input_file}")

    # Limit chunks if specified (for testing)
    if limit:
        chunks = chunks[:limit]
        logfire.info(f"Limited to {limit} chunks")

    # Generate Q&A pairs (3 per chunk)
    with logfire.span("generate_qa_batches"):
        qa_batches = await process_with_concurrency(
            chunks,
            generate_qa_batch,
            max_concurrent=max_concurrent,
            desc="Generating Q&A batches"
        )

    # Flatten the batches into individual QA pairs
    qa_pairs = []
    for batch in qa_batches:
        qa_pairs.extend(batch)

    logfire.info(f"Generated {len(qa_pairs)} Q&A pairs from {len(chunks)} chunks ({len(qa_pairs)/len(chunks):.1f} per chunk)")

    # Filter out Q&A pairs with future dates to avoid hallucination issues
    from datetime import datetime
    today = datetime.now()
    filtered_qa = []
    for qa in qa_pairs:
        # Simple check: if answer or question mentions a date in 2025 or later, skip it
        # This is a pragmatic filter - could be made more sophisticated
        text = qa.answer + " " + qa.question
        has_future_date = any(year in text for year in ["2025", "2026", "2027", "2028"])
        if not has_future_date:
            filtered_qa.append(qa)

    if len(filtered_qa) < len(qa_pairs):
        logfire.info(f"Filtered out {len(qa_pairs) - len(filtered_qa)} Q&A pairs with future dates")
    qa_pairs = filtered_qa

    # Save results
    save_jsonl(qa_pairs, output_file)

    # Print sample for inspection
    if qa_pairs:
        print("\n" + "="*80)
        print("SAMPLE Q&A PAIR:")
        print("="*80)
        sample = qa_pairs[0]
        print(f"Question: {sample.question}")
        print(f"Answer: {sample.answer}")
        print(f"Difficulty: {sample.difficulty}")
        print(f"Categories: {', '.join(sample.categories)}")
        print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
