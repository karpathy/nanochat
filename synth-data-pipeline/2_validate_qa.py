"""
Stage 2: Validate Q&A pairs for quality and accuracy.

This script:
1. Loads Q&A pairs from Stage 1
2. Uses Gemini 2.5 Flash Lite to validate each pair
3. Filters out pairs that fail validation
4. Saves validated pairs to output
"""

import asyncio

import logfire
from dotenv import load_dotenv

from src.synth_data_pipeline.agents import qa_validator

from src.synth_data_pipeline.models import QAPair, QAValidation, ValidatedQAPair
from src.synth_data_pipeline.config import PATHS, STAGE_CONFIGS, FULL_PARAMS
from src.synth_data_pipeline.utils import (
    load_jsonl,
    save_jsonl,
    process_with_concurrency,
)

# Load environment variables
load_dotenv()

# Configure logging
logfire.configure(scrubbing=False)
logfire.instrument_pydantic_ai()

# Get configuration for this stage
config = STAGE_CONFIGS["stage2_qa_validation"]

# Load validation agent definition
validation_prompt_template = qa_validator.get_prompt_template()
validation_agent = qa_validator.build_agent(config)


async def validate_qa_pair(qa_pair: QAPair) -> ValidatedQAPair:
    """
    Validate a Q&A pair using Gemini.

    Args:
        qa_pair: QAPair object to validate

    Returns:
        ValidatedQAPair with validation result
    """
    # Format the prompt
    prompt_text = validation_prompt_template.prompt.format(
        question=qa_pair.question,
        answer=qa_pair.answer,
        source_text=qa_pair.source_text,
        context_before=qa_pair.context_before,
        context_after=qa_pair.context_after,
    )

    # Validate using the agent
    result = await validation_agent.run(prompt_text)
    validation = result.output

    return ValidatedQAPair(
        qa_pair=qa_pair,
        validation=validation
    )


async def main(
    input_file: str = None,
    output_file: str = None,
    max_concurrent: int = None
):
    """
    Main function to validate Q&A pairs.

    Args:
        input_file: Path to input JSONL file (default from config)
        output_file: Path to output JSONL file (default from config)
        max_concurrent: Maximum concurrent API calls (default from config)
    """
    # Use defaults from config if not specified
    input_file = input_file or PATHS.stage1_qa_pairs
    output_file = output_file or PATHS.stage2_qa_validated
    max_concurrent = max_concurrent or config.max_concurrent

    logfire.info("Starting Q&A validation", input_file=input_file)

    # Load Q&A pairs
    qa_pairs = load_jsonl(input_file, model_class=QAPair)
    logfire.info(f"Loaded {len(qa_pairs)} Q&A pairs")

    # Validate all pairs
    with logfire.span("validate_qa_pairs"):
        validated_pairs = await process_with_concurrency(
            qa_pairs,
            validate_qa_pair,
            max_concurrent=max_concurrent,
            desc="Validating Q&A pairs"
        )

    # Count passed/failed
    passed = [vp for vp in validated_pairs if vp.validation.passed]
    failed = [vp for vp in validated_pairs if not vp.validation.passed]

    logfire.info(
        f"Validation complete: {len(passed)} passed, {len(failed)} failed "
        f"({100 * len(failed) / len(validated_pairs):.1f}% rejection rate)"
    )

    # Save all validated pairs (with validation results)
    save_jsonl(validated_pairs, output_file)

    # Also save just the passed Q&A pairs (without validation metadata) for next stage
    passed_qa_pairs = [vp.qa_pair for vp in passed]
    if output_file == PATHS.stage2_qa_validated:
        passed_output = PATHS.stage2_qa_validated_passed
    else:
        passed_output = output_file.replace('.jsonl', '_passed.jsonl')

    save_jsonl(passed_qa_pairs, passed_output)
    logfire.info(f"Saved {len(passed_qa_pairs)} passed Q&A pairs to {passed_output}")

    # Print sample
    if validated_pairs:
        print("\n" + "="*80)
        print("VALIDATION SAMPLE:")
        print("="*80)
        sample = validated_pairs[0]
        print(f"Question: {sample.qa_pair.question}")
        print(f"Answer: {sample.qa_pair.answer[:100]}...")
        print(f"\nValidation:")
        print(f"  uses_source_fact: {sample.validation.uses_source_fact}")
        print(f"  realistic_question: {sample.validation.realistic_question}")
        print(f"  sensible_answer: {sample.validation.sensible_answer}")
        print(f"  PASSED: {sample.validation.passed}")
        print(f"  Feedback: {sample.validation.feedback}")
        print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
