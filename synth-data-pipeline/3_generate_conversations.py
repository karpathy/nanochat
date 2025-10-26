"""
Stage 3: Generate conversations from validated Q&A pairs.

This script:
1. Loads validated Q&A pairs from output/qa_pairs_validated_passed.jsonl (or the provided path)
2. Samples different conversation configurations
3. Uses Gemini to generate natural conversations
4. Saves results to output/conversations_raw.jsonl
"""

import asyncio
import random

import logfire
from dotenv import load_dotenv

from src.synth_data_pipeline.agents import conversation_generator

from src.synth_data_pipeline.models import QAPair, Conversation
from src.synth_data_pipeline.config import (
    PATHS,
    STAGE_CONFIGS,
    FULL_PARAMS,
)
from src.synth_data_pipeline.sampling import (
    stratified_sample_configs,
    load_system_prompts_from_files,
    set_random_seed,
)
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
config = STAGE_CONFIGS["stage3_conversation_generation"]

# Load conversation generation agent definition
conv_prompt_template = conversation_generator.get_prompt_template()
conv_agent = conversation_generator.build_agent(config)


async def generate_conversation(
    qa_pairs: list[QAPair],
    config_dict: dict
) -> Conversation:
    """
    Generate a conversation from a configuration.

    Args:
        qa_pairs: All available Q&A pairs
        config_dict: Configuration dict with num_turns, style, persona, system_prompt

    Returns:
        Conversation object
    """
    # Sample Q&A pairs for this conversation
    num_turns = config_dict["num_turns"]

    # Filter Q&A by topic relevance to system prompt (pragmatic matching)
    system_prompt_name = config_dict["system_prompt"].name if hasattr(config_dict["system_prompt"], "name") else "helpful"

    # Simple persona matching: sales/solutions prompts shouldn't get governance/company info Q&A
    if "sales" in system_prompt_name.lower() or "solutions" in system_prompt_name.lower():
        # Prefer product/feature Q&A, avoid company registration/director questions
        avoid_keywords = ["appointed", "director", "registered", "incorporation", "companies house"]
        filtered_qa = [qa for qa in qa_pairs if not any(kw in qa.question.lower() for kw in avoid_keywords)]
        qa_pool = filtered_qa if filtered_qa else qa_pairs
    else:
        qa_pool = qa_pairs

    sampled_qa = random.sample(qa_pool, min(num_turns, len(qa_pool)))

    # Format Q&A pairs for the prompt
    qa_text = "\n\n".join([
        f"Q: {qa.question}\nA: {qa.answer}\nCategories: {', '.join(qa.categories)}"
        for qa in sampled_qa
    ])

    # Get system prompt text
    system_prompt = config_dict["system_prompt"].template

    # Format the prompt
    prompt_text = conv_prompt_template.prompt.format(
        num_turns=num_turns,
        style=config_dict["style"],
        user_persona=config_dict["persona"].description,
        system_prompt=system_prompt,
        qa_pairs=qa_text,
    )

    # Generate conversation using the agent
    result = await conv_agent.run(prompt_text)
    conversation = result.output

    # Add source Q&A pairs to conversation for fact-checking
    conversation.source_qa_pairs = sampled_qa

    return conversation


async def main(
    qa_file: str = None,
    output_file: str = None,
    num_conversations: int = None,
    max_concurrent: int = None,
):
    """
    Main function to generate conversations from Q&A pairs.

    Args:
        qa_file: Path to Q&A pairs JSONL file (default from config)
        output_file: Path to output JSONL file (default from config)
        num_conversations: Number of conversations to generate (default from config)
        max_concurrent: Maximum concurrent API calls (default from config)
    """
    # Use defaults from config if not specified
    qa_file = qa_file or PATHS.stage2_qa_validated_passed
    output_file = output_file or PATHS.stage3_conversations_raw
    max_concurrent = max_concurrent or config.max_concurrent

    # Set random seed if specified
    if FULL_PARAMS.random_seed is not None:
        set_random_seed(FULL_PARAMS.random_seed)

    logfire.info("Starting conversation generation", qa_file=qa_file)

    # Load Q&A pairs
    qa_pairs = load_jsonl(qa_file, model_class=QAPair)
    logfire.info(f"Loaded {len(qa_pairs)} Q&A pairs")

    # Determine how many conversations to generate based on available QA pairs
    requested_conversations = num_conversations or FULL_PARAMS.num_conversations
    auto_cap = 0
    if FULL_PARAMS.conversations_per_qa:
        auto_cap = len(qa_pairs) * FULL_PARAMS.conversations_per_qa

    target_conversations = requested_conversations
    if auto_cap:
        target_conversations = min(requested_conversations, auto_cap)

    if target_conversations == 0:
        logfire.warning("No conversations generated because no Q&A pairs are available.")
        return

    logfire.info(
        "Conversation target determined",
        requested=requested_conversations,
        auto_cap=auto_cap,
        final=target_conversations,
    )

    # Load system prompt templates from files (for runtime flexibility)
    system_prompts_from_files = load_system_prompts_from_files()
    logfire.info(f"Loaded {len(system_prompts_from_files)} system prompts from files")

    # Sample conversation configurations
    configs = stratified_sample_configs(
        target_conversations,
        ensure_coverage=True
    )
    logfire.info(f"Sampled {len(configs)} conversation configurations")

    # Generate conversations
    with logfire.span("generate_conversations"):
        # Create a closure that includes qa_pairs
        async def generate_fn(config_dict):
            return await generate_conversation(qa_pairs, config_dict)

        conversations = await process_with_concurrency(
            configs,
            generate_fn,
            max_concurrent=max_concurrent,
            desc="Generating conversations"
        )

    logfire.info(f"Generated {len(conversations)} conversations")

    # Save results
    save_jsonl(conversations, output_file)

    # Print sample for inspection
    if conversations:
        print("\n" + "="*80)
        print("SAMPLE CONVERSATION:")
        print("="*80)
        sample = conversations[0]
        for msg in sample.messages:
            print(f"{msg.role.upper()}: {msg.content[:200]}...")
            print()
        print(f"Metadata: {sample.metadata}")
        print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
