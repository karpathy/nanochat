"""
Trial run script to validate the pipeline with a small dataset.

This script:
1. Runs all 3 stages with limited data
2. Validates prompts and logic
3. Prints samples for manual inspection
"""

import asyncio
import json
from pathlib import Path

import logfire

# Import the main functions from each stage
from src.synth_data_pipeline.models import QAPair, Conversation, JudgedConversation

# We'll use the actual script functions
import sys
sys.path.append(str(Path(__file__).parent))


TRIAL_QA_CHUNK_LIMIT = 10
TRIAL_FINAL_CONVERSATIONS = "output/trial_conversations_final.jsonl"


async def trial_extract_qa():
    """Trial run of Q&A extraction with 10 chunks."""
    print("\n" + "="*80)
    print("STAGE 1: Q&A EXTRACTION (Trial with 10 chunks)")
    print("="*80)

    from importlib import import_module
    stage1 = import_module('1_extract_qa')

    # Run with trial parameters (uses STAGE_CONFIGS for max_concurrent)
    await stage1.main(
        input_file="data/swap_facts.md",
        output_file="output/trial_qa_pairs.jsonl",
        limit=TRIAL_QA_CHUNK_LIMIT,
    )

    # Load and show results
    qa_pairs = []
    with open("output/trial_qa_pairs.jsonl", 'r') as f:
        for line in f:
            qa_pairs.append(QAPair.model_validate_json(line))

    print(f"\n✓ Generated {len(qa_pairs)} Q&A pairs")

    # Show first 3
    for i, qa in enumerate(qa_pairs[:3], 1):
        print(f"\n--- Q&A Pair {i} ---")
        print(f"Q: {qa.question}")
        print(f"A: {qa.answer[:150]}...")
        print(f"Difficulty: {qa.difficulty}")
        print(f"Categories: {', '.join(qa.categories)}")

    return len(qa_pairs)


async def trial_validate_qa(num_qa_pairs: int):
    """Trial run of Q&A validation."""
    print("\n" + "="*80)
    print("STAGE 2: Q&A VALIDATION (Trial with production configs)")
    print("="*80)

    from importlib import import_module
    stage2 = import_module('2_validate_qa')

    await stage2.main(
        input_file="output/trial_qa_pairs.jsonl",
        output_file="output/trial_qa_validated.jsonl",
    )

    passed_pairs = []
    with open("output/trial_qa_validated_passed.jsonl", 'r') as f:
        for line in f:
            passed_pairs.append(QAPair.model_validate_json(line))

    print(f"\n✓ Validated {num_qa_pairs} Q&A pairs")
    print(f"✓ {len(passed_pairs)} passed validation")

    for i, qa in enumerate(passed_pairs[:3], 1):
        print(f"\n--- Passed Q&A {i} ---")
        print(f"Q: {qa.question}")
        print(f"A: {qa.answer[:150]}...")

    return len(passed_pairs)


async def trial_generate_conversations(num_valid_pairs: int):
    """Trial run of conversation generation with 20 conversations."""
    print("\n" + "="*80)
    print("STAGE 3: CONVERSATION GENERATION (Trial with production configs)")
    print("="*80)

    from importlib import import_module
    stage3 = import_module('3_generate_conversations')

    # Run with trial parameters (uses STAGE_CONFIGS for max_concurrent)
    await stage3.main(
        qa_file="output/trial_qa_validated_passed.jsonl",
        output_file="output/trial_conversations_raw.jsonl",
    )

    # Load and show results
    conversations = []
    with open("output/trial_conversations_raw.jsonl", 'r') as f:
        for line in f:
            conversations.append(Conversation.model_validate_json(line))

    print(f"\n✓ Valid Q&A pairs available: {num_valid_pairs}")
    print(f"✓ Generated {len(conversations)} conversations")

    # Show first 2
    for i, conv in enumerate(conversations[:2], 1):
        print(f"\n--- Conversation {i} ---")
        print(f"Style: {conv.metadata.style}")
        print(f"Persona: {conv.metadata.user_persona}")
        print(f"Turns: {conv.metadata.num_turns}")
        print("\nMessages:")
        for msg in conv.messages:
            print(f"  {msg.role.upper()}: {msg.content[:100]}...")

    return len(conversations)


async def trial_judge_conversations(num_conversations: int):
    """Trial run of judging all conversations."""
    print("\n" + "="*80)
    print("STAGE 4: JUDGING & SELECTION (Trial with all conversations)")
    print("="*80)

    from importlib import import_module
    stage3 = import_module('4_judge_and_save')

    # Judge all and save top K (uses STAGE_CONFIGS for max_concurrent)
    await stage3.main(
        input_file="output/trial_conversations_raw.jsonl",
        judged_output="output/trial_conversations_judged.jsonl",
        nanochat_output=TRIAL_FINAL_CONVERSATIONS,
    )

    # Load and show results
    judged = []
    with open("output/trial_conversations_judged.jsonl", 'r') as f:
        for line in f:
            judged.append(JudgedConversation.model_validate_json(line))

    print(f"\n✓ Judged {len(judged)} conversations")

    # Show pass/fail statistics (bool-based system)
    total = len(judged)
    passing = sum(1 for jc in judged if jc.judgment.overall_pass)
    factual_pass = sum(1 for jc in judged if jc.judgment.factually_accurate)
    natural_pass = sum(1 for jc in judged if jc.judgment.natural_conversation)
    ontopic_pass = sum(1 for jc in judged if jc.judgment.on_topic)
    value_pass = sum(1 for jc in judged if jc.judgment.adds_value)

    print(f"\nQuality statistics:")
    print(f"  Overall PASS (all 4 criteria): {passing}/{total} ({passing/total*100:.1f}%)")
    print(f"\nIndividual criteria:")
    print(f"  Factually accurate : {factual_pass}/{total} ({factual_pass/total*100:.1f}%)")
    print(f"  Natural conversation: {natural_pass}/{total} ({natural_pass/total*100:.1f}%)")
    print(f"  On topic           : {ontopic_pass}/{total} ({ontopic_pass/total*100:.1f}%)")
    print(f"  Adds value         : {value_pass}/{total} ({value_pass/total*100:.1f}%)")

    # Show sample passing and failing conversations
    passing_convs = [jc for jc in judged if jc.judgment.overall_pass]
    failing_convs = [jc for jc in judged if not jc.judgment.overall_pass]

    if passing_convs:
        sample = passing_convs[0]
        print(f"\n--- Sample PASSING Conversation ---")
        print(f"Feedback: {sample.judgment.feedback}")

    if failing_convs:
        sample = failing_convs[0]
        print(f"\n--- Sample FAILING Conversation ---")
        print(f"Failed criteria: ", end="")
        failed = []
        if not sample.judgment.factually_accurate: failed.append("factual")
        if not sample.judgment.natural_conversation: failed.append("natural")
        if not sample.judgment.on_topic: failed.append("on-topic")
        if not sample.judgment.adds_value: failed.append("adds-value")
        print(", ".join(failed))
        print(f"Feedback: {sample.judgment.feedback}")
        if sample.judgment.issues:
            print(f"Issues: {', '.join(sample.judgment.issues)}")

    return len(judged)


def validate_output_formats():
    """Validate that output files match expected formats."""
    print("\n" + "="*80)
    print("VALIDATION: Checking output formats")
    print("="*80)

    checks = {
        "Q&A pairs JSONL": "output/trial_qa_pairs.jsonl",
        "Raw conversations JSONL": "output/trial_conversations_raw.jsonl",
        "Judged conversations JSONL": "output/trial_conversations_judged.jsonl",
        "NanoChat format JSONL": TRIAL_FINAL_CONVERSATIONS,
    }

    all_valid = True

    for name, path in checks.items():
        if not Path(path).exists():
            print(f"✗ {name}: FILE NOT FOUND")
            all_valid = False
            continue

        try:
            with open(path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    print(f"✗ {name}: EMPTY FILE")
                    all_valid = False
                    continue

                # Try parsing first line as JSON
                json.loads(lines[0])
                print(f"✓ {name}: Valid ({len(lines)} entries)")

        except Exception as e:
            print(f"✗ {name}: {e}")
            all_valid = False

    return all_valid


async def main():
    """Run the complete trial pipeline."""
    print("="*80)
    print("SYNTHETIC DATA PIPELINE - TRIAL RUN")
    print("="*80)
    print("\nThis will:")
    print(f"  1. Extract up to {TRIAL_QA_CHUNK_LIMIT} chunks worth of Q&A pairs")
    print("  2. Validate those Q&A pairs with the production agent")
    print("  3. Generate conversations using the same production configuration")
    print("  4. Judge all conversations and select the configured top K")
    print()

    # Configure logfire without sending to cloud (for trial runs)
    logfire.configure(send_to_logfire=False)

    try:
        # Stage 1: Extract Q&A
        num_qa = await trial_extract_qa()

        # Stage 2: Validate Q&A
        num_valid = await trial_validate_qa(num_qa)

        # Stage 3: Generate conversations
        num_conv = await trial_generate_conversations(num_valid)

        # Stage 4: Judge and select
        num_judged = await trial_judge_conversations(num_conv)

        # Validate formats
        all_valid = validate_output_formats()

        # Final summary
        print("\n" + "="*80)
        print("TRIAL RUN COMPLETE")
        print("="*80)
        print(f"✓ Q&A pairs extracted: {num_qa}")
        print(f"✓ Q&A pairs passed validation: {num_valid}")
        print(f"✓ Conversations generated: {num_conv}")
        print(f"✓ Conversations judged: {num_judged}")
        print(f"✓ Output formats valid: {'YES' if all_valid else 'NO'}")
        print()

        if all_valid:
            print("🎉 Trial run successful! You can now run the full pipeline.")
            print()
            print("Next steps:")
            print("  1. Review the trial outputs in output/ directory")
            print("  2. Adjust prompts if needed")
            print("  3. Run full pipeline:")
            print("     - uv run 1_extract_qa.py")
            print("     - uv run 2_validate_qa.py")
            print("     - uv run 3_generate_conversations.py")
            print("     - uv run 4_judge_and_save.py")
        else:
            print("⚠️  Some validations failed. Please review the errors above.")

    except Exception as e:
        print(f"\n❌ Trial run failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
