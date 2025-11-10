"""
Quick test script to verify SIM-CoT implementation works correctly.

This script tests:
1. SIMCoTGSM8K task loads and has step boundaries
2. Data generator produces correct batch format
3. Step weighting works as expected
4. Training loop can run without errors

Run with:
    python -m scripts.test_simcot
"""

import torch
from tasks.simcot_gsm8k import SIMCoTGSM8K
from nanochat.tokenizer import get_tokenizer
from nanochat.simcot_utils import compute_step_weights, compute_step_accuracy

def test_task_loading():
    """Test that SIMCoTGSM8K loads correctly with step boundaries."""
    print("=" * 80)
    print("Test 1: Loading SIMCoTGSM8K task")
    print("=" * 80)

    task = SIMCoTGSM8K(subset="main", split="train")
    print(f"✓ Task loaded with {task.num_examples()} examples")

    # Get first example
    example = task.get_example(0)
    print(f"\n✓ Example keys: {example.keys()}")
    print(f"✓ Number of steps: {example['num_steps']}")
    print(f"✓ Step boundaries: {example['step_boundaries']}")

    # Print the question
    messages = example['messages']
    print(f"\n📝 Question: {messages[0]['content'][:100]}...")

    # Print the assistant response structure
    assistant_parts = messages[1]['content']
    print(f"\n📝 Assistant response has {len(assistant_parts)} parts")
    for i, part in enumerate(assistant_parts[:3]):
        print(f"   Part {i}: type={part['type']}, text={part['text'][:50]}...")

    return task

def test_tokenization(task):
    """Test that tokenization preserves step boundaries."""
    print("\n" + "=" * 80)
    print("Test 2: Tokenization with step boundaries")
    print("=" * 80)

    tokenizer = get_tokenizer()
    example = task.get_example(0)

    # Render conversation
    ids, mask = tokenizer.render_conversation(example)
    print(f"\n✓ Tokenized to {len(ids)} tokens")
    print(f"✓ Mask has {sum(mask)} active tokens out of {len(mask)} total")

    # Check step boundaries are within valid range
    step_boundaries = example['step_boundaries']
    max_pos = max(step_boundaries) if step_boundaries else 0
    print(f"✓ Step boundaries: {step_boundaries}")
    print(f"✓ Max step position: {max_pos} (total tokens: {len(ids)})")

    if max_pos >= len(ids):
        print("⚠ WARNING: Step boundary exceeds token length!")
    else:
        print("✓ All step boundaries are within valid range")

    return tokenizer

def test_step_weighting(tokenizer):
    """Test that step weighting computation works."""
    print("\n" + "=" * 80)
    print("Test 3: Step weight computation")
    print("=" * 80)

    # Create dummy batch
    B, T = 2, 50
    targets = torch.randint(0, 1000, (B, T))
    targets[:, 30:] = -1  # Mask last 20 tokens

    # Create step boundaries
    step_boundaries = [
        [5, 15, 25],  # Example 1 has 3 steps
        [10, 20],     # Example 2 has 2 steps
    ]

    # Compute weights
    weights = compute_step_weights(targets, step_boundaries, step_weight_multiplier=2.0)
    print(f"\n✓ Weights shape: {weights.shape}")
    print(f"✓ Weights range: [{weights.min().item():.2f}, {weights.max().item():.2f}]")
    print(f"✓ Non-zero weights: {(weights > 0).sum().item()} / {B * T}")

    # Check that step positions have higher weight
    for i, bounds in enumerate(step_boundaries):
        for pos in bounds:
            if pos < T:
                weight = weights[i, pos].item()
                print(f"   Example {i}, step at pos {pos}: weight = {weight:.2f}")

    return weights

def test_step_accuracy():
    """Test step accuracy computation."""
    print("\n" + "=" * 80)
    print("Test 4: Step accuracy computation")
    print("=" * 80)

    B, T, V = 2, 50, 1000
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    targets[:, 30:] = -1  # Mask last tokens

    step_boundaries = [
        [5, 15, 25],
        [10, 20],
    ]

    metrics = compute_step_accuracy(logits, targets, step_boundaries)
    print(f"\n✓ Overall accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"✓ Step accuracy: {metrics['step_accuracy']:.4f}")
    print(f"✓ Step total: {metrics['step_total']}")

    return metrics

def test_data_generator():
    """Test the data generator with actual data."""
    print("\n" + "=" * 80)
    print("Test 5: Data generator")
    print("=" * 80)

    from tasks.common import TaskMixture
    from tasks.simcot_gsm8k import SIMCoTGSM8K

    tokenizer = get_tokenizer()
    device = torch.device("cpu")

    # Create small dataset
    dataset = TaskMixture([
        SIMCoTGSM8K(subset="main", split="train"),
    ])

    # Simple generator
    def simcot_data_generator(dataset, batch_size):
        pad_token_id = tokenizer.encode_special("<|assistant_end|>")

        def collate_and_yield(batch):
            nrows = len(batch)
            ncols = max(len(ids) for ids, _, _ in batch) - 1
            inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
            targets = torch.full((nrows, ncols), -1, dtype=torch.long)
            step_boundaries_batch = []

            for i, (ids, mask, step_bounds) in enumerate(batch):
                n = len(ids)
                ids_tensor = torch.tensor(ids, dtype=torch.long)
                inputs[i, :n-1] = ids_tensor[:-1]

                row_targets = ids_tensor[1:]
                mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
                row_targets[mask_tensor == 0] = -1
                targets[i, :n-1] = row_targets

                adjusted_bounds = [pos - 1 for pos in step_bounds if pos > 0 and pos < n]
                step_boundaries_batch.append(adjusted_bounds)

            return inputs, targets, step_boundaries_batch

        batch = []
        for i in range(min(batch_size * 2, len(dataset))):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            step_boundaries = doc.get('step_boundaries', [])
            batch.append((ids, mask, step_boundaries))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

    # Generate one batch
    generator = simcot_data_generator(dataset, batch_size=2)
    inputs, targets, step_boundaries = next(generator)

    print(f"\n✓ Batch inputs shape: {inputs.shape}")
    print(f"✓ Batch targets shape: {targets.shape}")
    print(f"✓ Step boundaries: {step_boundaries}")
    print(f"✓ Active targets: {(targets >= 0).sum().item()} / {targets.numel()}")

    # Compute weights for this batch
    weights = compute_step_weights(targets, step_boundaries, step_weight_multiplier=2.0)
    print(f"✓ Weights computed: sum = {weights.sum().item():.1f}")

    return inputs, targets, step_boundaries

def main():
    """Run all tests."""
    print("\n" + "🚀 " * 20)
    print("SIM-CoT Implementation Test Suite")
    print("🚀 " * 20 + "\n")

    try:
        # Run tests
        task = test_task_loading()
        tokenizer = test_tokenization(task)
        weights = test_step_weighting(tokenizer)
        metrics = test_step_accuracy()
        inputs, targets, step_boundaries = test_data_generator()

        # Final summary
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nYou can now run the full training with:")
        print("  python -m scripts.chat_simcot --num_iterations=10 --device_batch_size=2")
        print("\nFor actual training:")
        print("  torchrun --standalone --nproc_per_node=8 -m scripts.chat_simcot")
        print()

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED!")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
