# Feature Implementation Guide - Part 2

This is the continuation of the Feature Implementation Guide, covering features 6-10.

**See [Part 1](09_feature_implementation_guide.md) for Features 1-5.**

## Table of Contents (Part 2)
6. [Feature 6: Generation Parameter Explorer](#feature-6-generation-parameter-explorer)
7. [Feature 7: Training Resume Helper](#feature-7-training-resume-helper)
8. [Feature 8: Simple Attention Visualizer](#feature-8-simple-attention-visualizer)
9. [Feature 9: Learning Rate Finder](#feature-9-learning-rate-finder)
10. [Feature 10: Conversation Template Builder](#feature-10-conversation-template-builder)

---

## Feature 6: Generation Parameter Explorer

### Why This Feature is Useful

**Problem it solves:**
- Hard to understand how temperature/top-k/top-p affect output
- Can't compare different sampling strategies side-by-side
- No way to see model's internal probabilities
- Difficult to find optimal generation settings

**Learning benefits:**
- **Sampling strategies**: See temperature, top-k, top-p in action
- **Probability distributions**: Understand what the model "thinks"
- **Determinism vs randomness**: Learn trade-offs
- **Parameter sensitivity**: See how small changes affect output

**Practical benefits:**
- Find best settings for your use case
- Debug why model outputs are too random or too boring
- Understand model confidence
- Create diverse outputs when needed

### Implementation Details

#### New Files to Create

**File 1: `tools/generation_explorer.py`**

```python
"""
Generation Parameter Explorer

Interactively explore how sampling parameters affect model outputs.
"""

import argparse
import torch
import torch.nn.functional as F
from nanochat.gpt import GPT
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine


class GenerationExplorer:
    """Tool for exploring generation parameters."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.engine = Engine(model, tokenizer)

    def sample_with_probabilities(self, prompt: str, max_tokens: int = 50,
                                  temperature: float = 1.0, top_k: int = None,
                                  show_probs: bool = True, num_alternatives: int = 5):
        """
        Generate text and show probability distribution at each step.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            show_probs: Whether to show probabilities
            num_alternatives: Number of alternative tokens to show
        """
        # Encode prompt
        prompt_tokens = self.tokenizer.encode(prompt, prepend="<|bos|>")

        print(f"\n{'='*100}")
        print(f"GENERATION WITH PROBABILITIES")
        print(f"{'='*100}\n")
        print(f"Prompt: \"{prompt}\"")
        print(f"Temperature: {temperature}, Top-k: {top_k}\n")
        print(f"{'-'*100}\n")

        # Generate token by token
        generated_tokens = []
        current_tokens = prompt_tokens.copy()

        for step in range(max_tokens):
            # Forward pass
            with torch.no_grad():
                ids = torch.tensor([current_tokens], dtype=torch.long, device=self.model.get_device())
                logits = self.model(ids)
                logits = logits[0, -1, :]  # Last position

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float('Inf')

            # Get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample
            if temperature > 0:
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = torch.argmax(logits).item()

            # Show probabilities
            if show_probs:
                # Get top alternatives
                top_probs, top_indices = torch.topk(probs, num_alternatives)

                print(f"Step {step + 1}:")
                print(f"  Sampled: [{next_token}] \"{self.tokenizer.decode([next_token])}\" (p={probs[next_token]:.4f})")

                print(f"  Top {num_alternatives} alternatives:")
                for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    token_str = self.tokenizer.decode([idx.item()])
                    marker = "←" if idx.item() == next_token else " "
                    print(f"    {i+1}. [{idx.item():5d}] \"{token_str:20s}\" p={prob:.4f} {marker}")
                print()

            # Add to sequence
            generated_tokens.append(next_token)
            current_tokens.append(next_token)

            # Stop at end token
            if next_token == self.tokenizer.encode_special("<|assistant_end|>"):
                break

        # Print full generation
        print(f"{'-'*100}")
        print(f"\nFull generation:")
        print(f"\"{self.tokenizer.decode(generated_tokens)}\"")
        print()

    def compare_temperatures(self, prompt: str, temperatures: list = [0.1, 0.5, 0.9, 1.2, 1.5],
                           max_tokens: int = 50, num_samples: int = 3):
        """
        Compare outputs at different temperatures.

        Args:
            prompt: Input prompt
            temperatures: List of temperatures to try
            max_tokens: Maximum tokens per generation
            num_samples: Number of samples per temperature
        """
        print(f"\n{'='*100}")
        print(f"TEMPERATURE COMPARISON")
        print(f"{'='*100}\n")
        print(f"Prompt: \"{prompt}\"\n")

        prompt_tokens = self.tokenizer.encode(prompt, prepend="<|bos|>")

        for temp in temperatures:
            print(f"\n{'-'*100}")
            print(f"Temperature: {temp}")
            print(f"{'-'*100}\n")

            for i in range(num_samples):
                samples, _ = self.engine.generate_batch(
                    prompt_tokens,
                    num_samples=1,
                    max_tokens=max_tokens,
                    temperature=temp,
                    seed=42 + i
                )

                output = self.tokenizer.decode(samples[0])
                print(f"  Sample {i+1}: {output}")

            print()

    def compare_top_k(self, prompt: str, top_k_values: list = [10, 50, 100, None],
                     max_tokens: int = 50):
        """
        Compare outputs with different top-k values.

        Args:
            prompt: Input prompt
            top_k_values: List of top-k values to try (None = no filtering)
            max_tokens: Maximum tokens per generation
        """
        print(f"\n{'='*100}")
        print(f"TOP-K COMPARISON")
        print(f"{'='*100}\n")
        print(f"Prompt: \"{prompt}\"\n")

        prompt_tokens = self.tokenizer.encode(prompt, prepend="<|bos|>")

        for k in top_k_values:
            print(f"\n{'-'*100}")
            print(f"Top-k: {k if k is not None else 'None (no filtering)'}")
            print(f"{'-'*100}\n")

            samples, _ = self.engine.generate_batch(
                prompt_tokens,
                num_samples=1,
                max_tokens=max_tokens,
                temperature=0.9,
                top_k=k,
                seed=42
            )

            output = self.tokenizer.decode(samples[0])
            print(f"  Output: {output}\n")

    def interactive_mode(self):
        """Interactive mode for exploring generation parameters."""
        print(f"\n{'='*100}")
        print(f"GENERATION PARAMETER EXPLORER - Interactive Mode")
        print(f"{'='*100}\n")
        print("Commands:")
        print("  - Type a prompt to generate with current settings")
        print("  - 'temp <value>' to set temperature (e.g., 'temp 0.9')")
        print("  - 'topk <value>' to set top-k (e.g., 'topk 50')")
        print("  - 'probs' to toggle probability display")
        print("  - 'compare-temp <prompt>' to compare temperatures")
        print("  - 'quit' to exit\n")

        # Default settings
        temperature = 0.9
        top_k = None
        show_probs = False

        while True:
            try:
                print(f"Current settings: temperature={temperature}, top_k={top_k}, show_probs={show_probs}")
                user_input = input("\n> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if user_input.lower().startswith('temp '):
                    temperature = float(user_input.split()[1])
                    print(f"Set temperature to {temperature}")
                    continue

                if user_input.lower().startswith('topk '):
                    value = user_input.split()[1]
                    top_k = None if value.lower() == 'none' else int(value)
                    print(f"Set top-k to {top_k}")
                    continue

                if user_input.lower() == 'probs':
                    show_probs = not show_probs
                    print(f"Probability display: {'ON' if show_probs else 'OFF'}")
                    continue

                if user_input.lower().startswith('compare-temp '):
                    prompt = user_input[13:].strip()
                    self.compare_temperatures(prompt)
                    continue

                # Default: generate with current settings
                self.sample_with_probabilities(
                    user_input,
                    temperature=temperature,
                    top_k=top_k,
                    show_probs=show_probs
                )

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generation Parameter Explorer")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", help="Prompt to use")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature")
    parser.add_argument("--top-k", type=int, help="Top-k filtering")
    parser.add_argument("--compare-temp", action="store_true", help="Compare temperatures")
    parser.add_argument("--compare-topk", action="store_true", help="Compare top-k values")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--show-probs", action="store_true", help="Show probabilities")

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint)
    model.eval()

    print("Loading tokenizer...")
    tokenizer = get_tokenizer()

    explorer = GenerationExplorer(model, tokenizer)

    if args.interactive:
        explorer.interactive_mode()
    elif args.compare_temp and args.prompt:
        explorer.compare_temperatures(args.prompt)
    elif args.compare_topk and args.prompt:
        explorer.compare_top_k(args.prompt)
    elif args.prompt:
        explorer.sample_with_probabilities(
            args.prompt,
            temperature=args.temperature,
            top_k=args.top_k,
            show_probs=args.show_probs
        )
    else:
        print("Please provide --prompt or use --interactive mode")


if __name__ == "__main__":
    main()
```

#### Usage Examples

```bash
# Interactive mode
python tools/generation_explorer.py --checkpoint out/chat_checkpoints/d20_sft --interactive

# Compare temperatures
python tools/generation_explorer.py --checkpoint out/chat_checkpoints/d20_sft \
    --prompt "The capital of France is" --compare-temp

# Show probabilities
python tools/generation_explorer.py --checkpoint out/chat_checkpoints/d20_sft \
    --prompt "What is 2+2?" --show-probs
```

### Learning Outcomes

- ✅ Understanding sampling strategies
- ✅ Working with probability distributions
- ✅ Tensor operations in PyTorch
- ✅ Interactive CLI design
- ✅ Model inference best practices

---

## Feature 7: Training Resume Helper

### Why This Feature is Useful

**Problem it solves:**
- Training crashes and you lose all progress
- Don't know which checkpoint to resume from
- Learning rate is wrong when resuming
- Hard to calculate remaining steps

**Learning benefits:**
- **Checkpoint management**: Understand how to save/load state
- **Training dynamics**: Learn about learning rate schedules
- **Error recovery**: Handle training interruptions gracefully

**Practical benefits:**
- Never lose training progress
- Automatically resume from best checkpoint
- Calculate remaining training time
- Adjust hyperparameters for resume

### Implementation Details

#### New Files to Create

**File 1: `tools/training_resume_helper.py`**

```python
"""
Training Resume Helper

Automatically detect and resume interrupted training.
"""

import os
import argparse
import torch
from pathlib import Path
from typing import Optional, Dict
from nanochat.checkpoint_manager import load_checkpoint


class TrainingResumeHelper:
    """Helper for resuming interrupted training."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir

    def find_latest_checkpoint(self) -> Optional[str]:
        """
        Find the most recent checkpoint in the directory.

        Returns:
            Path to latest checkpoint, or None if not found
        """
        checkpoint_file = os.path.join(self.checkpoint_dir, "checkpoint.pt")

        if os.path.exists(checkpoint_file):
            return checkpoint_file

        return None

    def load_checkpoint_info(self, checkpoint_path: str) -> Dict:
        """
        Load checkpoint metadata without loading full model.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Dictionary with checkpoint information
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        metadata = checkpoint.get('metadata', {})

        return {
            'step': metadata.get('step', 0),
            'val_bpb': metadata.get('val_bpb', None),
            'model_config': metadata.get('model_config', {}),
            'user_config': metadata.get('user_config', {}),
            'device_batch_size': metadata.get('device_batch_size', None),
            'max_seq_len': metadata.get('max_seq_len', None),
        }

    def verify_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Verify checkpoint integrity.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            True if checkpoint is valid
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Check required fields
            required_fields = ['model', 'metadata']
            for field in required_fields:
                if field not in checkpoint:
                    print(f"✗ Missing required field: {field}")
                    return False

            # Check model state dict
            model_state = checkpoint['model']
            if not isinstance(model_state, dict) or len(model_state) == 0:
                print("✗ Invalid or empty model state")
                return False

            print("✓ Checkpoint is valid")
            return True

        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            return False

    def calculate_resume_params(self, checkpoint_info: Dict, target_steps: int) -> Dict:
        """
        Calculate parameters for resuming training.

        Args:
            checkpoint_info: Info from checkpoint
            target_steps: Total target training steps

        Returns:
            Dictionary with resume parameters
        """
        current_step = checkpoint_info['step']
        remaining_steps = target_steps - current_step

        # Calculate progress percentage
        progress_pct = 100.0 * current_step / target_steps if target_steps > 0 else 0

        # Suggest learning rate (typically continue with warmdown if near end)
        warmdown_threshold = 0.8  # Last 20% of training
        in_warmdown = progress_pct >= (warmdown_threshold * 100)

        return {
            'current_step': current_step,
            'remaining_steps': remaining_steps,
            'progress_pct': progress_pct,
            'in_warmdown': in_warmdown,
            'target_steps': target_steps
        }

    def print_resume_report(self, checkpoint_path: str, target_steps: int = None):
        """
        Print a report about resuming from this checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
            target_steps: Target total steps (optional)
        """
        print(f"\n{'='*80}")
        print(f"TRAINING RESUME REPORT")
        print(f"{'='*80}\n")

        # Load checkpoint info
        info = self.load_checkpoint_info(checkpoint_path)

        print(f"Checkpoint: {checkpoint_path}")
        print(f"Last saved step: {info['step']}")
        print(f"Validation BPB: {info['val_bpb']:.4f}" if info['val_bpb'] else "Validation BPB: N/A")

        # Model config
        model_config = info['model_config']
        if model_config:
            print(f"\nModel Configuration:")
            print(f"  Layers: {model_config.get('n_layer', 'N/A')}")
            print(f"  Hidden dim: {model_config.get('n_embd', 'N/A')}")
            print(f"  Sequence length: {model_config.get('sequence_len', 'N/A')}")

        # Training config
        if info['user_config']:
            print(f"\nTraining Configuration:")
            for key in ['device_batch_size', 'total_batch_size', 'learning_rate']:
                if key in info['user_config']:
                    print(f"  {key}: {info['user_config'][key]}")

        # Resume parameters
        if target_steps:
            resume_params = self.calculate_resume_params(info, target_steps)

            print(f"\nResume Parameters:")
            print(f"  Current step: {resume_params['current_step']}")
            print(f"  Target steps: {resume_params['target_steps']}")
            print(f"  Remaining: {resume_params['remaining_steps']}")
            print(f"  Progress: {resume_params['progress_pct']:.1f}%")
            print(f"  In warmdown: {'Yes' if resume_params['in_warmdown'] else 'No'}")

        print(f"\n{'='*80}\n")

    def generate_resume_command(self, checkpoint_path: str, script: str = "base_train.py") -> str:
        """
        Generate command to resume training.

        Args:
            checkpoint_path: Path to checkpoint
            script: Training script name

        Returns:
            Command string to resume training
        """
        info = self.load_checkpoint_info(checkpoint_path)

        # Build command
        cmd_parts = [
            f"python -m scripts.{script}",
            f"--resume_from={checkpoint_path}",
        ]

        # Add key parameters from checkpoint
        if info['device_batch_size']:
            cmd_parts.append(f"--device_batch_size={info['device_batch_size']}")

        return " \\\n    ".join(cmd_parts)


def main():
    parser = argparse.ArgumentParser(description="Training Resume Helper")
    parser.add_argument("checkpoint_dir", help="Checkpoint directory")
    parser.add_argument("--target-steps", type=int, help="Target total training steps")
    parser.add_argument("--verify", action="store_true", help="Verify checkpoint integrity")
    parser.add_argument("--command", action="store_true", help="Generate resume command")

    args = parser.parse_args()

    helper = TrainingResumeHelper(args.checkpoint_dir)

    # Find latest checkpoint
    checkpoint_path = helper.find_latest_checkpoint()

    if not checkpoint_path:
        print(f"No checkpoint found in {args.checkpoint_dir}")
        return

    if args.verify:
        helper.verify_checkpoint(checkpoint_path)
    elif args.command:
        cmd = helper.generate_resume_command(checkpoint_path)
        print("\nResume command:\n")
        print(cmd)
        print()
    else:
        helper.print_resume_report(checkpoint_path, args.target_steps)


if __name__ == "__main__":
    main()
```

#### Files to Modify

**File: `scripts/base_train.py`** (Add resume capability)

Add near the top:
```python
# Add resume_from argument
resume_from = "" # path to checkpoint to resume from
```

Add after model initialization:
```python
# Resume from checkpoint if specified
if resume_from and os.path.exists(resume_from):
    print0(f"Resuming from checkpoint: {resume_from}")
    checkpoint = torch.load(resume_from, map_location=device)

    # Load model state
    orig_model.load_state_dict(checkpoint['model'])

    # Load optimizer states
    if 'optimizers' in checkpoint:
        for opt, opt_state in zip(optimizers, checkpoint['optimizers']):
            opt.load_state_dict(opt_state)

    # Get starting step
    start_step = checkpoint['metadata'].get('step', 0) + 1
    print0(f"Resuming from step {start_step}")
else:
    start_step = 0

# Adjust training loop
for step in range(start_step, num_iterations + 1):
    # ... rest of training loop
```

#### Usage Examples

```bash
# Check checkpoint and calculate remaining work
python tools/training_resume_helper.py out/base_checkpoints/d20 --target-steps 5400

# Verify checkpoint integrity
python tools/training_resume_helper.py out/base_checkpoints/d20 --verify

# Generate resume command
python tools/training_resume_helper.py out/base_checkpoints/d20 --command
```

### Learning Outcomes

- ✅ Checkpoint save/load mechanics
- ✅ Training state management
- ✅ Error handling and recovery
- ✅ Command generation
- ✅ File system operations

---

## Summary of All 10 Features

### Quick Reference Table

| # | Feature | Status | Difficulty | Implementation Time | Key Learning |
|---|---------|--------|------------|---------------------|--------------|
| 1 | Tokenizer Playground | ✅ **DONE** | ⭐ | 2-3 hours | Tokenization, text processing |
| 2 | Training Dashboard | 📝 Planned | ⭐⭐ | 4-6 hours | Visualization, real-time monitoring |
| 3 | Checkpoint Browser | 📝 Planned | ⭐ | 2-3 hours | File management, metadata |
| 4 | Dataset Inspector | 📝 Planned | ⭐ | 2-4 hours | Data validation, statistics |
| 5 | Model Calculator | ✅ **DONE** | ⭐ | 1-2 hours | Parameter counting, estimation |
| 6 | Generation Explorer | 📝 Planned | ⭐⭐ | 3-5 hours | Sampling, probabilities |
| 7 | Training Resume | 📝 Planned | ⭐ | 2-3 hours | Checkpointing, state management |
| 8 | Attention Visualizer | 📝 Planned | ⭐⭐ | 4-6 hours | Attention mechanics, visualization |
| 9 | LR Finder | 📝 Planned | ⭐⭐ | 3-4 hours | Optimization, hyperparameter tuning |
| 10 | Conversation Builder | 📝 Planned | ⭐⭐ | 3-5 hours | Data creation, UX design |

### Recommended Implementation Order

For beginners, implement in this order:

1. ✅ **#5 (Model Calculator)** - Already implemented! Use it to plan your experiments
2. ✅ **#1 (Tokenizer Playground)** - Already implemented! Great for understanding LLMs
3. **#4 (Dataset Inspector)** - Important before training (good next choice!)
4. **#7 (Training Resume)** - Saves time when training crashes
5. **#3 (Checkpoint Browser)** - Useful day-to-day for managing models
6. **#2 (Training Dashboard)** - Makes training more visible
7. **#6 (Generation Explorer)** - Fun and educational
8. **Rest as needed** - Based on your interests

### Total Implementation Time

- **All 10 features**: 26-41 hours (✅ 2 done, ~8 remaining = 23-36 hours)
- **Top 5 essentials (#1, #2, #3, #4, #5)**: 11-18 hours (✅ 2 done, 3 remaining = 8-13 hours)
- **Quick wins (#1, #3, #5)**: 5-8 hours (✅ 2 done, #3 remaining = 2-3 hours)

---

## Next Steps

1. **Choose a feature** from the list that interests you
2. **Read the full documentation** for that feature
3. **Create the new files** as specified
4. **Test the feature** with the provided examples
5. **Iterate and improve** based on your needs

**Note:** Features 8, 9, and 10 are not included in this document due to length. They are more advanced and follow similar patterns to the features documented here. If you'd like detailed documentation for those features, let me know!

---

**Ready to implement?** Pick a feature and start coding! Each one will teach you something new about LLMs and software engineering.