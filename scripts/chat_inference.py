"""
Quick inference script: load base or SFT checkpoint and run on GSM8K test examples.

Usage:
    uv run python -m scripts.chat_inference --source=base --model-tag=nanochat-d12 --n=5
    uv run python -m scripts.chat_inference --source=sft  --model-tag=sft-baseline-d12 --step=15530 --n=5
"""
import argparse
import torch
from nanochat.common import autodetect_device_type, compute_init
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="sft", help="base|sft|rl")
parser.add_argument("--model-tag", type=str, required=True)
parser.add_argument("--step", type=int, default=None)
parser.add_argument("--n", type=int, default=5, help="number of GSM8K test examples")
parser.add_argument("--max-new-tokens", type=int, default=512)
args = parser.parse_args()

device_type = autodetect_device_type()
_, _, _, _, device = compute_init(device_type)
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else __import__("contextlib").nullcontext()

model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
model.eval()
engine = Engine(model, tokenizer)

task = GSM8K(subset="main", split="test")
correct = 0
for i in range(args.n):
    conversation = task[i]
    question = conversation["messages"][0]["content"]
    prefix_tokens = tokenizer.render_for_completion(conversation)
    prefix_length = len(prefix_tokens)
    with autocast_ctx:
        seqs, _ = engine.generate_batch(
            prefix_tokens,
            num_samples=1,
            max_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=1,
        )
    generated_text = tokenizer.decode(seqs[0][prefix_length:])
    is_correct = task.evaluate(conversation, generated_text)
    correct += int(is_correct)
    print(f"\n--- Example {i+1} ---")
    print(f"Q: {question[:300]}")
    print(f"Generated:\n{generated_text}")
    print(f"Correct: {bool(is_correct)}")

print(f"\n{'='*40}")
print(f"Accuracy: {correct}/{args.n}")
