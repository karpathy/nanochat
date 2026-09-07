"""
Evaluate a base model: the CORE metric (DCLM's 22-task ensemble, the number the
curve is judged on) and a few greedy samples to look at.

python -m scripts.base_eval -g d12
torchrun --nproc_per_node=8 -m scripts.base_eval -- -g d12
"""

import argparse

from nanochat.common import print0
from harness.experiment import format_record, format_invocation
from harness.runtime import compute_init, compute_cleanup, autodetect_device_type
from harness.checkpoint import load_model, find_largest_model
from evals.core import evaluate_core
from nanochat.engine import Engine

PROMPTS = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Base model evaluation: CORE and samples")
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load (default: the largest base model)')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load (default: the last)')
    parser.add_argument('-x', '--max-per-task', type=int, default=-1, help='Examples per CORE task (-1 = all; a debug knob)')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='cuda|cpu|mps (empty = autodetect)')
    args = parser.parse_args()
    print0(format_invocation(args))

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    model_tag = args.model_tag if args.model_tag is not None else find_largest_model("base")
    model, tokenizer, meta = load_model("base", device, model_tag=model_tag, step=args.step)

    # samples: greedy continuations of a few prompts, then free-running ones (rank 0 only)
    if ddp_rank == 0:
        engine = Engine(model, tokenizer)
        for prompt in PROMPTS:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            sample = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        tokens = tokenizer("", prepend="<|bos|>")
        for sample in engine.generate_batch(tokens, num_samples=4, max_tokens=64, temperature=1.0):
            print0("-" * 80)
            print0(tokenizer.decode(sample))

    # CORE: one `task` record per task, the aggregate in the summary
    core = evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task)
    for label, accuracy in core["results"].items():
        centered = core["centered_results"][label]
        print0(format_record("task", task=label, accuracy=round(accuracy, 6), centered=round(centered, 6)))
    print0(f"CORE metric: {core['core_metric']:.4f}")

    # the stage record (see harness/experiment.py); this is what downstream tooling consumes
    summary = {"model_tag": model_tag, "step": meta["step"], "core": round(core["core_metric"], 6)}
    print0(format_record("summary", **summary))

    compute_cleanup()
