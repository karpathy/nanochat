"""
Sweep the IV guidance weight `w` across a set of evaluation tasks and report
accuracy vs w. Used to validate that clarinet's IV-conditioned inference
beats the conditional-only baseline on reasoning tasks at moderate w, and to
locate the over-guidance collapse point.

Loads the model once, then for each w in --weights reconstructs a
ClarinetEngine bound to that weight and runs the requested tasks via
chat_eval's helpers.

Example:
    python -m scripts.iv_eval -i sft -a GSM8K,ARC-Easy --weights 0,1,1.5,2,3,5
"""

import argparse

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, print0

from clarinet.engine import ClarinetEngine
from scripts.chat_eval import run_chat_eval


def parse_weights(spec):
    return [float(x) for x in spec.split(",") if x]


def make_engine(model, tokenizer, iv_weight, wald_scale):
    class _BoundClarinetEngine(ClarinetEngine):
        def generate(self, *args, **kwargs):
            kwargs.setdefault("iv_weight", iv_weight)
            kwargs.setdefault("wald_scale", wald_scale)
            yield from super().generate(*args, **kwargs)
    return _BoundClarinetEngine(model, tokenizer)


def main():
    parser = argparse.ArgumentParser(description="IV guidance weight sweep")
    parser.add_argument("-i", "--source", type=str, required=True, help="Model source: base|sft|rl")
    parser.add_argument("-a", "--task-names", type=str, default="GSM8K,ARC-Easy,ARC-Challenge,MMLU,HumanEval,SpellingBee",
                        help="Comma-separated task names from chat_eval's task registry.")
    parser.add_argument("--weights", type=str, default="0,0.5,1.0,1.5,2.0,3.0,5.0",
                        help="Comma-separated IV guidance weights to sweep.")
    parser.add_argument("--wald-scale", type=float, default=1.0)
    parser.add_argument("-t", "--temperature", type=float, default=0.0)
    parser.add_argument("-m", "--max-new-tokens", type=int, default=512)
    parser.add_argument("-n", "--num-samples", type=int, default=1)
    parser.add_argument("-k", "--top-k", type=int, default=50)
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    parser.add_argument("-g", "--model-tag", type=str, default=None)
    parser.add_argument("-s", "--step", type=int, default=None)
    parser.add_argument("-x", "--max-problems", type=int, default=None)
    parser.add_argument("--device-type", type=str, default="", choices=["", "cuda", "cpu", "mps"])
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _ddp, _rank, _local, _world, device = compute_init(device_type)

    model, tokenizer, _meta = load_model(args.source, device, phase="eval",
                                         model_tag=args.model_tag, step=args.step)

    weights = parse_weights(args.weights)
    task_names = [t for t in args.task_names.split(",") if t]

    # results[task][weight] = accuracy
    results = {t: {} for t in task_names}

    for w in weights:
        print0(f"\n========== IV weight w = {w} ==========")
        engine = make_engine(model, tokenizer, iv_weight=w, wald_scale=args.wald_scale)
        for task in task_names:
            acc = run_chat_eval(
                task,
                model, tokenizer, engine,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                max_problems=args.max_problems,
            )
            results[task][w] = acc
            print0(f"  w={w:.2f}  {task}: {100*acc:.2f}%")

    print0("\n========== Summary (rows=task, cols=w) ==========")
    header = "task".ljust(18) + "".join(f"{w:>9.2f}" for w in weights)
    print0(header)
    for task in task_names:
        row = task.ljust(18) + "".join(f"{100*results[task][w]:>8.2f}%" for w in weights)
        print0(row)

    compute_cleanup()


if __name__ == "__main__":
    main()
