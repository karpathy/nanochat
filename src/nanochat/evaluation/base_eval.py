"""
Unified evaluation script for base models.

Supports three evaluation modes (comma-separated):
  --eval core    : CORE metric (accuracy on ICL tasks)
  --eval bpb     : Bits per byte on train/val splits
  --eval sample  : Generate samples from the model

Default is all three: --eval core,bpb,sample

Examples:

    # Evaluate a HuggingFace model (e.g. GPT-2 124M) using 8 GPUs
    torchrun --nproc_per_node=8 -m nanochat.scripts.base_eval --hf-path openai-community/gpt2

    # Evaluate a nanochat model (e.g. d24) using 8 GPUs
    torchrun --nproc_per_node=8 -m nanochat.scripts.base_eval --model-tag d24 --device-batch-size=16

    # Quick/approximate evaluation using a single GPU
    python -m nanochat.scripts.base_eval --model-tag d24 --device-batch-size=16 --max-per-task=100 --split-tokens=524288
"""

import os
from typing import cast

from nanochat.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    eval_results_dir,
    print0,
)
from nanochat.config import Config
from nanochat.evaluation.core_benchmark import evaluate_core
from nanochat.evaluation.engine import Engine
from nanochat.evaluation.hf_model import get_hf_token_bytes, load_hf_model
from nanochat.evaluation.loss_eval import evaluate_bpb
from nanochat.report import get_report
from nanochat.tokenizer import get_token_bytes
from nanochat.training.checkpoint import load_model_from_dir
from nanochat.training.dataloader import tokenizing_distributed_data_loader_bos_bestfit

# -----------------------------------------------------------------------------
# base_eval


def base_eval(config: Config):
    base_dir = config.common.base_dir

    # Parse evaluation modes
    eval_modes = set(mode.strip() for mode in config.evaluation.modes.split(","))

    # Distributed / precision setup
    device_type = autodetect_device_type() if config.common.device_type == "" else config.common.device_type
    _, ddp_rank, _, ddp_world_size, device = compute_init(device_type)
    # Load model and tokenizer
    is_hf_model = config.evaluation.hf_path is not None
    if is_hf_model:
        model, tokenizer = load_hf_model(config.evaluation.hf_path, device)
        sequence_len = model.max_seq_len or 1024
        token_bytes = get_hf_token_bytes(tokenizer, device=device)
        model_name = config.evaluation.hf_path
        model_slug = config.evaluation.hf_path.replace("/", "-")
    else:
        model, tokenizer, meta = load_model_from_dir(
            base_dir=base_dir,
            phase="base",
            device=device,
            model_tag=config.evaluation.model_tag,
            step=config.evaluation.step,
        )
        sequence_len = cast(int, cast(dict[str, object], meta["model_config"])["sequence_len"])
        token_bytes = get_token_bytes(base_dir=base_dir, device=device)
        model_name = f"base_model (step {cast(int, meta['step'])})"
        model_slug = f"base_model_{cast(int, meta['step']):06d}"

    print0(f"Evaluating model: {model_name}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    # Results to log
    core_results = None
    bpb_results = {}
    samples = []
    unconditioned_samples = []

    # --- Sampling ---
    if "sample" in eval_modes and not is_hf_model:
        print0("\n" + "=" * 80)
        print0("Model Samples")
        print0("=" * 80)
        if ddp_rank == 0:
            prompts = [
                "The capital of France is",
                "The chemical symbol of gold is",
                "If yesterday was Friday, then tomorrow will be",
                "The opposite of hot is",
                "The planets of the solar system are:",
                "My favorite color is",
                "If 5*x + 3 = 13, then x is",
            ]
            engine = Engine(model, tokenizer)
            print0("\nConditioned samples:")
            for prompt in prompts:
                tokens = tokenizer(prompt, prepend="<|bos|>")
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                sample_str = tokenizer.decode(sample[0])
                print0("-" * 80)
                print0(sample_str)
                samples.append(sample_str)

            print0("\nUnconditioned samples:")
            tokens = tokenizer("", prepend="<|bos|>")
            uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, temperature=1.0)
            for sample in uncond:
                sample_str = tokenizer.decode(sample)
                print0("-" * 80)
                print0(sample_str)
                unconditioned_samples.append(sample_str)
    elif "sample" in eval_modes and is_hf_model:
        print0("\nSkipping sampling for HuggingFace models (not supported)")

    # --- BPB evaluation ---
    if "bpb" in eval_modes:
        print0("\n" + "=" * 80)
        print0("BPB Evaluation")
        print0("=" * 80)
        tokens_per_step = config.evaluation.device_batch_size * sequence_len * ddp_world_size
        if config.evaluation.split_tokens % tokens_per_step != 0:
            # Adjust to nearest multiple
            config.evaluation.split_tokens = (config.evaluation.split_tokens // tokens_per_step) * tokens_per_step
            print0(
                f"Adjusted split_tokens to {config.evaluation.split_tokens} (must be divisible by {tokens_per_step})"
            )
        steps = config.evaluation.split_tokens // tokens_per_step

        for split_name in ["train", "val"]:
            loader = tokenizing_distributed_data_loader_bos_bestfit(
                base_dir, tokenizer, config.evaluation.device_batch_size, sequence_len, split_name, device=device
            )
            bpb = evaluate_bpb(model, loader, steps, token_bytes)
            bpb_results[split_name] = bpb
            print0(f"{split_name} bpb: {bpb:.6f}")

    # --- CORE evaluation ---
    if "core" in eval_modes:
        print0("\n" + "=" * 80)
        print0("CORE Evaluation")
        print0("=" * 80)
        core_results = evaluate_core(
            base_dir=base_dir,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_per_task=config.evaluation.max_per_task,
        )

        # Write CSV output
        if ddp_rank == 0:
            output_csv_path = os.path.join(eval_results_dir(base_dir), f"{model_slug}.csv")
            with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in cast(dict[str, object], core_results["results"]):
                    acc = cast(dict[str, object], core_results["results"])[label]
                    centered = cast(dict[str, object], core_results["centered_results"])[label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
                f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")
            print0(f"\nResults written to: {output_csv_path}")
            print0(f"CORE metric: {core_results['core_metric']:.4f}")

    # --- Log to report ---

    report_data = [{"model": model_name}]

    if core_results:
        report_data[0]["CORE metric"] = core_results["core_metric"]
        report_data.append(core_results["centered_results"])

    if bpb_results:
        report_data[0]["train bpb"] = bpb_results.get("train")
        report_data[0]["val bpb"] = bpb_results.get("val")

    if samples:
        report_data.append({f"sample {i}": s for i, s in enumerate(samples)})
    if unconditioned_samples:
        report_data.append({f"unconditioned {i}": s for i, s in enumerate(unconditioned_samples)})

    get_report(base_dir=base_dir).log(section="Base model evaluation", data=report_data)

    compute_cleanup()
