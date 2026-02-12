"""
Unified evaluation script for base models.

Supports three evaluation modes (comma-separated):
  --eval core    : CORE metric (accuracy on ICL tasks)
  --eval bpb     : Bits per byte on train/val splits
  --eval sample  : Generate samples from the model

Default is all three: --eval core,bpb,sample

Examples:

    # Evaluate a HuggingFace model (e.g. GPT-2 124M) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_eval --hf-path openai-community/gpt2

    # Evaluate a nanochat model (e.g. d24) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_eval --model-tag d24 --device-batch-size=16

    # Quick/approximate evaluation using a single GPU
    python -m scripts.base_eval --model-tag d24 --device-batch-size=16 --max-per-task=100 --split-tokens=524288
"""
import os
import csv
import time
import json
import yaml
import shutil
import random
import hashlib
import zipfile
import tempfile
import argparse
from contextlib import nullcontext

import torch

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock
from nanochat.tokenizer import HuggingFaceTokenizer, get_token_bytes
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task, prepare_task_data
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine

# -----------------------------------------------------------------------------
# HuggingFace loading utilities

class ModelWrapper:
    """Lightweight wrapper to give HuggingFace models a nanochat-compatible interface."""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction='mean'):
        logits = self.model(input_ids).logits
        if targets is None:
            return logits
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction
        )
        return loss

    def get_device(self):
        return next(self.model.parameters()).device


def load_hf_model(hf_path: str, device):
    """Load a HuggingFace model and tokenizer."""
    print0(f"Loading HuggingFace model from: {hf_path}")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer


def get_hf_token_bytes(tokenizer, device="cpu"):
    """Compute token_bytes tensor for a HuggingFace tokenizer."""
    vocab_size = tokenizer.tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    for token_id in range(vocab_size):
        token_str = tokenizer.tokenizer.decode([token_id])
        token_bytes[token_id] = len(token_str.encode('utf-8'))
    return token_bytes

# -----------------------------------------------------------------------------
# CORE evaluation

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def place_eval_bundle(file_path):
    """Unzip eval_bundle.zip and place it in the base directory."""
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")


_eval_data_cache = None  # (task_inputs, random_baselines, w_label, w_shot, w_type)
_batch_cache = {}        # {label: collated_batches} — cached after first run
_batch_cache_key = None  # (max_per_task, max_seq_len) — invalidate if these change
_prev_centered = {}      # {label: centered_result} — previous run for delta display
_prev_core = None        # previous core_metric


def _get_disk_cache_dir(max_per_task):
    """Get disk cache dir for base-4 collated batches, keyed by tokenizer file hash.
    Returns None if no local tokenizer file is found (e.g. HuggingFace models)."""
    base_dir = get_base_dir()
    for fname in ("tokenizer.pkl", "tokenizer.json"):
        path = os.path.join(base_dir, "tokenizer", fname)
        if os.path.exists(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            tok_hash = h.hexdigest()[:16]
            return os.path.join(base_dir, "core_token_cache", f"{tok_hash}_n{max_per_task}")
    return None


def evaluate_model(model, tokenizer, device, max_per_task=-1):
    """
    Evaluate a base model on the CORE benchmark.
    - max_per_task: crop the data to this many examples per task for testing (-1 = disable)
    Collated batches are cached across calls since the tokenizer is fixed.
    Second+ runs skip prepare and collation entirely — just GPU forward passes.
    """
    global _eval_data_cache, _batch_cache, _batch_cache_key, _prev_centered, _prev_core
    from concurrent.futures import ThreadPoolExecutor

    max_seq_len = getattr(model, 'max_seq_len', None)
    cache_key = (max_per_task, max_seq_len)

    # Invalidate batch cache if parameters changed
    if cache_key != _batch_cache_key:
        _batch_cache.clear()
        _batch_cache_key = cache_key

    # Load and cache task data + baselines (only read from disk once)
    if _eval_data_cache is None:
        base_dir = get_base_dir()
        eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
        if not os.path.exists(eval_bundle_dir):
            download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
        config_path = os.path.join(eval_bundle_dir, "core.yaml")
        data_base_path = os.path.join(eval_bundle_dir, "eval_data")
        eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        tasks = config['icl_tasks']

        random_baselines = {}
        with open(eval_meta_data, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                random_baselines[row['Eval Task']] = float(row['Random baseline'])

        task_inputs = []
        for task in tasks:
            label = task['label']
            task_meta = {
                'task_type': task['icl_task_type'],
                'dataset_uri': task['dataset_uri'],
                'num_fewshot': task['num_fewshot'][0],
                'continuation_delimiter': task.get('continuation_delimiter', ' ')
            }
            data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
            with open(data_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line.strip()) for line in f]
            shuffle_rng = random.Random(1337)
            shuffle_rng.shuffle(data)
            if max_per_task > 0:
                data = data[:max_per_task]
            task_inputs.append((label, task_meta, data))

        w_label = max(len(t[0]) for t in task_inputs)
        w_shot = max(len(f"{t[1]['num_fewshot']}-shot") for t in task_inputs)
        w_type = max(len(t[1]['task_type']) for t in task_inputs)
        _eval_data_cache = (task_inputs, random_baselines, w_label, w_shot, w_type)

    task_inputs, random_baselines, w_label, w_shot, w_type = _eval_data_cache

    # First run: eagerly prepare next task while evaluating current, cache collated batches.
    # Cached runs: pass collated batches directly — no threads, no prepare, no collation.
    results = {}
    centered_results = {}
    cached_run = all(label in _batch_cache for label, _, _ in task_inputs)
    disk_cache_dir = _get_disk_cache_dir(max_per_task)

    # Try loading from disk cache if in-memory cache is empty
    if not cached_run and disk_cache_dir is not None:
        all_on_disk = os.path.isdir(disk_cache_dir) and all(
            os.path.exists(os.path.join(disk_cache_dir, f"{label}.pt"))
            for label, _, _ in task_inputs
        )
        if all_on_disk:
            for label, _, _ in task_inputs:
                d = torch.load(os.path.join(disk_cache_dir, f"{label}.pt"), weights_only=False)
                _batch_cache[label] = d['collated']
            cached_run = True
            print0("  (loaded collated batches from disk cache)")

    first_run = not cached_run  # track whether we did prepare+collate (for disk save)

    if not cached_run:
        executor = ThreadPoolExecutor(max_workers=1)
        first_uncached = next(i for i, (l, _, _) in enumerate(task_inputs) if l not in _batch_cache)
        _, first_meta, first_data = task_inputs[first_uncached]
        next_future = executor.submit(prepare_task_data, tokenizer, first_data, first_meta, max_seq_len)

    for i, (label, task_meta, data) in enumerate(task_inputs):
        shot_str = f"{task_meta['num_fewshot']}-shot"
        prefix = f"  {label:<{w_label}}  {shot_str:<{w_shot}}  {task_meta['task_type']:<{w_type}}"
        print0(f"{prefix}  ...", end="", flush=True)
        t0 = time.time()

        if label in _batch_cache:
            accuracy, collated = evaluate_task(model, data, device, collated=_batch_cache[label])
        else:
            prepared = next_future.result()
            # Kick off prepare for the next uncached task
            for j in range(i + 1, len(task_inputs)):
                next_label, next_meta, next_data = task_inputs[j]
                if next_label not in _batch_cache:
                    next_future = executor.submit(prepare_task_data, tokenizer, next_data, next_meta, max_seq_len)
                    break
            accuracy, collated = evaluate_task(model, data, device, prepared=prepared)
            _batch_cache[label] = collated

        elapsed = time.time() - t0
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        results[label] = accuracy
        centered_results[label] = centered_result
        delta_str = ""
        if label in _prev_centered:
            d = centered_result - _prev_centered[label]
            arrow = "\u2191" if d > 0 else "\u2193" if d < 0 else "="
            delta_str = f"  {arrow}{d:+.4f}"
        print0(f"\r{prefix}  acc: {accuracy:.4f}  centered: {centered_result:>7.4f}{delta_str}  time: {elapsed:.2f}s")

    if not cached_run:
        executor.shutdown(wait=False)

    # Save collated batches to disk after first run (so bench/future runs skip prepare+collate)
    if first_run and disk_cache_dir is not None:
        pad_id = tokenizer.get_bos_token_id()
        os.makedirs(disk_cache_dir, exist_ok=True)
        for label, _, _ in task_inputs:
            if label in _batch_cache:
                torch.save({'collated': _batch_cache[label], 'pad_token_id': pad_id},
                           os.path.join(disk_cache_dir, f"{label}.pt"))
        print0(f"  (saved collated batches to {disk_cache_dir})")

    core_metric = sum(centered_results.values()) / len(centered_results)
    if _prev_core is not None:
        d = core_metric - _prev_core
        arrow = "\u2191" if d > 0 else "\u2193" if d < 0 else "="
        print0(f"CORE: {core_metric:.4f}  {arrow}{d:+.4f}")
    else:
        print0(f"CORE: {core_metric:.4f}")
    _prev_centered = dict(centered_results)
    _prev_core = core_metric
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out

# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument('--eval', type=str, default='core,bpb,sample', help='Comma-separated evaluations to run: core,bpb,sample (default: all)')
    parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace model path (e.g. openai-community/gpt2-xl)')
    parser.add_argument('--model-tag', type=str, default=None, help='nanochat model tag to identify the checkpoint directory')
    parser.add_argument('--step', type=int, default=None, help='Model step to load (default = last)')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per CORE task (-1 = all)')
    parser.add_argument('--device-batch-size', type=int, default=32, help='Per-device batch size for BPB evaluation')
    parser.add_argument('--split-tokens', type=int, default=40*524288, help='Number of tokens to evaluate per split for BPB')
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (empty = autodetect)')
    args = parser.parse_args()

    # Parse evaluation modes
    eval_modes = set(mode.strip() for mode in args.eval.split(','))
    valid_modes = {'core', 'bpb', 'sample'}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    # Distributed / precision setup
    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # Load model and tokenizer
    is_hf_model = args.hf_path is not None
    if is_hf_model:
        model, tokenizer = load_hf_model(args.hf_path, device)
        sequence_len = model.max_seq_len or 1024
        token_bytes = get_hf_token_bytes(tokenizer, device=device)
        model_name = args.hf_path
        model_slug = args.hf_path.replace("/", "-")
    else:
        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
        sequence_len = meta["model_config"]["sequence_len"]
        token_bytes = get_token_bytes(device=device)
        model_name = f"base_model (step {meta['step']})"
        model_slug = f"base_model_{meta['step']:06d}"

    print0(f"Evaluating model: {model_name}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    # Results to log
    core_results = None
    bpb_results = {}
    samples = []
    unconditioned_samples = []

    # --- Sampling ---
    if 'sample' in eval_modes and not is_hf_model:
        print0("\n" + "="*80)
        print0("Model Samples")
        print0("="*80)
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
                with autocast_ctx:
                    sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                sample_str = tokenizer.decode(sample[0])
                print0("-" * 80)
                print0(sample_str)
                samples.append(sample_str)

            print0("\nUnconditioned samples:")
            tokens = tokenizer("", prepend="<|bos|>")
            with autocast_ctx:
                uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, temperature=1.0)
            for sample in uncond:
                sample_str = tokenizer.decode(sample)
                print0("-" * 80)
                print0(sample_str)
                unconditioned_samples.append(sample_str)
    elif 'sample' in eval_modes and is_hf_model:
        print0("\nSkipping sampling for HuggingFace models (not supported)")

    # --- BPB evaluation ---
    if 'bpb' in eval_modes:
        print0("\n" + "="*80)
        print0("BPB Evaluation")
        print0("="*80)
        tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
        if args.split_tokens % tokens_per_step != 0:
            # Adjust to nearest multiple
            args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
            print0(f"Adjusted split_tokens to {args.split_tokens} (must be divisible by {tokens_per_step})")
        steps = args.split_tokens // tokens_per_step

        for split_name in ["train", "val"]:
            loader = tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, sequence_len, split_name, device=device)
            with autocast_ctx:
                bpb = evaluate_bpb(model, loader, steps, token_bytes)
            bpb_results[split_name] = bpb
            print0(f"{split_name} bpb: {bpb:.6f}")

    # --- CORE evaluation ---
    if 'core' in eval_modes:
        print0("\n" + "="*80)
        print0("CORE Evaluation")
        print0("="*80)
        with autocast_ctx:
            core_results = evaluate_model(model, tokenizer, device, max_per_task=args.max_per_task)

        # Write CSV output
        if ddp_rank == 0:
            base_dir = get_base_dir()
            output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in core_results["results"]:
                    acc = core_results["results"][label]
                    centered = core_results["centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
                f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")
            print0(f"\nResults written to: {output_csv_path}")
            print0(f"CORE metric: {core_results['core_metric']:.4f}")

    # --- Log to report ---
    from nanochat.report import get_report
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

    get_report().log(section="Base model evaluation", data=report_data)

    compute_cleanup()


if __name__ == "__main__":
    main()
