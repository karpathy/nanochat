"""
Benchmark the CORE evaluation pipeline.

Compares three modes:
  1. Old sequential (per-example) evaluation
  2. New batched evaluation (first run — includes prepare + collate + forward)
  3. New batched evaluation (cached run — forward only)

Also sweeps batch_size and queue_size to find optimal hyperparameters.

Usage:
    python -m scripts.bench_core_eval
    python -m scripts.bench_core_eval --max-per-task 100        # quick test
    python -m scripts.bench_core_eval --hf-path openai-community/gpt2
"""
import os
import csv
import json
import time
import yaml
import random
import shutil
import zipfile
import tempfile
import argparse
from contextlib import nullcontext

import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock
from nanochat.tokenizer import HuggingFaceTokenizer
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import (
    forward_model, prepare_example, check_result, stack_sequences,
    prepare_task_data, _collate_batches, _forward_batches, evaluate_task,
    render_prompts_mc, render_prompts_schema, render_prompts_lm,
    batch_sequences_mc, batch_sequences_schema, batch_sequences_lm,
)

# ---- eval bundle loading (reused from base_eval) ----

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(file_path):
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        shutil.move(os.path.join(tmpdir, "eval_bundle"), eval_bundle_dir)
    print0(f"Placed eval_bundle at {eval_bundle_dir}")

def load_tasks(max_per_task=-1):
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    task_inputs = []
    for task in config['icl_tasks']:
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
    return task_inputs

# ---- old sequential evaluation (baseline) ----

@torch.no_grad()
def evaluate_example_old(idx, model, tokenizer, data, device, task_meta):
    """Original per-example sequential evaluation (the old code)."""
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])
                new_start_idxs.append(s - num_to_crop)
                new_end_idxs.append(e - num_to_crop)
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id).to(device)
    losses, predictions = forward_model(model, input_ids)
    return check_result(losses, predictions, input_ids, start_idxs, end_idxs, item.get('gold', None), task_type)


def evaluate_task_old(model, tokenizer, data, device, task_meta):
    """Original sequential evaluate_task."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example_old(idx, model, tokenizer, data, device, task_meta)
        correct[idx] = float(is_correct)
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    return correct.mean().item()

# ---- HuggingFace model wrapper ----

class ModelWrapper:
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len
    def __call__(self, input_ids):
        return self.model(input_ids).logits

def load_hf_model(hf_path, device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "gpt2" in hf_path else None
    return ModelWrapper(model, max_seq_len=max_seq_len), HuggingFaceTokenizer.from_pretrained(hf_path)

# ---- benchmark helpers ----

def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def bench_old(model, tokenizer, task_inputs, device):
    """Benchmark old sequential evaluation across all tasks."""
    sync_cuda()
    t0 = time.time()
    results = {}
    for label, task_meta, data in task_inputs:
        acc = evaluate_task_old(model, tokenizer, data, device, task_meta)
        results[label] = acc
    sync_cuda()
    return time.time() - t0, results


def bench_new_first(model, tokenizer, task_inputs, device, batch_size, queue_size):
    """Benchmark new batched evaluation (first run, no cache)."""
    sync_cuda()
    t0 = time.time()
    results = {}
    collated_cache = {}
    max_seq_len = getattr(model, 'max_seq_len', None)
    for label, task_meta, data in task_inputs:
        prepared = prepare_task_data(tokenizer, data, task_meta, max_seq_len)
        acc, collated = evaluate_task(model, data, device, batch_size=batch_size,
                                      queue_size=queue_size, prepared=prepared)
        results[label] = acc
        collated_cache[label] = collated
    sync_cuda()
    return time.time() - t0, results, collated_cache


def bench_new_cached(model, task_inputs, device, collated_cache):
    """Benchmark new batched evaluation (cached run, forward only)."""
    sync_cuda()
    t0 = time.time()
    results = {}
    for label, task_meta, data in task_inputs:
        acc, _ = evaluate_task(model, data, device, collated=collated_cache[label])
        results[label] = acc
    sync_cuda()
    return time.time() - t0, results


def verify_results(old_results, new_results, label="new"):
    """Check that old and new produce the same accuracies."""
    mismatches = []
    for task in old_results:
        if task in new_results and abs(old_results[task] - new_results[task]) > 1e-6:
            mismatches.append((task, old_results[task], new_results[task]))
    if mismatches:
        print0(f"  WARNING: {label} mismatches vs old:")
        for task, old, new in mismatches:
            print0(f"    {task}: old={old:.6f}  {label}={new:.6f}")
    else:
        print0(f"  {label} results match old (verified)")


# ---- main ----

def main():
    parser = argparse.ArgumentParser(description="Benchmark CORE eval pipeline")
    parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace model path')
    parser.add_argument('--model-tag', type=str, default=None, help='nanochat model tag')
    parser.add_argument('--step', type=int, default=None, help='Model step to load')
    parser.add_argument('--max-per-task', type=int, default=500, help='Max examples per task')
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps')
    parser.add_argument('--batch-sizes', type=str, default='1,2,4,8,16,32,64', help='Comma-separated batch sizes to sweep')
    parser.add_argument('--queue-sizes', type=str, default='2,4,8,16', help='Comma-separated queue sizes to sweep')
    parser.add_argument('--skip-old', action='store_true', help='Skip old sequential baseline (slow)')
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]
    queue_sizes = [int(x) for x in args.queue_sizes.split(',')]

    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # Load model
    if args.hf_path is not None:
        model, tokenizer = load_hf_model(args.hf_path, device)
        model_name = args.hf_path
    else:
        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
        model_name = f"base_model (step {meta['step']})"

    print0(f"Model: {model_name}")
    print0(f"Max per task: {args.max_per_task}")
    print0(f"Device: {device}")
    print0("")

    # Load tasks
    task_inputs = load_tasks(max_per_task=args.max_per_task)
    total_examples = sum(len(data) for _, _, data in task_inputs)
    print0(f"Loaded {len(task_inputs)} tasks, {total_examples} total examples")
    print0("")

    # ---- 1. Old sequential baseline ----
    old_results = None
    if not args.skip_old:
        print0("=" * 80)
        print0("OLD: Sequential per-example evaluation")
        print0("=" * 80)
        with autocast_ctx:
            old_time, old_results = bench_old(model, tokenizer, task_inputs, device)
        print0(f"  Time: {old_time:.2f}s  ({total_examples / old_time:.1f} examples/s)")
        print0("")

    # ---- 2. Sweep batch_size x queue_size for first run ----
    print0("=" * 80)
    print0("NEW: Batched evaluation — hyperparameter sweep (first run)")
    print0("=" * 80)
    print0("")

    # Header
    qs_header = "".join(f"{'q=' + str(q):>10}" for q in queue_sizes)
    print0(f"  {'batch_size':>10}{qs_header}")
    print0(f"  {'':>10}" + "-" * (10 * len(queue_sizes)))

    best_time = float('inf')
    best_params = None
    best_collated = None
    sweep_results = {}

    for bs in batch_sizes:
        row = f"  {bs:>10}"
        for qs in queue_sizes:
            with autocast_ctx:
                t, results, collated_cache = bench_new_first(model, tokenizer, task_inputs, device, bs, qs)
            sweep_results[(bs, qs)] = t
            row += f"{t:>9.2f}s"
            if t < best_time:
                best_time = t
                best_params = (bs, qs)
                best_collated = collated_cache
                best_first_results = results
        print0(row)

    print0("")
    print0(f"  Best: batch_size={best_params[0]}, queue_size={best_params[1]} -> {best_time:.2f}s  ({total_examples / best_time:.1f} examples/s)")

    # Verify correctness
    if old_results is not None:
        verify_results(old_results, best_first_results, label="new-first")
    print0("")

    # ---- 3. Cached run (forward only) ----
    print0("=" * 80)
    print0("NEW: Cached run (forward only, using best params)")
    print0("=" * 80)
    with autocast_ctx:
        cached_time, cached_results = bench_new_cached(model, task_inputs, device, best_collated)
    print0(f"  Time: {cached_time:.2f}s  ({total_examples / cached_time:.1f} examples/s)")
    if old_results is not None:
        verify_results(old_results, cached_results, label="new-cached")
    print0("")

    # ---- Summary ----
    print0("=" * 80)
    print0("SUMMARY")
    print0("=" * 80)
    if old_results is not None:
        print0(f"  Old (sequential):    {old_time:>8.2f}s")
    print0(f"  New (first run):     {best_time:>8.2f}s   batch_size={best_params[0]}, queue_size={best_params[1]}")
    print0(f"  New (cached):        {cached_time:>8.2f}s")
    if old_results is not None:
        print0(f"  Speedup (first):     {old_time / best_time:>8.2f}x")
        print0(f"  Speedup (cached):    {old_time / cached_time:>8.2f}x")

    compute_cleanup()

if __name__ == "__main__":
    main()
