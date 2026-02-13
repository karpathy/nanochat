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
import hashlib
import zipfile
import tempfile
import argparse
from contextlib import nullcontext
from tqdm import tqdm

import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock
from nanochat.tokenizer import HuggingFaceTokenizer
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import (
    forward_model, prepare_example, check_result, stack_sequences,
    prepare_task_data, _collate_batches, _forward_all_cached,
    evaluate_task,
    render_prompts_mc, render_prompts_schema, render_prompts_lm,
    batch_sequences_mc, batch_sequences_schema, batch_sequences_lm,
)

# ---- eval bundle loading (reused from base_eval) ----
torch.backends.fp32_precision = "tf32"
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

BASE_BATCH_SIZE = 4

def file_checksum(path):
    """SHA-256 checksum of a file, truncated to 16 hex chars."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def collate_prepared(prepared, batch_size):
    """Collate prepared examples into batches (non-threaded). Returns (collated, pad_token_id)."""
    pad_id = prepared[0][1]['pad_token_id']
    collated = []
    for batch_start in range(0, len(prepared), batch_size):
        batch = prepared[batch_start:batch_start + batch_size]
        batch_preps = [p for _, p in batch]
        max_len = max(p['seq_len'] for p in batch_preps)
        total_rows = sum(p['num_options'] for p in batch_preps)
        combined_ids = torch.full((total_rows, max_len), pad_id, dtype=torch.long)
        batch_meta = []
        offset = 0
        for idx, p in batch:
            n, sl = p['num_options'], p['seq_len']
            combined_ids[offset:offset+n, :sl] = p['input_ids']
            batch_meta.append((idx, n, p['start_idxs'], p['end_idxs'], p['gold'], p['task_type']))
            offset += n
        collated.append((combined_ids, batch_meta))
    return collated, pad_id


def build_or_load_base_collated(tok_hash, tokenizer, task_inputs, max_seq_len, max_per_task):
    """Build or load base-4 collated data for all tasks, with disk caching."""
    base_dir = get_base_dir()
    cache_dir = os.path.join(base_dir, "core_token_cache", f"{tok_hash}_n{max_per_task}")

    all_cached = os.path.isdir(cache_dir) and all(
        os.path.exists(os.path.join(cache_dir, f"{label}.pt"))
        for label, _, _ in task_inputs
    )

    base_cache = {}  # label -> (collated, pad_token_id)
    if all_cached:
        print0(f"Loading base-{BASE_BATCH_SIZE} collated cache from {cache_dir}")
        for label, _, _ in tqdm(task_inputs, desc="Loading cache", leave=False):
            data = torch.load(os.path.join(cache_dir, f"{label}.pt"), weights_only=False)
            base_cache[label] = (data['collated'], data['pad_token_id'])
    else:
        print0(f"Building base-{BASE_BATCH_SIZE} collated cache (saving to {cache_dir})")
        os.makedirs(cache_dir, exist_ok=True)
        for label, task_meta, data in tqdm(task_inputs, desc="Preparing tasks", leave=False):
            prepared = prepare_task_data(tokenizer, data, task_meta, max_seq_len)
            collated, pad_id = collate_prepared(prepared, BASE_BATCH_SIZE)
            base_cache[label] = (collated, pad_id)
            torch.save({'collated': collated, 'pad_token_id': pad_id},
                       os.path.join(cache_dir, f"{label}.pt"))
            del prepared
        print0(f"Saved {len(base_cache)} task caches to {cache_dir}")

    return base_cache


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


def evaluate_task_old(model, tokenizer, data, device, task_meta, pbar=None):
    """Original sequential evaluate_task."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example_old(idx, model, tokenizer, data, device, task_meta)
        correct[idx] = float(is_correct)
        if pbar is not None:
            pbar.update(1)
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
    total = sum(len(data) for _, _, data in task_inputs)
    max_label_len = max(len(label) for label, _, _ in task_inputs)
    pbar = tqdm(total=total, desc="Sequential", leave=False)
    for label, task_meta, data in task_inputs:
        pbar.set_description(f"Sequential: {label:<{max_label_len}}")
        acc = evaluate_task_old(model, tokenizer, data, device, task_meta, pbar=pbar)
        results[label] = acc
    pbar.close()
    sync_cuda()
    return time.time() - t0, results


def bench_new_first(model, tokenizer, task_inputs, device, batch_size, queue_size, pbar=None):
    """Benchmark new batched evaluation (first run, no cache)."""
    sync_cuda()
    t0 = time.time()
    results = {}
    collated_cache = {}
    max_seq_len = getattr(model, 'max_seq_len', None)
    max_label_len = max(len(label) for label, _, _ in task_inputs)
    for label, task_meta, data in task_inputs:
        if pbar is not None:
            pbar.set_description(f"{label:<{max_label_len}}")
        prepared = prepare_task_data(tokenizer, data, task_meta, max_seq_len)
        acc, collated = evaluate_task(model, data, device, batch_size=batch_size,
                                      queue_size=queue_size, prepared=prepared, pbar=pbar)
        results[label] = acc
        collated_cache[label] = collated
    sync_cuda()
    return time.time() - t0, results, collated_cache


def bench_new_cached(model, task_inputs, device, collated_cache, pbar=None,
                     merge=1, split=1, pad_token_id=0):
    """Benchmark new batched evaluation (cached run, forward only).
    Uses continuous pipeline across all tasks to eliminate inter-task stalls.
    merge/split control GPU-side composition: merge > 1 cats batches, split > 1 slices them."""
    import torch.distributed as dist
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    sync_cuda()
    t0 = time.time()
    task_collated = [(collated_cache[label], data) for label, _, data in task_inputs]
    correct_list = _forward_all_cached(model, task_collated, device, pbar=pbar,
                                       merge=merge, split=split, pad_token_id=pad_token_id)
    sync_cuda()
    elapsed = time.time() - t0
    results = {}
    for (label, _, data), correct in zip(task_inputs, correct_list):
        if world_size > 1:
            dist.barrier()
            dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        results[label] = correct.mean().item()
    return elapsed, results


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
    parser.add_argument('--skip-first', action='store_true', help='Skip first-run sweep (requires cached collated data)')
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
    # Compute tok_hash early — needed for both skip-first check and cache loading
    max_seq_len = getattr(model, 'max_seq_len', None)
    if args.hf_path is not None:
        tok_hash = hashlib.sha256(args.hf_path.encode()).hexdigest()[:16]
    else:
        base_dir = get_base_dir()
        for fname in ("tokenizer.pkl", "tokenizer.json"):
            tok_path = os.path.join(base_dir, "tokenizer", fname)
            if os.path.exists(tok_path):
                tok_hash = file_checksum(tok_path)
                break

    # Check if we can skip the first-run sweep
    best_time = None
    best_params = None
    if args.skip_first:
        cache_dir = os.path.join(get_base_dir(), "core_token_cache", f"{tok_hash}_n{args.max_per_task}")
        has_cache = os.path.isdir(cache_dir) and all(
            os.path.exists(os.path.join(cache_dir, f"{label}.pt"))
            for label, _, _ in task_inputs
        )
        if has_cache:
            print0("Skipping first-run sweep (--skip-first, cache exists)")
            print0("")
        else:
            print0(f"--skip-first: cache missing, running single bs={BASE_BATCH_SIZE} eval to create it...")
            pbar = tqdm(total=total_examples, desc="Creating cache", leave=False)
            with autocast_ctx:
                _, _, collated_cache = bench_new_first(model, tokenizer, task_inputs, device,
                                                       batch_size=BASE_BATCH_SIZE, queue_size=2, pbar=pbar)
            pbar.close()
            pad_id = tokenizer.get_bos_token_id()
            os.makedirs(cache_dir, exist_ok=True)
            for label in collated_cache:
                torch.save({'collated': collated_cache[label], 'pad_token_id': pad_id},
                           os.path.join(cache_dir, f"{label}.pt"))
            print0(f"Cache created ({len(collated_cache)} tasks)")
            print0("")

    if not args.skip_first:
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
        total_combos = len(batch_sizes) * len(queue_sizes)
        outer_pbar = tqdm(total=total_combos, desc="First-run sweep", leave=False, position=0)
        inner_pbar = tqdm(total=total_examples, desc="", leave=False, position=1)

        for bs in batch_sizes:
            row = f"  {bs:>10}"
            for qs in queue_sizes:
                outer_pbar.set_description(f"First-run: bs={bs} qs={qs}")
                inner_pbar.reset()
                with autocast_ctx:
                    t, results, collated_cache = bench_new_first(model, tokenizer, task_inputs, device, bs, qs, pbar=inner_pbar)
                sweep_results[(bs, qs)] = t
                row += f"{t:>9.2f}s"
                if t < best_time:
                    best_time = t
                    best_params = (bs, qs)
                    best_collated = collated_cache
                    best_first_results = results
                outer_pbar.update(1)
            outer_pbar.write(row)
        inner_pbar.close()
        outer_pbar.close()

        print0("")
        print0(f"  Best: batch_size={best_params[0]}, queue_size={best_params[1]} -> {best_time:.2f}s  ({total_examples / best_time:.1f} examples/s)")

        # Verify correctness
        if old_results is not None:
            verify_results(old_results, best_first_results, label="new-first")
        print0("")

    # ---- 3. Build/load base-4 collated cache ----
    base_cache = build_or_load_base_collated(tok_hash, tokenizer, task_inputs, max_seq_len, args.max_per_task)

    # ---- 4. Cached run sweep (forward only, composed from base-4) ----
    print0("=" * 80)
    print0(f"NEW: Cached run (forward only, composed from base-{BASE_BATCH_SIZE})")
    print0("=" * 80)
    print0("")

    best_cached_time = float('inf')
    best_cached_params = None
    pad_id = next(iter(base_cache.values()))[1]
    # Preload ALL base-4 batches to GPU once (~144MB for full CORE eval).
    # All batch-size sweeps compose from these GPU-resident tensors — zero CPU→GPU transfers.
    gpu_collated = {}
    for label, (collated, _) in base_cache.items():
        gpu_collated[label] = [(ids.to(device), meta) for ids, meta in collated]
    outer_pbar = tqdm(total=len(batch_sizes), desc="Cached sweep", leave=False, position=0)
    inner_pbar = tqdm(total=total_examples, desc="", leave=False, position=1)

    for bs in batch_sizes:
        outer_pbar.set_description(f"Cached: bs={bs}")
        inner_pbar.reset()

        # All composition happens on GPU: merge for bs >= base, split for bs < base
        if bs >= BASE_BATCH_SIZE:
            merge, split = bs // BASE_BATCH_SIZE, 1
        else:
            merge, split = 1, BASE_BATCH_SIZE // bs

        with autocast_ctx:
            t, cached_results = bench_new_cached(model, task_inputs, device, gpu_collated,
                                                 pbar=inner_pbar, merge=merge, split=split,
                                                 pad_token_id=pad_id)

        outer_pbar.write(f"  batch_size={bs:>3}:  {t:.2f}s  ({total_examples / t:.1f} examples/s)")
        outer_pbar.update(1)

        if t < best_cached_time:
            best_cached_time = t
            best_cached_params = bs
            best_cached_results = cached_results
    inner_pbar.close()
    outer_pbar.close()

    print0("")
    print0(f"  Best: batch_size={best_cached_params} -> {best_cached_time:.2f}s  ({total_examples / best_cached_time:.1f} examples/s)")

    if old_results is not None:
        verify_results(old_results, best_cached_results, label="new-cached")
    print0("")

    # ---- Summary ----
    print0("=" * 80)
    print0("SUMMARY")
    print0("=" * 80)
    if old_results is not None:
        print0(f"  Old (sequential):    {old_time:>8.2f}s")
    if best_time is not None:
        print0(f"  New (first run):     {best_time:>8.2f}s   batch_size={best_params[0]}, queue_size={best_params[1]}")
    print0(f"  New (cached):        {best_cached_time:>8.2f}s   batch_size={best_cached_params}")
    if old_results is not None:
        if best_time is not None:
            print0(f"  Speedup (first):     {old_time / best_time:>8.2f}x")
        print0(f"  Speedup (cached):    {old_time / best_cached_time:>8.2f}x")

    compute_cleanup()

if __name__ == "__main__":
    main()
