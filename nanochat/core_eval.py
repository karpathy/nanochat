"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794

TODOs:
- All tasks ~match except for squad. We get 31% reference is 37%. Figure out why.
"""
import random

from jinja2 import Template
import torch
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Prompt rendering utilities

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompt for a language modeling task.
    Notice that we manually trim the context in the template,
    which in some datasets seems to have trailing whitespace (which we don't want).
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # Return two prompts: without and with the continuation
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # Due to the way the data seems to be stored, I think I need to strip in the case of LM here.
    # Otherwise we may get trailing whitespaces in prompt_without (which get absorbed into the next
    # token in prompt_with), meaning we don't get a nice and clean prefix in the token space
    # to detect the final continuation. Tokenizers...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    """
    Find the length of the common prefix or suffix across token sequences
    - direction: 'left' for prefix, 'right' for suffix
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    # Find the first position where the token sequences differ
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """Stack up a list of token sequences, pad to longest on the right"""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    # In multiple choice, contexts are the same but the continuation is different (common prefix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each continuation
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    # In schema tasks, contexts vary but continuation is the same (common suffix)
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each context
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    # In LM tasks, we have two prompts: without and with continuation
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    end_idx = len(tokens_with)
    # Find longest common prefix — greedy trie tokenizers are not always
    # prefix-stable, so we can't assume an exact prefix match.
    start_idx = 0
    for i in range(min(len(tokens_without), len(tokens_with))):
        if tokens_without[i] != tokens_with[i]:
            break
        start_idx = i + 1
    assert start_idx < end_idx, "continuation must produce additional tokens"
    # we only need the with continuation prompt in the LM task, i.e. batch size of 1
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
@torch.compiler.disable
def forward_model(model, input_ids):
    """
    Take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    The last column of losses is set to nan because we don't have autoregressive targets there.
    """
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids)
    # Roll the tensor to the left by one position to get the (autoregressive) target ids
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # Calculate cross entropy at all positions
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)
    # Set the last column to be nan because there is no autoregressive loss there
    losses[:, -1] = float('nan')
    # Get the argmax predictions at each position
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


def prepare_example(idx, tokenizer, data, task_meta, max_seq_len=None):
    """CPU-only: render prompts, tokenize, stack into tensors. Returns a dict."""
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    # Sample few-shot examples (excluding current item)
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    # Render prompts and batch sequences based on task type
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

    # Truncate sequences for models with a max length (e.g. GPT-2)
    if max_seq_len is not None:
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_seq_len:
                num_to_crop = len(t) - max_seq_len
                new_tokens.append(t[-max_seq_len:])
                new_start_idxs.append(s - num_to_crop)
                new_end_idxs.append(e - num_to_crop)
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id)  # (num_options, seq_len)

    return {
        'input_ids': input_ids,
        'start_idxs': start_idxs,
        'end_idxs': end_idxs,
        'gold': item.get('gold', None),
        'task_type': task_type,
        'num_options': input_ids.size(0),
        'seq_len': input_ids.size(1),
        'pad_token_id': pad_token_id,
    }


def check_result(losses, predictions, input_ids, start_idxs, end_idxs, gold, task_type):
    """Analyze forward pass outputs for one example, return True if correct."""
    if task_type == 'language_modeling':
        si, ei = start_idxs[0], end_idxs[0]
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        return torch.all(predicted_tokens == actual_tokens).item()
    elif task_type in ['multiple_choice', 'schema']:
        mean_losses = [losses[i, si-1:ei-1].mean().item()
                       for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        return mean_losses.index(min(mean_losses)) == gold
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def _collate_batches(prepared, batch_size, queue):
    """Background thread: collate batches on CPU and push to queue."""
    for batch_start in range(0, len(prepared), batch_size):
        batch = prepared[batch_start:batch_start + batch_size]
        batch_preps = [p for _, p in batch]
        max_len = max(p['seq_len'] for p in batch_preps)
        total_rows = sum(p['num_options'] for p in batch_preps)
        pad_id = batch_preps[0]['pad_token_id']

        combined_ids = torch.full((total_rows, max_len), pad_id, dtype=torch.long)

        batch_meta = []
        offset = 0
        for idx, p in batch:
            n, sl = p['num_options'], p['seq_len']
            combined_ids[offset:offset+n, :sl] = p['input_ids']
            batch_meta.append((idx, n, p['start_idxs'], p['end_idxs'], p['gold'], p['task_type']))
            offset += n

        queue.put((combined_ids, batch_meta))
    queue.put(None)  # sentinel


def prepare_task_data(tokenizer, data, task_meta, max_seq_len=None):
    """CPU-only: prepare and sort all examples for a task. Can run on a background thread."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    indices = list(range(rank, len(data), world_size))
    prepared = [(idx, prepare_example(idx, tokenizer, data, task_meta, max_seq_len)) for idx in indices]
    prepared.sort(key=lambda x: x[1]['seq_len'])
    return prepared


def _prefetch_to_device(tensor, device):
    """Pin and async-transfer a CPU tensor to GPU, overlapping with current GPU work."""
    return tensor.pin_memory().to(device, non_blocking=True)


def _forward_batches(model, collated, data, device, pbar=None):
    """Run GPU forward passes on pre-collated batches, return per-example correctness tensor.

    Uses double-buffered prefetching on CUDA: while the GPU processes batch N,
    batch N+1 is pinned and DMA-transferred asynchronously, keeping the GPU fed.
    """
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    if not collated:
        return correct

    use_prefetch = torch.cuda.is_available() and 'cuda' in str(device)

    # Prefetch first batch
    if use_prefetch:
        next_ids = _prefetch_to_device(collated[0][0], device)
    else:
        next_ids = collated[0][0].to(device)

    for i, (_, batch_meta) in enumerate(collated):
        combined_ids = next_ids
        # Start async transfer of next batch while GPU computes on current
        if i + 1 < len(collated):
            if use_prefetch:
                next_ids = _prefetch_to_device(collated[i + 1][0], device)
            else:
                next_ids = collated[i + 1][0].to(device)

        losses, predictions = forward_model(model, combined_ids)

        offset = 0
        for idx, n, start_idxs, end_idxs, gold, task_type in batch_meta:
            is_correct = check_result(
                losses[offset:offset+n], predictions[offset:offset+n],
                combined_ids[offset:offset+n],
                start_idxs, end_idxs, gold, task_type,
            )
            correct[idx] = float(is_correct)
            offset += n
        if pbar is not None:
            pbar.update(len(batch_meta))
    return correct


def _forward_all_cached(model, task_collated, device, pbar=None, task_labels=None,
                        on_task_done=None, batched=False, merge=1, split=1, pad_token_id=0):
    """Run all tasks' cached batches through the model in one pass.

    All batch tensors are moved to device upfront (~144MB for full CORE eval).
    If tensors are already on device (caller preloaded), .to() is a no-op.

    Default mode (batched=False): forwards each example individually, trimming
    collation padding to recover the exact per-example tensor shape. This
    guarantees identical results to sequential per-example evaluation.

    Batched mode (batched=True): forwards collated batches with optional GPU
    composition. Faster but may produce tiny FP differences vs sequential eval
    due to different cuBLAS kernel paths for different matrix dimensions.
    - merge > 1: pad+cat consecutive base batches on GPU before forwarding.
    - split > 1: slice each group into chunks by example boundaries.

    Args:
        task_collated: list of (collated_batches, data) per task
        pbar: optional progress bar, updated per example (or per batch chunk)
        task_labels: optional list of task names for pbar description updates
        on_task_done: optional callback(task_idx, correct_tensor) fired when a task completes
        batched: if True, forward whole batches (faster, approximate). Default False (exact).
        merge/split/pad_token_id: only used when batched=True
    Returns:
        list of correct tensors (one per task, on device)
    """
    # Flatten all batches and move to device upfront (no-op if already there)
    flat_stream = []  # (gpu_ids, batch_meta, task_idx)
    correct = []
    task_batch_counts = []
    for task_idx, (collated, data) in enumerate(task_collated):
        correct.append(torch.zeros(len(data), dtype=torch.float32, device=device))
        task_batch_counts.append(len(collated))
        for combined_ids, batch_meta in collated:
            flat_stream.append((combined_ids.to(device), batch_meta, task_idx))

    if not flat_stream:
        return correct

    task_batches_remaining = list(task_batch_counts)
    current_task = -1

    if not batched:
        # Per-example forwarding: identical results to sequential evaluation.
        # Each example's rows are trimmed to their original seq_len (= max(end_idxs)),
        # removing collation padding so forward_model sees the same tensor shape as
        # the sequential path.
        for combined_ids, batch_meta, task_idx in flat_stream:
            if task_idx != current_task:
                current_task = task_idx
                if pbar is not None and task_labels is not None:
                    pbar.set_description(task_labels[task_idx])

            offset = 0
            for idx, n, start_idxs, end_idxs, gold, task_type in batch_meta:
                seq_len = max(end_idxs)
                example_ids = combined_ids[offset:offset+n, :seq_len]
                losses, predictions = forward_model(model, example_ids)
                is_correct = check_result(
                    losses, predictions, example_ids,
                    start_idxs, end_idxs, gold, task_type,
                )
                correct[task_idx][idx] = float(is_correct)
                offset += n

            if pbar is not None:
                pbar.update(len(batch_meta))
            if on_task_done is not None:
                task_batches_remaining[task_idx] -= 1
                if task_batches_remaining[task_idx] == 0:
                    on_task_done(task_idx, correct[task_idx])
    else:
        # Batched forwarding with optional merge/split composition.
        buffer_ids = []
        buffer_info = []

        for i, (combined_ids, batch_meta, task_idx) in enumerate(flat_stream):
            if task_idx != current_task:
                current_task = task_idx
                if pbar is not None and task_labels is not None:
                    pbar.set_description(task_labels[task_idx])
            buffer_ids.append(combined_ids)
            buffer_info.append((batch_meta, task_idx))

            if len(buffer_ids) < merge and i < len(flat_stream) - 1:
                continue

            # GPU compose: pad+cat if multiple batches, otherwise use as-is
            if len(buffer_ids) == 1:
                mega_ids = buffer_ids[0]
            else:
                max_len = max(t.shape[1] for t in buffer_ids)
                parts = []
                for t in buffer_ids:
                    if t.shape[1] < max_len:
                        pad = torch.full((t.shape[0], max_len - t.shape[1]), pad_token_id,
                                         dtype=t.dtype, device=t.device)
                        t = torch.cat([t, pad], dim=1)
                    parts.append(t)
                mega_ids = torch.cat(parts, dim=0)

            examples = []
            row_bounds = [0]
            for bm, tidx in buffer_info:
                for idx, n, start_idxs, end_idxs, gold, task_type in bm:
                    examples.append((idx, n, start_idxs, end_idxs, gold, task_type, tidx))
                    row_bounds.append(row_bounds[-1] + n)

            n_ex = len(examples)
            chunk_size = -(-n_ex // split)

            for cs in range(0, n_ex, chunk_size):
                ce = min(cs + chunk_size, n_ex)
                chunk = examples[cs:ce]
                chunk_ids = mega_ids[row_bounds[cs]:row_bounds[ce]]

                losses, predictions = forward_model(model, chunk_ids)

                offset = 0
                for idx, n, start_idxs, end_idxs, gold, task_type, tidx in chunk:
                    is_correct = check_result(
                        losses[offset:offset+n], predictions[offset:offset+n],
                        chunk_ids[offset:offset+n],
                        start_idxs, end_idxs, gold, task_type,
                    )
                    correct[tidx][idx] = float(is_correct)
                    offset += n
                if pbar is not None:
                    pbar.update(len(chunk))

            if on_task_done is not None:
                for bm, tidx in buffer_info:
                    task_batches_remaining[tidx] -= 1
                    if task_batches_remaining[tidx] == 0:
                        on_task_done(tidx, correct[tidx])

            buffer_ids.clear()
            buffer_info.clear()

    return correct


def compose_collated(base_collated, target_batch_size, base_batch_size=4, pad_token_id=0):
    """Compose base-sized collated batches into target-sized batches.

    Supports both merging (target > base) by concatenating consecutive groups,
    and splitting (target < base) by slicing along example boundaries.
    Examples are sorted by seq_len within each base batch, so splitting can
    trim trailing padding columns for efficiency.
    """
    if target_batch_size == base_batch_size:
        return base_collated
    elif target_batch_size > base_batch_size:
        # Merge consecutive base batches
        n_merge = target_batch_size // base_batch_size
        composed = []
        for i in range(0, len(base_collated), n_merge):
            group = base_collated[i:i + n_merge]
            if len(group) == 1:
                composed.append(group[0])
                continue
            max_len = max(ids.shape[1] for ids, _ in group)
            parts = []
            merged_meta = []
            for ids, meta in group:
                if ids.shape[1] < max_len:
                    pad = torch.full((ids.shape[0], max_len - ids.shape[1]), pad_token_id, dtype=ids.dtype)
                    ids = torch.cat([ids, pad], dim=1)
                parts.append(ids)
                merged_meta.extend(meta)
            composed.append((torch.cat(parts, dim=0), merged_meta))
        return composed
    else:
        # Split base batches into smaller chunks
        composed = []
        for combined_ids, batch_meta in base_collated:
            for chunk_start in range(0, len(batch_meta), target_batch_size):
                chunk_meta = batch_meta[chunk_start:chunk_start + target_batch_size]
                row_start = sum(m[1] for m in batch_meta[:chunk_start])
                row_end = row_start + sum(m[1] for m in chunk_meta)
                chunk_ids = combined_ids[row_start:row_end]
                # Trim trailing padding (examples sorted by seq_len, so chunks
                # near the start of a base batch may need fewer columns)
                non_pad = (chunk_ids != pad_token_id)
                if non_pad.any():
                    last_col = non_pad.any(dim=0).nonzero()[-1].item() + 1
                    if last_col < chunk_ids.shape[1]:
                        chunk_ids = chunk_ids[:, :last_col].contiguous()
                composed.append((chunk_ids, chunk_meta))
        return composed


def evaluate_task(model, data, device, batch_size=4, queue_size=2, prepared=None,
                  collated=None, tokenizer=None, task_meta=None, pbar=None):
    """
    Evaluate one task across many examples with batched GPU forward passes.
    Examples are sorted by sequence length so similar-length sequences are batched
    together, minimizing padding waste and increasing GPU utilization.

    Three modes (checked in order):
    - collated: skip prepare + collation, go straight to GPU forward passes.
    - prepared: skip prepare, collation runs on a background thread pipelined with GPU.
    - neither: full pipeline (prepare + collate + forward).

    Returns (accuracy, collated_batches) so the caller can cache collated batches.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if collated is not None:
        # Fast path: just GPU forward passes, no threads
        correct = _forward_batches(model, collated, data, device, pbar=pbar)
    else:
        from queue import Queue
        from threading import Thread

        if prepared is None:
            max_seq_len = getattr(model, 'max_seq_len', None)
            prepared = prepare_task_data(tokenizer, data, task_meta, max_seq_len)

        # Collation thread pipelined with GPU forward passes.
        # Double-buffered: while GPU processes batch N, batch N+1 is
        # pin_memory()'d and DMA-transferred asynchronously.
        queue = Queue(maxsize=queue_size)
        collator = Thread(target=_collate_batches, args=(prepared, batch_size, queue), daemon=True)
        collator.start()

        use_prefetch = torch.cuda.is_available() and 'cuda' in str(device)
        def transfer(tensor):
            return _prefetch_to_device(tensor, device) if use_prefetch else tensor.to(device)

        collated = []
        correct = torch.zeros(len(data), dtype=torch.float32, device=device)

        # Prime: get first batch and start its transfer
        item = queue.get()
        if item is not None:
            next_ids = transfer(item[0])

        while item is not None:
            collated.append(item)
            combined_ids = next_ids
            _, batch_meta = item

            # Start async transfer of next batch (overlaps with forward pass below)
            item = queue.get()
            if item is not None:
                next_ids = transfer(item[0])

            losses, predictions = forward_model(model, combined_ids)

            offset = 0
            for idx, n, start_idxs, end_idxs, gold, task_type in batch_meta:
                is_correct = check_result(
                    losses[offset:offset+n], predictions[offset:offset+n],
                    combined_ids[offset:offset+n],
                    start_idxs, end_idxs, gold, task_type,
                )
                correct[idx] = float(is_correct)
                offset += n
            if pbar is not None:
                pbar.update(len(batch_meta))

        collator.join()
        del prepared

    # sync results across all the processes if running distributed
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    return correct.mean().item(), collated
