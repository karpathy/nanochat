"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794

TODOs:
- All tasks ~match except for squad. We get 31% reference is 37%. Figure out why.
"""
import math
import random
import re

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
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    # we only need the with continuation prompt in the LM task, i.e. batch size of 1
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
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


# -----------------------------------------------------------------------------
# Counterfactual controls
# After the normal pass, re-score with the part of the input a competent model must read corrupted
# (swapped with another item's), keeping the few-shot examples intact. A model that actually reads
# the input drops toward chance; one whose answer comes from a prior (boolq yes/no), an answer-position
# bias, or just the more likely choice on its own does not -- so a larger clean->corrupted drop means
# the score is more likely real. The control is auto-detected from each task's structure.

_OPTION_LINE = re.compile(r"^\s*([A-Z])[.)]+\s+(.*)$")  # "A. <text>" / "A.) <text>" lettered-MC option lines


def control_for_task(data, task_type):
    """Return the counterfactual control that fits a task's structure, or None."""
    if not data:
        return None
    item = data[0]
    if task_type == 'multiple_choice':
        choices = item['choices']
        query = item.get('query', '')
        if all(isinstance(c, str) and len(c.strip()) == 1 and c.strip().isalpha() for c in choices):
            # lettered MC (csqa/language_id/lsat): the options are listed inside the query and the
            # model picks a letter, so the stem to swap is the text before them. Only controllable if
            # the option lines actually parse (else skip, rather than record a silent no-op).
            n_opt = sum(1 for ln in query.split('\n') if _OPTION_LINE.match(ln))
            return 'stem_swap' if n_opt == len(choices) else None
        if 'Passage:' in query and 'Question:' in query:
            return 'passage_swap'     # reading comprehension (e.g. boolq): swap the passage
        return 'stem_swap'            # content MC (piqa/arc/hellaswag): the query is the stem
    if task_type == 'language_modeling':
        return 'context_swap'
    return None                       # schema (and anything else): no control yet


def apply_control(item, data, idx, control, rng):
    """Corrupt the readable input of `item` (returning a copy) and return (item, gold).

    No control remaps gold: each control keeps the options/choices in place, so the original gold
    index stays valid -- it is returned (unchanged) only so callers have a uniform interface.
    """
    donor = data[rng.choice([i for i in range(len(data)) if i != idx])]
    if control == 'context_swap':
        return {**item, 'context': donor['context']}, item.get('gold')
    if control == 'passage_swap':
        # boolq-style: swap the passage but keep this item's question (the text after "\nQuestion:")
        delim = '\nQuestion:'
        if delim in item['query'] and delim in donor['query']:
            new_query = donor['query'].split(delim, 1)[0] + delim + item['query'].split(delim, 1)[1]
            return {**item, 'query': new_query}, item['gold']
        return {**item, 'query': donor['query']}, item['gold']
    if control == 'stem_swap':
        # Swap the question/stem with another item's, keeping this item's choices + gold. Content MC
        # keeps its choices in a separate field, so the query is just the stem -> swap it whole.
        # Lettered MC lists the options inside the query ("A. ..."), so swap only the text before the
        # first option line and keep this item's options. Either way the model can no longer match the
        # question to its answer -- the same "reads -> drops" direction as the other controls.
        lines = item['query'].split('\n')
        first_opt = next((i for i, ln in enumerate(lines) if _OPTION_LINE.match(ln)), None)
        if first_opt is None:
            return {**item, 'query': donor['query']}, item['gold']        # content MC: swap whole query
        donor_lines = donor['query'].split('\n')
        donor_opt = next((i for i, ln in enumerate(donor_lines) if _OPTION_LINE.match(ln)), None)
        if first_opt and donor_opt:  # lettered MC: a non-empty stem before parseable options
            return {**item, 'query': '\n'.join(donor_lines[:donor_opt] + lines[first_opt:])}, item['gold']
        return item, item['gold']
    return item, item['gold']


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta, control=None):
    """Evaluate a single example; return (is_correct, info).

    If `control` is set (see control_for_task / apply_control) the test item's readable input is
    corrupted (the original gold is kept), while the few-shot examples stay intact -- a counterfactual
    a reading model should fail. `info` records the response for logging: multiple_choice/schema ->
    {idx, gold, pred, scores}; language_modeling -> {idx, correct}.
    """
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    # Sample few-shot examples (excluding current item), from the original (un-corrupted) data
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    # Counterfactual control: corrupt the test item's readable input (the original gold is kept)
    gold = item.get('gold')
    if control is not None:
        item, gold = apply_control(item, data, idx, control, random.Random(99 + idx))

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

    # Some models can't forward sequences beyond a certain length (e.g. GPT-2)
    # In these cases, we have to truncate sequences to max length and adjust the indices
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:]) # take the last max_tokens tokens
                new_start_idxs.append(s - num_to_crop) # shift the indices down
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0, "this should never happen right?"
                assert e - num_to_crop >= 0, "this should never happen right?"
            else:
                new_tokens.append(t) # keep unchanged
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    # Stack up all the sequences into a batch
    pad_token_id = tokenizer.get_bos_token_id() # use BOS as pad token is ok
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    # Forward the model, get the autoregressive loss and argmax prediction at each token
    losses, predictions = forward_model(model, input_ids)

    # See if the losses/predictions come out correctly
    if task_type == 'language_modeling':
        # language modeling task is currently always batch size 1
        si = start_idxs[0]
        ei = end_idxs[0]
        # predictions[i] predict input_ids[i+1] autoregressively
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
        info = {'idx': idx, 'correct': bool(is_correct)}
    elif task_type in ['multiple_choice', 'schema']:
        # For MC/schema: find the option with lowest average loss
        mean_losses = [losses[i, si-1:ei-1].mean().item()
                        for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = pred_idx == gold
        info = {'idx': idx, 'gold': gold, 'pred': pred_idx,
                'scores': [round(m, 5) if math.isfinite(m) else None for m in mean_losses]}
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return is_correct, info


def evaluate_task(model, tokenizer, data, device, task_meta, control=None, collect_responses=False):
    """
    Evaluate one task across many examples, dispatching to all processes under torchrun.
    Returns (mean_correct, responses). `control` runs the counterfactual control (see
    control_for_task); `collect_responses=True` additionally returns this rank's per-example `info`
    records for logging (this rank's strided slice, not gathered across ranks).
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    responses = []
    # stride the examples to each rank
    for idx in range(rank, len(data), world_size):
        is_correct, info = evaluate_example(idx, model, tokenizer, data, device, task_meta, control=control)
        correct[idx] = float(is_correct)
        if collect_responses:
            responses.append(info)
    # sync results across all the processes if running distributed
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    # compute the mean
    mean_correct = correct.mean().item()
    return mean_correct, responses
