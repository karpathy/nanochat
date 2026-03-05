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


def stack_sequences(tokens, pad_token_id) -> torch.Tensor:
    """根据最长的 token 序列长度将短序列 padding 到同一长度，并堆叠成一个 batch

    Args:
        tokens (List[List[int]]): token id 序列列表
        pad_token_id (int): 用于 padding 的 token id
    
    Returns:
        input_ids (torch.LongTensor): padding 后的 token id 序列张量, 形状为 (batch_size, max_seq_len) 的 tensor
    """
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
    """将输入的token id序列输入到模型中, 得到损失和预测结果
    Args:
        - model: 评测用的语言模型
        - input_ids: 输入的token id序列, 形状为 (batch_size, seq_len)

    Returns:
        - losses: 损失, 形状为 (batch_size, seq_len)
        - predictions: 预测结果, 形状为 (batch_size, seq_len)
    """
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids)
    # 构造目标token id序列, 通过将输入序列向左滚动一位得到
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # 计算每个位置的交叉熵损失
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)
    # 忽略最后一个位置的损失, 因为它没有对应的目标token
    losses[:, -1] = float('nan')
    # 得到每个位置的预测token id, 通过取输出logits的argmax得到
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta) -> bool:
    """测试一个题目的正确性, 返回True/False, 支持多选题、schema题和语言模型题

    Args:
        idx: 题目在数据中的索引
        model: 评测用的语言模型
        tokenizer: 评测用的tokenizer
        data: 评测数据列表，每个元素是一个题目的字典表示
        device: 评测用的设备(CPU/GPU)
        task_meta: 任务的元信息字典，包含以下字段：
            - task_type: 任务类型，字符串，取值为'multiple_choice'、'schema'或'language_modeling'
            - num_fewshot: 每个题目需要采样的few-shot示例数量, 整数
            - continuation_delimiter: prompt中上下文和续写之间的分隔符, 字符串
    
    Returns:
        is_correct: 该题目的预测是否正确, 布尔值
    """
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    # 生成模仿示例
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    # 将题目和模仿示例渲染成完整的prompt，并tokenize成输入模型的token序列
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

    # 根据模型的最大输入长度从左侧裁剪token序列, 并相应地调整start/end indices
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

    # 将token序列堆叠成一个batch，并移动到评测设备上
    pad_token_id = tokenizer.get_bos_token_id() # use BOS as pad token is ok
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    # 前向传播得到loss和预测
    losses, predictions = forward_model(model, input_ids)

    # lm 题目, inputs 形状为 (1, seq_len), predictions 形状为 (1, seq_len), start_idxs 和 end_idxs 是长度为1的列表
    if task_type == 'language_modeling':
        si = start_idxs[0]
        ei = end_idxs[0]
        # 截取模型对答案的预测部分，并和正确答案进行比较
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
    # choice/schema 题目, inputs 形状为 (num_options, seq_len), predictions 形状为 (num_options, seq_len), start_idxs 和 end_idxs 是长度为 num_options 的列表
    elif task_type in ['multiple_choice', 'schema']:
        # 计算模型在每个选项上的平均loss, 选loss最小的选项作为模型的预测, 并和正确答案进行比较
        mean_losses = [losses[i, si-1:ei-1].mean().item()
                        for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = pred_idx == item['gold']
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return bool(is_correct)


def evaluate_task(model, tokenizer, data, device, task_meta) -> float:
    """评测一个任务(含有多个题目)的CORE分数, 支持多选题、schema题和语言模型题

    Args: 
        model: 评测用的语言模型
        tokenizer: 评测用的tokenizer
        data: 评测数据列表，每个元素是一个题目的字典表示
        device: 评测用的设备(CPU/GPU)
        task_meta: 任务的元信息字典，包含以下字段：
            - task_type: 任务类型，字符串，取值为'multiple_choice'、'schema'或'language_modeling'
            - num_fewshot: 每个题目需要采样的few-shot示例数量, 整数
            - continuation_delimiter: prompt中上下文和续写之间的分隔符, 字符串
    
    Returns:
        mean_correct: 该任务的CORE分数, 浮点数, 取值范围为0.0到1.0, 代表模型在该任务上预测正确的题目比例
    """
    # 获取当前进程的rank和总的进程数
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    # 为每个进程分配不同的题目进行评测, 以实现并行评测
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
        correct[idx] = float(is_correct)
    # 在所有进程之间同步correct张量，并求和得到总的正确题目数量
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    # 计算正确题目的比例作为CORE分数
    mean_correct = correct.mean().item()
    return mean_correct
