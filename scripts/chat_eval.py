"""
Evaluate the Chat model.
All the generic code lives here, and all the evaluation-specific
code lives in nanochat directory and is imported from here.

Example runs:
python -m scripts.chat_eval -a ARC-Easy
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -a ARC-Easy
"""

import argparse
import os
from functools import partial
from contextlib import nullcontext

import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, get_dist_info, print0, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.spellingbee import SpellingBee
from tasks.aime_2024 import AIME2024I, AIME2024II, AIME2024
from tasks.aime_2025 import AIME2025

import numpy as np
from collections import defaultdict
import json

# -----------------------------------------------------------------------------
# Pass@k calculation utilities

def unbiased_pass_at_k(n_correct_per_question, n_total_per_question, k):
    """
    Calculate unbiased pass@k probability using the estimator from the OpenAI paper.
    
    Args:
        n_correct_per_question: List of number of correct samples per question
        n_total_per_question: List of total samples per question
        k: The k in pass@k
    
    Returns:
        Average pass@k probability across all questions
    """
    total_pass_k_prob = 0.0
    for c, n in zip(n_correct_per_question, n_total_per_question):
        assert n >= k, f"n ({n}) must be >= k ({k})"
        # If at least k samples are correct, pass@k = 1
        if n - c < k:
            prob = 1.0
        else:
            # 1 - (n-c)/n * (n-c-1)/(n-1) * ... * (n-c-k+1)/(n-k+1)
            prob = 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
        total_pass_k_prob += prob
    return total_pass_k_prob / len(n_correct_per_question) if n_correct_per_question else 0.0


def compute_pass_at_k_from_results(results_per_question, ks=[1, 2, 4, 8, 16]):
    """
    Compute pass@k for various k values from per-question results.
    
    Args:
        results_per_question: Dict mapping question_id -> list of (correct: bool, completion: str)
        ks: List of k values to compute
    
    Returns:
        Dict mapping k -> pass@k score
    """
    # Get n (total samples per question) - assume all questions have same n
    n = len(next(iter(results_per_question.values())))
    
    # Count correct per question
    n_correct_per_question = []
    for question_id, results in results_per_question.items():
        c = sum(1 for correct, _ in results if correct)
        n_correct_per_question.append(c)
    
    n_total_per_question = [n] * len(n_correct_per_question)
    
    pass_at_k_scores = {}
    for k in ks:
        if k <= n:
            score = unbiased_pass_at_k(n_correct_per_question, n_total_per_question, k)
            pass_at_k_scores[k] = score
    
    return pass_at_k_scores


# -----------------------------------------------------------------------------
# Generative evaluation loop (we go one problem at a time, sample, evaluate)

def run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=None, save_results_path=None, compute_passatk=False, passatk_ks=None):
    """
    Run generative evaluation with optional pass@k computation.
    
    Args:
        compute_passatk: If True, compute and report pass@k scores
        passatk_ks: List of k values for pass@k (default: [1, 2, 4, 8, 16] if compute_passatk)
        save_results_path: If provided, save detailed results to this JSONL file
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    
    if passatk_ks is None:
        passatk_ks = [1, 2, 4, 8, 16]

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # Run the evaluation
    num_passed, total = 0, 0
    # For pass@k, we need to track all outcomes per question
    local_results_per_question = {} if compute_passatk or save_results_path else None
    
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]
        question_id = conversation.get('question_id', f'q_{i}')

        # Tokenize the prompt
        encoded_prompt = tokenizer.render_for_completion(conversation)
        # Get the completions
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # Decode the completions as text
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        # Evaluate success criteria
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        # Keep stats
        total += 1
        num_passed += int(passed)
        
        # Track results for pass@k if needed
        if local_results_per_question is not None:
            local_results_per_question[question_id] = [
                (bool(outcome), completion) for outcome, completion in zip(outcomes, completions)
            ]

        # Logging (overwrite the same line in the console)
        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    # Finish the in-place progress line with a newline before final summary
    print()

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    print0("=" * 50)
    print0(f"Final: {num_passed}/{total} ({100*num_passed/total:.2f}%)")
    
    # Compute pass@k if requested
    passatk_scores = {}
    if compute_passatk:
        # Gather results from all ranks
        if ddp:
            # Serialize local results
            local_data = json.dumps(local_results_per_question)
            # Gather sizes
            local_size = torch.tensor([len(local_data)], dtype=torch.long, device=device)
            sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(ddp_world_size)]
            dist.all_gather(sizes, local_size)
            # Gather data
            max_size = max(s.item() for s in sizes)
            local_padded = local_data.ljust(max_size, '\0')
            gathered = [torch.zeros(max_size, dtype=torch.uint8, device=device) for _ in range(ddp_world_size)]
            local_tensor = torch.tensor([ord(c) for c in local_padded], dtype=torch.uint8, device=device)
            dist.all_gather(gathered, local_tensor)
            
            # Deserialize on rank 0
            if ddp_rank == 0:
                all_results = {}
                for rank, data_tensor in enumerate(gathered):
                    data_str = ''.join(chr(x.item()) for x in data_tensor[:sizes[rank].item()])
                    rank_results = json.loads(data_str)
                    all_results.update(rank_results)
            else:
                all_results = None
        else:
            all_results = local_results_per_question
        
        # Compute pass@k on rank 0
        if ddp_rank == 0 or not ddp:
            passatk_scores = compute_pass_at_k_from_results(all_results, ks=passatk_ks)
            print0("\nPass@k scores:")
            for k in sorted(passatk_scores.keys()):
                print0(f"  Pass@{k}: {passatk_scores[k]:.4f}")
        
        # Broadcast passatk_scores to all ranks
        if ddp:
            if ddp_rank == 0:
                scores_data = json.dumps(passatk_scores)
                scores_size = torch.tensor([len(scores_data)], dtype=torch.long, device=device)
            else:
                scores_size = torch.zeros(1, dtype=torch.long, device=device)
            dist.broadcast(scores_size, src=0)
            
            if ddp_rank == 0:
                scores_padded = scores_data.ljust(scores_size.item(), '\0')
                scores_tensor = torch.tensor([ord(c) for c in scores_padded], dtype=torch.uint8, device=device)
            else:
                scores_tensor = torch.zeros(scores_size.item(), dtype=torch.uint8, device=device)
            dist.broadcast(scores_tensor, src=0)
            
            if ddp_rank != 0:
                scores_str = ''.join(chr(x.item()) for x in scores_tensor)
                passatk_scores = json.loads(scores_str)
    
    # Save results if requested (only on rank 0)
    if save_results_path and (ddp_rank == 0 or not ddp):
        if local_results_per_question is not None:
            # In non-DDP mode, we have all results locally
            # In DDP mode, we gathered them above
            results_to_save = all_results if ddp else local_results_per_question
            
            with open(save_results_path, 'w') as f:
                for question_id, results in results_to_save.items():
                    for sample_idx, (correct, completion) in enumerate(results):
                        record = {
                            'question_id': question_id,
                            'sample_idx': sample_idx,
                            'label': int(correct),
                            'completion': completion,
                        }
                        f.write(json.dumps(record) + '\n')
            print0(f"\nSaved detailed results to {save_results_path}")

    # Return the accuracy and pass@k scores
    return num_passed/total, passatk_scores

# -----------------------------------------------------------------------------
# Categorical evaluation loop
# A lot easier because we don't have to sample. Therefore, we can actually go
# batches at a time and just check the logits for correct answer choices.

def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id() # use BOS as pad token is ok, these positions are ignored

    # We'll process batches of independent problems at a time because there is no sampling needed
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    # Run the evaluation
    letter_to_id_cache = {} # many letters will repeat often, let's save the tokenizer some work
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        # Prepare the batch of problems. They might all be of different length, so we pad/collate them.
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conversation) for conversation in conversations] # TODO: remake the way this works
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids] # where the last token is (and the predicted answer)
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        # Get the logits for the whole batch of conversations in parallel (efficiency win here)
        with torch.no_grad():
            logits = model(prompt_ids) # (B, T, V)

        # Focus on the available answer on just the letters corresponding to choices
        # Note that this helps the evaluation a lot because it specifically narrows the focus to only the available letters
        # The much harder alternative would be to just generate from the Assistant and check if it responded with the correct
        # letter (e.g. A, B, C, D), but evaluations typically make the task easier in this way.
        for idx, conversation in enumerate(conversations):
            # get the token ids of all the available letters of this problem
            letters = conversation['letters']
            letter_ids = []
            for letter in letters:
                if not letter in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "Each letter must be a single token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            # focus logits just down to the answer position and the available letters of the answer
            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]
            # get the argmax letter (the predicted answer)
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            # evaluate the outcome
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    average = num_passed/total
    print0(f"Final: {num_passed}/{total} ({100*average:.2f}%)")
    return average

# -----------------------------------------------------------------------------

def run_chat_eval(task_name, model, tokenizer, engine,
                   batch_size=1, num_samples=1, max_new_tokens=512, temperature=0.0, top_k=50,
                   max_problems=None, save_results_path=None, compute_passatk=False, passatk_ks=None):
    """
    Run evaluation for a single task.
    
    Returns:
        If compute_passatk: (accuracy, passatk_scores_dict)
        Otherwise: accuracy
    """
    # Create the evaluation object
    # Note: AIME tasks are available but NOT included in all_tasks (not auto-run)
    task_module = {
        'HumanEval': HumanEval,
        'MMLU': partial(MMLU, subset="all", split="test"),
        'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test"),
        'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test"),
        'GSM8K': partial(GSM8K, subset="main", split="test"),
        'SpellingBee': partial(SpellingBee, size=256, split="test"),
        'AIME-2024-I': AIME2024I,
        'AIME-2024-II': AIME2024II,
        'AIME-2024': AIME2024,
        'AIME-2025': AIME2025,
    }[task_name]
    task_object = task_module()
    # Run the evaluation
    if task_object.eval_type == 'generative':
        result = run_generative_eval(
            task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k,
            max_problems=max_problems,
            save_results_path=save_results_path,
            compute_passatk=compute_passatk,
            passatk_ks=passatk_ks,
        )
    elif task_object.eval_type == 'categorical':
        acc = run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=max_problems)
        result = (acc, {}) if compute_passatk else acc
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")
    return result

# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', type=str, required=True, help="Source of the model: sft|mid|rl")
    parser.add_argument('-a', '--task-name', type=str, default=None, help="Task name. Default = all tasks. Use | to split multiple tasks.")
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'])
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch size for categorical evaluation')
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
    parser.add_argument('-x', '--max-problems', type=int, default=None, help='Max problems to evaluate')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
    parser.add_argument('--passatk', action='store_true', help='Compute pass@k scores (for generative tasks)')
    parser.add_argument('--passatk-ks', type=str, default='1,2,4,8,16', help='Comma-separated k values for pass@k (default: 1,2,4,8,16)')
    parser.add_argument('--save-results', type=str, default=None, help='Save detailed results to this JSONL file')
    args = parser.parse_args()
    
    # Parse passatk_ks
    passatk_ks = [int(k.strip()) for k in args.passatk_ks.split(',')] if args.passatk else None

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    # Get the tasks to evaluate on
    all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
    baseline_accuracies = {
        'ARC-Easy': 0.25, # multiple choice 1 of 4 => 25%
        'ARC-Challenge': 0.25, # multiple choice 1 of 4 => 25%
        'MMLU': 0.25, # multiple choice 1 of 4 => 25%
        'GSM8K': 0.0, # open-ended => 0%
        'HumanEval': 0.0, # open-ended => 0%
        'SpellingBee': 0.0, # open-ended => 0%
    }
    task_names = all_tasks if args.task_name is None else args.task_name.split('|')

    # Run all the task evaluations sequentially
    results = {}
    passatk_results = {}
    for task_name in task_names:
        with autocast_ctx:
            # Determine save path for this task (add task name suffix)
            save_path = None
            if args.save_results:
                base, ext = os.path.splitext(args.save_results)
                save_path = f"{base}_{task_name}{ext}"
            
            result = run_chat_eval(
                task_name,
                model, tokenizer, engine,
                batch_size=args.batch_size,
                num_samples=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                max_problems=args.max_problems,
                save_results_path=save_path,
                compute_passatk=args.passatk,
                passatk_ks=passatk_ks,
            )
            
            # Handle return value (may be tuple with pass@k scores)
            if args.passatk and isinstance(result, tuple):
                acc, passatk_scores = result
                passatk_results[task_name] = passatk_scores
            else:
                acc = result
            
            results[task_name] = acc
            print0(f"{task_name} accuracy: {100 * acc:.2f}%")

    # Log to report
    from nanochat.report import get_report
    all_tasks_were_evaluated = all(task_name in results for task_name in all_tasks)
    # calculate the ChatCORE metric if we can (similar to CORE, it's the mean centered accuracy)
    # this way, ChatCORE ranges from 0 (at random baseline) to 1 (peak performance)
    chatcore_metric_dict = {}
    if all_tasks_were_evaluated:
        centered_mean = 0
        for task_name, acc in results.items():
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        chatcore_metric = centered_mean / len(results)
        chatcore_metric_dict = {"ChatCORE metric": chatcore_metric}
    report_data = [
        vars(args), # CLI args
        results,
        chatcore_metric_dict,
    ]
    if passatk_results:
        report_data.append({"pass@k": passatk_results})
    get_report().log(section="Chat evaluation " + args.source, data=report_data)

    compute_cleanup()
