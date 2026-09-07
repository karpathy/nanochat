"""
Evaluate the Chat model on multiple-choice tasks (ARC-Easy, ARC-Challenge, MMLU) and
report ChatCORE, their mean accuracy centered on chance. Each task is a batch of
forward passes: the answer is the letter with the highest logit at the answer position.

Example runs:
python -m scripts.chat_eval -i chat -a ARC-Easy
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -i chat -a ARC-Easy
"""

import argparse
from functools import partial
import torch
import torch.distributed as dist

from nanochat.common import get_dist_info, print0

from harness.runtime import compute_init, compute_cleanup, autodetect_device_type
from harness.experiment import format_record, format_invocation
from harness.checkpoint import load_model, find_largest_model

from harness.tasks import MMLU, ARC

# -----------------------------------------------------------------------------
# Categorical evaluation loop
# A lot easier because we don't have to sample. Therefore, we can actually go
# batches at a time and just check the logits for correct answer choices.

def run_categorical_eval(task, tokenizer, model, batch_size, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id() # use BOS as pad token is ok, these positions are ignored

    # We'll process batches of independent problems at a time because there is no sampling needed
    num_problems = len(task) if max_problems is None else min(len(task), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    # Run the evaluation
    letter_to_id_cache = {} # many letters will repeat often, let's save the tokenizer some work
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        # Prepare the batch of problems. They might all be of different length, so we pad/collate them.
        conversations = [task[ii] for ii in range(i0, i1)]
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
            correct_letter = conversation["messages"][-1]["content"]
            num_passed += int(predicted_letter == correct_letter)
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

TASKS = {
    'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test"),
    'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test"),
    'MMLU': partial(MMLU, split="test"),
}
CHANCE = 0.25 # every task is a 4-way multiple choice

# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', type=str, required=True, help="Source of the model: base|sft")
    parser.add_argument('-a', '--task-name', type=str, default=None, help="Task name. Default = all tasks. Use | to split multiple tasks.")
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch size for the evaluation')
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
    parser.add_argument('-x', '--max-problems', type=int, default=None, help='Max problems to evaluate')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
    args = parser.parse_args()
    print0(format_invocation(args))

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    model_tag = args.model_tag if args.model_tag is not None else find_largest_model(args.source)
    model, tokenizer, meta = load_model(args.source, device, model_tag=model_tag, step=args.step)

    # Run all the task evaluations sequentially
    task_names = list(TASKS) if args.task_name is None else args.task_name.split('|')
    results = {}
    for task_name in task_names:
        task = TASKS[task_name]()
        acc = run_categorical_eval(task, tokenizer, model, args.batch_size, max_problems=args.max_problems)
        results[task_name] = acc
        print0(f"{task_name} accuracy: {100 * acc:.2f}%")

    # ChatCORE (like CORE): the mean accuracy centered so that chance is 0 and perfect is 1
    all_tasks_were_evaluated = all(task_name in results for task_name in TASKS)
    if all_tasks_were_evaluated:
        centered = [(acc - CHANCE) / (1.0 - CHANCE) for acc in results.values()]
        chatcore_metric = sum(centered) / len(centered)
        print0(f"ChatCORE metric: {chatcore_metric:.4f}")

    # the stage record (see harness/experiment.py); this is what downstream tooling consumes
    summary = {"model_tag": model_tag, "source": args.source, "step": meta["step"]}
    summary.update({task_name: round(acc, 6) for task_name, acc in results.items()})
    if all_tasks_were_evaluated:
        summary["chatcore"] = round(chatcore_metric, 6)
    print0(format_record("summary", **summary))

    compute_cleanup()
