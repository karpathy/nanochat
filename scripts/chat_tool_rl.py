"""
Lightweight RL stage for tool-use tuning on local ToolJSON datasets.

1 GPU:
python -m scripts.chat_tool_rl --train-jsonl seed_data/tool_eval_seed.jsonl --eval-jsonl seed_data/tool_eval_seed.jsonl

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_tool_rl -- --train-jsonl seed_data/tool_eval_seed.jsonl --eval-jsonl seed_data/tool_eval_seed.jsonl
"""

import argparse
import itertools
import os

import torch
import torch.distributed as dist
import wandb

from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.common import DummyWandb, autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from nanochat.engine import Engine
from nanochat.tools import DEFAULT_TOOL_SCHEMA
from tasks.tool_json import ToolJSON


parser = argparse.ArgumentParser(description="RL tuning on local tool-use JSONL tasks")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--model-tag", type=str, default=None, help="SFT model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="SFT model step to load from")
parser.add_argument("--train-jsonl", type=str, required=True, help="Training ToolJSONL file")
parser.add_argument("--eval-jsonl", type=str, default=None, help="Evaluation ToolJSONL file")
parser.add_argument("--num-epochs", type=int, default=1, help="Number of epochs over ToolJSON")
parser.add_argument("--device-batch-size", type=int, default=8, help="Max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=8, help="Examples per optimization step across all ranks")
parser.add_argument("--num-samples", type=int, default=8, help="Number of samples per example")
parser.add_argument("--max-new-tokens", type=int, default=256, help="Max tokens to generate")
parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (0 disables)")
parser.add_argument("--embedding-lr", type=float, default=0.2, help="Embedding LR")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="Unembedding LR")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="Matrix LR")
parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="Initial LR multiplier")
parser.add_argument("--eval-every", type=int, default=40, help="Evaluate every N steps")
parser.add_argument("--eval-examples", type=int, default=64, help="Maximum eval examples")
parser.add_argument("--save-every", type=int, default=40, help="Save every N steps")
args = parser.parse_args()
user_config = vars(args).copy()


device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-tool-rl", name=args.run, config=user_config)

model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer)

train_task = ToolJSON(filepath=args.train_jsonl)
eval_task = ToolJSON(filepath=args.eval_jsonl or args.train_jsonl)
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")


@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
    for example_idx in itertools.cycle(rank_indices):
        conversation = train_task[example_idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        model.eval()
        generated_token_sequences = []
        masks = []
        num_sampling_steps = args.num_samples // args.device_batch_size
        if args.num_samples % args.device_batch_size != 0:
            raise ValueError("num_samples must be divisible by device_batch_size")
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            generated_batch, mask_batch = engine.generate_batch(
                tokens,
                num_samples=args.device_batch_size,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=seed,
            )
            generated_token_sequences.extend(generated_batch)
            masks.extend(mask_batch)

        rewards = []
        for sample_tokens in generated_token_sequences:
            generated_text = tokenizer.decode(sample_tokens[prefix_length:])
            rewards.append(train_task.reward(conversation, generated_text))

        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        ids = torch.tensor(padded_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = -1
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)
        advantages = rewards_tensor - rewards_tensor.mean()
        yield generated_token_sequences, inputs, targets, rewards_tensor, advantages


@torch.no_grad()
def run_tool_eval(task, max_examples):
    max_examples = min(max_examples, len(task))
    total = 0
    reward_sum = 0.0
    passed = 0
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        generated_sequences, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=args.max_new_tokens, temperature=0.0, top_k=args.top_k)
        generated_text = tokenizer.decode(generated_sequences[0][len(tokens):])
        reward_sum += task.reward(conversation, generated_text)
        passed += task.evaluate(conversation, generated_text)
        total += 1

    reward_tensor = torch.tensor([reward_sum], dtype=torch.float, device=device)
    passed_tensor = torch.tensor([passed], dtype=torch.long, device=device)
    total_tensor = torch.tensor([total], dtype=torch.long, device=device)
    if ddp:
        dist.all_reduce(reward_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    total = max(total_tensor.item(), 1)
    return reward_tensor.item() / total, passed_tensor.item() / total


optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]


def get_lr_multiplier(it):
    return 1.0 - it / max(num_steps, 1)


assert args.examples_per_step % ddp_world_size == 0, "examples_per_step must be divisible by number of ranks"
examples_per_rank = args.examples_per_step // ddp_world_size
batch_iterator = get_batch()

for step in range(num_steps):
    if step % args.eval_every == 0:
        model.eval()
        mean_reward, pass_rate = run_tool_eval(eval_task, args.eval_examples)
        print0(f"Step {step} | tool_eval_reward={mean_reward:.4f} | tool_eval_pass={pass_rate:.4f}")
        wandb_run.log({"step": step, "tool_eval_reward": mean_reward, "tool_eval_pass": pass_rate})

    rewards_list = []
    for example_step in range(examples_per_rank):
        _, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        model.train()
        assert inputs_all.size(0) % args.device_batch_size == 0
        num_passes = inputs_all.size(0) // args.device_batch_size
        for pass_idx in range(num_passes):
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            logp = -model(inputs, targets, loss_reduction="none").view_as(inputs)
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            loss = -pg_obj
            loss.backward()
            print0(
                f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} "
                f"| loss={loss.item():.6f} | reward={rewards.mean().item():.4f}"
            )
        rewards_list.append(rewards_all.mean().item())

    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({"step": step, "lrm": lrm, "mean_reward": sum(rewards_list) / max(len(rewards_list), 1)})

    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        output_dirname = args.model_tag if args.model_tag else f"d{model.config.n_layer}"
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None,
            {
                "step": step,
                "stage": "tool_rl",
                "model_config": model.config.__dict__,
                "user_config": user_config,
                "tool_schema": DEFAULT_TOOL_SCHEMA,
                "source_hf_repo": meta.get("source_hf_repo"),
                "train_jsonl": args.train_jsonl,
                "eval_jsonl": args.eval_jsonl,
            },
        )
        print0(f"Saved tool RL checkpoint to {checkpoint_dir}")

wandb_run.finish()
compute_cleanup()
