"""
Reinforcement learning on GSM8K via "GRPO".

I put GRPO in quotes because we actually end up with something a lot
simpler and more similar to just REINFORCE:

1) Delete trust region, so there is no KL regularization to a reference model
2) We are on policy, so there's no need for PPO ratio+clip.
3) We use GAPO style normalization that is token-level, not sequence-level.
4) Instead of z-score normalization (r - mu)/sigma, only use (r - mu) as the advantage.

1 GPU:
python -m scripts.chat_rl

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
"""

import os
import itertools
import argparse
import wandb
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Chat RL training (GRPO/REINFORCE on GSM8K)")
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb)")
parser.add_argument("--source", type=str, default="sft", help="mid|sft — which checkpoint to load")
parser.add_argument("--device-batch-size", type=int, default=8, help="max forward pass batch size to avoid OOM")
parser.add_argument("--examples-per-step", type=int, default=16, help="total examples per step across all ranks")
parser.add_argument("--num-samples", type=int, default=16, help="number of samples per example/question")
parser.add_argument("--max-new-tokens", type=int, default=256, help="max new tokens per generation")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding (Adam)")
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over GSM8K")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps")
parser.add_argument("--eval-examples", type=int, default=400, help="number of examples for pass@k eval")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Init compute
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
dtype = torch.bfloat16

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=args.run, config=user_config)

# Init model and tokenizer
model, tokenizer, meta = load_model(args.source, device, phase="eval")
engine = Engine(model, tokenizer)

device_batch_size = args.device_batch_size
examples_per_step = args.examples_per_step
num_samples = args.num_samples
max_new_tokens = args.max_new_tokens
temperature = args.temperature
top_k = args.top_k

# -----------------------------------------------------------------------------
# Rollout / sampling generator loop

train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // examples_per_step) * args.num_epochs
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
        num_sampling_steps = num_samples // device_batch_size
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            generated_token_sequences_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=device_batch_size,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
            )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        rewards = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = -1
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        mu = rewards.mean()
        advantages = rewards - mu
        yield generated_token_sequences, inputs, targets, rewards, advantages

# -----------------------------------------------------------------------------
# Evaluation loop for GSM8K pass@k
def run_gsm8k_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        assert num_samples <= device_batch_size
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({"is_correct": is_correct})
        record = {"idx": idx, "outcomes": outcomes}
        yield record

# -----------------------------------------------------------------------------
# Training loop

optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]

def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

print0(f"Total sequences per step: {examples_per_step * num_samples}")
assert examples_per_step % ddp_world_size == 0, "examples_per_step must be divisible by world_size"
examples_per_rank = examples_per_step // ddp_world_size
print0(f"Calculated examples per rank: {examples_per_rank}")

save_every = args.save_every
eval_every = args.eval_every
eval_examples = args.eval_examples

batch_iterator = get_batch()
for step in range(num_steps):

    # Evaluate pass@k
    if step % eval_every == 0:
        model.eval()
        passk = torch.zeros(device_batch_size, device=device)
        records_iter = run_gsm8k_eval(val_task, tokenizer, engine, num_samples=device_batch_size, max_examples=eval_examples, temperature=1.0)
        records = list(records_iter)
        for k in range(1, device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item()
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, device_batch_size + 1)]
        print0(f"Step {step} | {', '.join(print_passk)}")
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, device_batch_size + 1)}
        wandb_run.log({"step": step, **log_passk})

    # Forward/Backward on rollouts
    rewards_list = []
    sequence_lengths = []
    for example_step in range(examples_per_rank):
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        model.train()
        assert inputs_all.size(0) % device_batch_size == 0
        num_passes = inputs_all.size(0) // device_batch_size
        for pass_idx in range(num_passes):
            b0, b1 = pass_idx * device_batch_size, (pass_idx + 1) * device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            _, nll, _ = model(inputs, targets, loss_reduction='none')
            logp = -nll.view_as(inputs)
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            loss = -pg_obj
            loss.backward()
            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}")
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # Logging
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    if ddp:
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}")
    wandb_run.log({"step": step, "reward": mean_reward, "sequence_length": mean_sequence_length})

    # Update parameters
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({"step": step, "lrm": lrm})

    # Save checkpoint
    if master_process and ((step > 0 and step % save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        model_tag = f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", model_tag)
        model_config_kwargs = model.config.__dict__
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None,
            {"model_config": model_config_kwargs},
        )
        print(f"Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report  # noqa: E402
get_report().log(section="Chat RL", data=[user_config])

wandb_run.finish()
compute_cleanup()
