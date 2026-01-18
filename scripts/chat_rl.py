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
import re
import wandb
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# RL hyperparameters
run = "dummy" # wandb run name
source = "sft" # mid|sft
dtype = "bfloat16"
model_tag = None
device_type = "" # cuda|cpu|mps (empty => autodetect)
device_batch_size = 8 # no forward pass will go above this to not OOM
examples_per_step = 16 # in total and across all ranks (note: examples, not samples/completions!)
num_samples = 16 # number of samples per example (/question)
max_new_tokens = 256
temperature = 1.0
top_k = 50 # TODO: try None?
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
learning_rate = 9e-5
betas = (0.9, 0.95)
weight_decay = 0.0
init_lr_frac = 0.05
num_epochs = 1 # how many epochs of gsm8k to train on
save_every = 60 # every how many steps to save the model
eval_every = 60 # every how many steps to evaluate the model for val pass@k
eval_examples = 400 # number of examples used for evaluating pass@k

# Debug knobs for MoE loss components (defaults preserve existing behavior)
disable_aux_loss = False
disable_router_z_loss = False
override_aux_loss_weight = -1.0  # <0 means do not override
override_router_z_loss_weight = -1.0  # <0 means do not override

# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Init compute/precision
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=run, config=user_config)

# Init model and tokenizer
model, tokenizer, meta = load_model(source, device, phase="eval", model_tag=model_tag)
engine = Engine(model, tokenizer) # for sampling rollouts

# Optional overrides for MoE auxiliary losses (useful when total loss plateaus)
if hasattr(model, "config"):
    if disable_aux_loss and getattr(model.config, "n_exp", 1) > 1:
        print0("Disabling MoE aux loss for this midtraining run")
        model.config.use_aux_loss = False
    if disable_router_z_loss and getattr(model.config, "n_exp", 1) > 1:
        print0("Disabling MoE router z loss for this midtraining run")
        model.config.use_router_z_loss = False
    if override_aux_loss_weight >= 0 and getattr(model.config, "n_exp", 1) > 1:
        print0(f"Overriding MoE aux_loss_weight to {override_aux_loss_weight}")
        model.config.aux_loss_weight = float(override_aux_loss_weight)
    if override_router_z_loss_weight >= 0 and getattr(model.config, "n_exp", 1) > 1:
        print0(f"Overriding MoE router_z_loss_weight to {override_router_z_loss_weight}")
        model.config.router_z_loss_weight = float(override_router_z_loss_weight)

print0(f"MoE training loss is configured to use aux_loss: {getattr(model.config, 'use_aux_loss', False)} with weight {getattr(model.config, 'aux_loss_weight', 0.0)}, router_z_loss: {getattr(model.config, 'use_router_z_loss', False)} with weight {getattr(model.config, 'router_z_loss_weight', 0.0)}")

# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training

train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // examples_per_step) * num_epochs
print0(f"Calculated number of steps: {num_steps}")

@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>") # ok to use this token, it's only for padding and isn't used in the loss.
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size) # each rank is responsible for different examples in the training data
    for example_idx in itertools.cycle(rank_indices):

        # First get the full conversation of both user and assistant messages
        conversation = train_task[example_idx]

        # Tokenize the conversation, deleting the last Assistant message and priming the Assistant for a completion instead
        # (i.e. keep the <|assistant_start|>, but delete everything after it)
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate num_samples samples using batched generation, use loop to avoid OOMs
        model.eval() # ensure the model is in eval mode
        generated_token_sequences = []
        masks = []
        num_sampling_steps = num_samples // device_batch_size # go sequentially to prevent OOMs
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF # positive half of int32
            with autocast_ctx:
                generated_token_sequences_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=device_batch_size,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed, # must make sure to change the seed for each sampling step
                )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # Calculate the rewards for each sample
        rewards = []
        for sample_tokens in generated_token_sequences:
            # Get just the generated tokens (after the prompt)
            generated_tokens = sample_tokens[prefix_length:]
            # Decode the generated response
            generated_text = tokenizer.decode(generated_tokens)
            # Calculate the reward
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # Pad the sequences so that their lengths (in time) match
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        # Stack up the sequences and masks into PyTorch tensors
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        # Generate autoregressive inputs and targets to the Transformer
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone() # clone to avoid in-place modification:
        targets[mask_ids[:, 1:] == 0] = -1 # <-- inplace modification right here. -1 is the ignore index
        # NOTE also that the Engine returns mask=0 for BOTH the prompt tokens AND the tool use tokens.
        # So we will (correctly) end up not training on the prompt tokens, or the tool use forced tokens.
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # Calculate the advantages by simply subtracting the mean (instead of z-score (x-mu)/sigma)
        mu = rewards.mean()
        advantages = rewards - mu
        # yield inputs/targets as (B, T) of ids and rewards as (B,) of floats
        yield generated_token_sequences, inputs, targets, rewards, advantages

# -----------------------------------------------------------------------------
# Simple evaluation loop for GSM8K pass@k
def run_gsm8k_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    """
    Evaluates GSM8K task and returns a list of records of evaluation outcomes.
    In a distributed setting, all ranks cooperate but this function will NOT
    do the reduction across ranks. This is the responsibility of the caller.
    Because the evaluation can take a while, this function will yield records one by one.
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # Generate k samples using batched generation inside the Engine
        assert num_samples <= device_batch_size # usually this is true. we can add a loop if not...
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )
        # Check each sample for correctness
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({
                "is_correct": is_correct
            })
        # A bit bloated because I wanted to do more complex logging at one point.
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

# -----------------------------------------------------------------------------
# Training loop

# Init the optimizer
adamw_optimizer = model.configure_optimizers(
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    betas=betas,
    device_type=device_type,
)
optimizers = [adamw_optimizer]
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"] 
# optimizers = model.setup_optimizers(
#     unembedding_lr=unembedding_lr,
#     embedding_lr=embedding_lr,
#     matrix_lr=matrix_lr,
#     weight_decay=weight_decay,
# )

# # Set the initial learning rate as a fraction of the base learning rate
# for opt in optimizers:
#     for group in opt.param_groups:
#         group["lr"] = group["lr"] * init_lr_frac
#         group["initial_lr"] = group["lr"] # save the initial learning so we can decay easily later

# Learning rate scheduler: simple rampdown to zero over num_steps
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

# Calculate the number of examples each rank handles to achieve the desired examples_per_step
print0(f"Total sequences per step: {examples_per_step * num_samples}") # total batch size in sequences/step
assert examples_per_step % ddp_world_size == 0, "Desired examples per step must be divisible by the number of ranks"
examples_per_rank = examples_per_step // ddp_world_size # per GPU
print0(f"Calculated examples per rank: {examples_per_rank}")

# Kick off the training loop
batch_iterator = get_batch()
for step in range(num_steps):

    # Evaluate the model once in a while and log to wandb
    if step % eval_every == 0:
        model.eval()
        passk = torch.zeros(device_batch_size, device=device) # pass@k for k=1..device_batch_size
        with autocast_ctx:
            records_iter = run_gsm8k_eval(val_task, tokenizer, engine, num_samples=device_batch_size, max_examples=eval_examples, temperature=1.0)
            records = list(records_iter) # collect all records
        for k in range(1, device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item() # normalize by the total number of records
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, device_batch_size + 1)]
        print0(f"Step {step} | {', '.join(print_passk)}")
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, device_batch_size + 1)}
        wandb_run.log({
            "step": step,
            **log_passk,
        })

    # Forward/Backward on rollouts over multiple examples in the dataset
    rewards_list = []
    sequence_lengths = []

    # Track loss components for logging
    pg_loss_list = []
    aux_contrib_list = []
    z_contrib_list = []
    total_loss_list = []

    for example_step in range(examples_per_rank):
        # Get one batch corresponding to one example in the training dataset
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        # Evaluate the loss and gradients
        model.train() # ensure the model is in train mode
        # We need one more loop because we can never exceed the device_batch_size
        assert inputs_all.size(0) % device_batch_size == 0
        num_passes = inputs_all.size(0) // device_batch_size
        for pass_idx in range(num_passes):
            # Pluck out the batch for this pass
            b0, b1 = pass_idx * device_batch_size, (pass_idx + 1) * device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            # Calculate log probabilities. Note that the loss calculates NLL = -logp, so we negate
            with autocast_ctx:
                _, loss2d, aux_contrib, z_contrib = model(
                    inputs,
                    targets,
                    loss_reduction='none',
                    return_moe_losses=True,
                )
                logp = -loss2d.view_as(inputs) # (B, T)
            # Calculate the PG objective. Note that ignore_index=-1 ensures that invalid tokens have loss 0.
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            # normalize by the number of valid tokens, number of passes, and examples_per_rank
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            # Note, there is no need to add PPO ratio+clip because we are on policy
            # Finally, formulate the loss that we want to minimize (instead of objective we wish to maximize)
            pg_loss = -pg_obj

            # Add MoE routing regularizers as separate scalar terms.
            # We scale by num_passes/examples_per_rank to match the pg_loss normalization.
            moe_scale = 1.0 / (num_passes * examples_per_rank)
            aux_term = (aux_contrib if aux_contrib is not None else 0.0) * moe_scale
            z_term = (z_contrib if z_contrib is not None else 0.0) * moe_scale
            loss = pg_loss + aux_term + z_term

            loss.backward()

            # For logging (detach to avoid autograd sync issues)
            pg_loss_list.append(float(pg_loss.detach().item()))
            aux_contrib_list.append(float((aux_term.detach().item()) if torch.is_tensor(aux_term) else aux_term))
            z_contrib_list.append(float((z_term.detach().item()) if torch.is_tensor(z_term) else z_term))
            total_loss_list.append(float(loss.detach().item()))

            print0(
                f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | "
                f"loss: {loss.item():.6f} | pg: {pg_loss.item():.6f} | aux: {float(aux_term):.6f} | z: {float(z_term):.6f} | "
                f"Average reward: {rewards.mean().item()}"
            )
        # For logging
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # A bunch of logging for how the rollouts went this step
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)

    mean_pg_loss = sum(pg_loss_list) / max(len(pg_loss_list), 1)
    mean_aux = sum(aux_contrib_list) / max(len(aux_contrib_list), 1)
    mean_z = sum(z_contrib_list) / max(len(z_contrib_list), 1)
    mean_total_loss = sum(total_loss_list) / max(len(total_loss_list), 1)

    if ddp: # aggregate across ranks
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()

        mean_pg_loss_tensor = torch.tensor(mean_pg_loss, dtype=torch.float, device=device)
        mean_aux_tensor = torch.tensor(mean_aux, dtype=torch.float, device=device)
        mean_z_tensor = torch.tensor(mean_z, dtype=torch.float, device=device)
        mean_total_loss_tensor = torch.tensor(mean_total_loss, dtype=torch.float, device=device)
        dist.all_reduce(mean_pg_loss_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_aux_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_z_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_total_loss_tensor, op=dist.ReduceOp.AVG)
        mean_pg_loss = mean_pg_loss_tensor.item()
        mean_aux = mean_aux_tensor.item()
        mean_z = mean_z_tensor.item()
        mean_total_loss = mean_total_loss_tensor.item()
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}")
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
        "train/pg_loss": mean_pg_loss,
        "train/aux_loss_contrib": mean_aux,
        "train/router_z_loss_contrib": mean_z,
        "train/total_loss": mean_total_loss,
    })

    # Update the model parameters
    lrm = get_lr_multiplier(step)
    for opt in optimizers: # first set the learning rate
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    for opt in optimizers: # then step the optimizers
        opt.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # Master process saves the model once in a while. Skip first step. Save last step.
    if master_process and ((step > 0 and step % save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        # model_tag = f"d{depth}" # base the model tag on the depth of the base model
        if disable_aux_loss:
            aux_tag = "noaux"
        else:
            aux_tag = "aux"
        if disable_router_z_loss:
            z_tag = "noz"
        else:
            z_tag = "z"
        output_dirname = f"d{depth}_{aux_tag}_{z_tag}_lr{learning_rate}_model{model_tag}"
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
        model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # note: we don't bother to save the optimizer state
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Chat RL", data=[
    user_config, # CLI args
])

wandb_run.finish() # wandb run finish
compute_cleanup()
