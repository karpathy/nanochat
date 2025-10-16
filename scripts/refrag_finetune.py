"""
REFRAG (Recursive Retrieval-Augmented Generation) Fine-tuning

Train models with multi-hop retrieval and reinforcement learning.
Optimized for Mamba and hybrid architectures.

Usage:
    torchrun --standalone --nproc_per_node=8 -m scripts.refrag_finetune \
        --knowledge_base data/kb \
        --max_hops 3 \
        --use_rewards true
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.distributed as dist
import wandb

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb
from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.retrieval import RetrievalManager
from nanochat.rag_utils import compute_rag_reward

from tasks.rag_task import MultiHopRAGTask
from tasks.smoltalk import SmolTalk

# -----------------------------------------------------------------------------
# REFRAG Hyperparameters
run = "dummy"
# Model
source = "mid"
model_tag = None
step = None
# RAG
knowledge_base = None  # REQUIRED
retriever_type = "dense"
max_hops = 3  # number of retrieval hops
top_k_per_hop = 3  # docs per hop
# RL options
use_rewards = True  # use RL-style rewards
reward_weight_answer = 0.6
reward_weight_relevance = 0.3
reward_weight_efficiency = 0.1
# Training
dtype = "bfloat16"
device_batch_size = 2  # smaller for multi-hop (longer contexts)
num_epochs = 1
max_iterations = 500  # REFRAG is expensive, limit iterations
target_examples_per_step = 16
# Optimization
unembedding_lr = 0.002  # lower LR for stability
embedding_lr = 0.1
matrix_lr = 0.01
weight_decay = 0.0
init_lr_frac = 0.01  # very conservative start
# Eval
eval_every = 50
eval_steps = 20
# CLI overrides
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

if knowledge_base is None:
    raise ValueError("--knowledge_base required")

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
dtype_torch = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype_torch)

# WandB
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat-refrag",
    name=run,
    config=user_config
)

# Load model
print0("Loading model...")
model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)

# Validate Mamba/hybrid
block_pattern = model.config.block_pattern
if block_pattern is None or "M" not in "".join(block_pattern):
    raise ValueError("REFRAG requires Mamba or hybrid models")

print0(f"✓ Model: {block_pattern.count('T')} transformer, {block_pattern.count('M')} Mamba blocks")

orig_model = model

# Load retrieval
print0(f"Loading knowledge base...")
retrieval_manager = RetrievalManager(
    retriever_type=retriever_type,
    knowledge_base_path=knowledge_base
)
print0("✓ Knowledge base loaded")

# Create multi-hop RAG task
print0(f"Creating multi-hop RAG task (max_hops={max_hops})...")
base_task = SmolTalk(split="train", stop=5000)  # Limit for REFRAG
train_task = MultiHopRAGTask(
    base_task=base_task,
    knowledge_base_path=knowledge_base,
    retriever_type=retriever_type,
    max_hops=max_hops,
    top_k_per_hop=top_k_per_hop
)

val_base = SmolTalk(split="test", stop=500)
val_task = MultiHopRAGTask(
    base_task=val_base,
    knowledge_base_path=knowledge_base,
    retriever_type=retriever_type,
    max_hops=max_hops,
    top_k_per_hop=top_k_per_hop
)

print0(f"✓ Train: {len(train_task)} examples")
print0(f"✓ Val: {len(val_task)} examples")

# DataLoader
def refrag_data_generator(dataset, batch_size):
    """Data generator for REFRAG (handles multi-hop retrieval)."""
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask, _ in batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        rewards_list = []
        
        for i, (ids, mask, reward) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :n-1] = row_targets
            
            rewards_list.append(reward)
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        rewards = torch.tensor(rewards_list, device=device, dtype=dtype_torch)
        
        return inputs, targets, rewards
    
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            conversation = dataset[i]
            ids, mask = tokenizer.render_conversation(conversation)
            
            # Truncate if needed
            max_len = 6144  # Allow longer for multi-hop
            if len(ids) > max_len:
                ids = ids[:max_len]
                mask = mask[:max_len]
            
            # Compute reward if using RL
            reward = 1.0  # default
            if use_rewards:
                # Simple reward: based on conversation structure
                # In full RL, would compare generated vs ground truth
                reward = compute_refrag_reward(conversation)
            
            batch.append((ids, mask, reward))
            
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

def compute_refrag_reward(conversation):
    """Compute reward for REFRAG training."""
    messages = conversation.get("messages", [])
    
    # Check if retrieval was successful
    has_retrieval = any(msg.get("role") == "retrieval" for msg in messages)
    if not has_retrieval:
        return 0.5  # penalty for no retrieval
    
    # Check if multi-hop
    retrieval_msg = next((m for m in messages if m.get("role") == "retrieval"), None)
    if retrieval_msg and retrieval_msg.get("multi_hop"):
        hops = retrieval_msg.get("hops", [])
        num_hops = len(hops)
        # Reward more hops (up to max_hops)
        hop_reward = min(num_hops / max_hops, 1.0)
    else:
        hop_reward = 0.3  # penalty for single-hop
    
    # Combine rewards
    return 0.5 + 0.5 * hop_reward

# Training setup
examples_per_step = device_batch_size * ddp_world_size
grad_accum_steps = target_examples_per_step // examples_per_step
num_iterations = min(max_iterations, (len(train_task) // target_examples_per_step) * num_epochs)

print0(f"\nTraining configuration:")
print0(f"  Device batch size: {device_batch_size}")
print0(f"  Gradient accumulation: {grad_accum_steps}")
print0(f"  Iterations: {num_iterations}")

train_loader = refrag_data_generator(train_task, device_batch_size)
build_val_loader = lambda: refrag_data_generator(val_task, device_batch_size)

# Optimizer
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay
)

for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"]

# Training loop
print0("\n" + "="*80)
print0("Starting REFRAG Training (Multi-hop RAG with RL)")
print0("="*80 + "\n")

def get_lr_multiplier(it):
    return 1.0 - it / num_iterations

step = 0
train_iter = iter(train_loader)
best_val_loss = float('inf')

for step in range(num_iterations):
    last_step = step == num_iterations - 1
    
    # Validation
    if last_step or step % eval_every == 0:
        model.eval()
        val_iter = iter(build_val_loader())
        losses = []
        rewards_list = []
        
        for _ in range(eval_steps):
            val_inputs, val_targets, val_rewards = next(val_iter)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            losses.append(loss)
            rewards_list.append(val_rewards.mean())
        
        val_loss = torch.stack(losses).mean()
        avg_reward = torch.stack(rewards_list).mean()
        
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(avg_reward, op=dist.ReduceOp.AVG)
        
        val_loss = val_loss.item()
        avg_reward = avg_reward.item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        print0(f"Step {step:05d} | Val loss: {val_loss:.6f} | Reward: {avg_reward:.4f} | Best: {best_val_loss:.6f}")
        wandb_run.log({"step": step, "val_loss": val_loss, "avg_reward": avg_reward})
        model.train()
    
    if last_step:
        break
    
    # Training step with reward weighting
    total_loss = 0
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets, train_rewards = next(train_iter)
        
        with autocast_ctx:
            loss = model(train_inputs, train_targets, loss_reduction='none')  # per-example loss
            
            if use_rewards:
                # Weight loss by rewards (RL-style)
                weighted_loss = (loss * train_rewards).mean()
            else:
                weighted_loss = loss.mean()
        
        train_loss = weighted_loss.detach()
        total_loss += train_loss
        weighted_loss = weighted_loss / grad_accum_steps
        weighted_loss.backward()
    
    # Update
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    
    # Logging
    if step % 10 == 0:
        avg_loss = (total_loss / grad_accum_steps).item()
        print0(f"Step {step:05d}/{num_iterations:05d} | Train loss: {avg_loss:.6f} | LR: {lrm:.4f}")
        wandb_run.log({"step": step, "train_loss": avg_loss, "lrm": lrm})

# Save
if master_process:
    base_dir = get_base_dir()
    depth = model.config.n_layer
    model_tag_out = f"d{depth}_refrag"
    checkpoint_dir = os.path.join(base_dir, "refrag_checkpoints", model_tag_out)
    
    model_config_kwargs = {k: v for k, v in model.config.__dict__.items() if not k.startswith('_')}
    
    save_checkpoint(
        checkpoint_dir,
        step,
        orig_model.state_dict(),
        None,
        {
            "step": step,
            "val_loss": val_loss,
            "model_config": model_config_kwargs,
            "refrag_config": {
                "knowledge_base": knowledge_base,
                "max_hops": max_hops,
                "use_rewards": use_rewards
            }
        }
    )
    print0(f"\n✅ Saved REFRAG model to {checkpoint_dir}")

print0("\n" + "="*80)
print0("REFRAG Training Complete!")
print0("="*80)

# Cleanup
wandb_run.finish()
compute_cleanup()

