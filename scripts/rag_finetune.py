"""
RAG Fine-tuning Script for Mamba and Hybrid Models

Fine-tune a pretrained model with retrieval-augmented generation.
Optimized for Mamba and hybrid (Transformer+Mamba) architectures.

Usage:
    # Single GPU
    python -m scripts.rag_finetune --knowledge_base data/kb

    # Multi-GPU
    torchrun --standalone --nproc_per_node=8 -m scripts.rag_finetune \
        --knowledge_base data/kb \
        --source mid \
        --retriever_type dense

Only works with Mamba or hybrid models (block_pattern must contain "M").
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import wandb
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb
from nanochat.checkpoint_manager import load_model, save_checkpoint
from nanochat.engine import Engine
from nanochat.retrieval import RetrievalManager
from nanochat.rag_utils import render_rag_conversation_for_tokenizer

from tasks.rag_task import RAGTask, create_rag_task
from tasks.smoltalk import SmolTalk
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# RAG Fine-tuning Hyperparameters
run = "dummy"  # wandb run name
# Model options
source = "mid"  # base|mid - which checkpoint to load
model_tag = None  # model tag to load
step = None  # step to load
# RAG options
knowledge_base = None  # REQUIRED: path to knowledge base
retriever_type = "simple"  # simple|dense
top_k = 5  # number of documents to retrieve
max_doc_length = 500  # max characters per document in prompt
insert_position = "before_user"  # where to insert retrieval
# Task options
base_tasks = "SmolTalk"  # comma-separated: SmolTalk,MMLU,ARC-Easy,GSM8K
task_samples = 10000  # samples per task (-1 = all)
# Training options
dtype = "bfloat16"
device_batch_size = 4  # smaller due to longer contexts with RAG
num_epochs = 1
max_iterations = -1
target_examples_per_step = 32
# Optimization
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02  # start with lower LR for stability
# Evaluation
eval_every = 100
eval_steps = 50
# Allow CLI overrides
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Validate
if knowledge_base is None:
    raise ValueError("--knowledge_base is required for RAG fine-tuning")

if not os.path.exists(knowledge_base):
    raise FileNotFoundError(f"Knowledge base not found: {knowledge_base}")

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
dtype_torch = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype_torch)

# WandB logging
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat-rag",
    name=run,
    config=user_config,
    save_code=True
)

# Load model and tokenizer
print0(f"Loading model from {source} checkpoint...")
model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)

# Validate model has Mamba blocks
block_pattern = model.config.block_pattern
if block_pattern is None or "M" not in "".join(block_pattern):
    raise ValueError(
        "RAG fine-tuning requires Mamba or hybrid models. "
        f"Current block_pattern: {block_pattern}. "
        "Please use a model with Mamba blocks (contains 'M')."
    )

print0(f"✓ Model has block pattern: {block_pattern}")
print0(f"  Transformer blocks: {block_pattern.count('T')}")
print0(f"  Mamba blocks: {block_pattern.count('M')}")

orig_model = model
# Don't compile for RAG (variable-length contexts)
# model = torch.compile(model, dynamic=True)

# Initialize retrieval manager
print0(f"Loading knowledge base from {knowledge_base}...")
print0(f"Using retriever type: {retriever_type}")
retrieval_manager = RetrievalManager(
    retriever_type=retriever_type,
    knowledge_base_path=knowledge_base
)
print0("✓ Knowledge base loaded")

# -----------------------------------------------------------------------------
# Create RAG tasks

print0(f"Creating RAG tasks from base tasks: {base_tasks}")
task_list = base_tasks.split(",")
train_rag_tasks = []
val_rag_tasks = []

for task_name in task_list:
    task_name = task_name.strip()
    print0(f"  Creating RAG wrapper for {task_name}...")
    
    # Create training task
    try:
        train_task = create_rag_task(
            task_name=task_name,
            split="train",
            knowledge_base_path=knowledge_base,
            retriever_type=retriever_type,
            top_k=top_k,
            stop=task_samples if task_samples > 0 else None
        )
        train_rag_tasks.append(train_task)
        print0(f"    Train: {len(train_task)} examples")
    except Exception as e:
        print0(f"    Warning: Could not create train task for {task_name}: {e}")
    
    # Create validation task
    try:
        val_task = create_rag_task(
            task_name=task_name,
            split="test" if task_name == "SmolTalk" else "val",
            knowledge_base_path=knowledge_base,
            retriever_type=retriever_type,
            top_k=top_k,
            stop=1000  # Limit validation size
        )
        val_rag_tasks.append(val_task)
        print0(f"    Val: {len(val_task)} examples")
    except Exception as e:
        print0(f"    Warning: Could not create val task for {task_name}: {e}")

# Combine tasks
from tasks.common import TaskMixture
train_ds = TaskMixture(train_rag_tasks) if len(train_rag_tasks) > 1 else train_rag_tasks[0]
val_ds = TaskMixture(val_rag_tasks) if len(val_rag_tasks) > 1 else (val_rag_tasks[0] if val_rag_tasks else train_rag_tasks[0])

print0(f"\n✓ Total training examples: {len(train_ds)}")
print0(f"✓ Total validation examples: {len(val_ds)}")

# -----------------------------------------------------------------------------
# DataLoader for RAG

def rag_data_generator(dataset, batch_size):
    """Data generator for RAG training with retrieved documents."""
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")
    
    def collate_and_yield(batch):
        """Collate RAG conversations into batch."""
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, :n-1] = row_targets
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        return inputs, targets
    
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            # Get RAG-augmented conversation
            conversation = dataset[i]
            
            # Render to tokens
            ids, mask = tokenizer.render_conversation(conversation)
            
            # Truncate if too long (RAG contexts can be long)
            max_len = 4096  # Allow longer contexts for Mamba
            if len(ids) > max_len:
                ids = ids[:max_len]
                mask = mask[:max_len]
            
            batch.append((ids, mask))
            
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

# Calculate gradient accumulation
examples_per_step = device_batch_size * ddp_world_size
print0(f"\nTraining configuration:")
print0(f"  Device batch size: {device_batch_size}")
print0(f"  Examples per step: {examples_per_step}")
assert target_examples_per_step % examples_per_step == 0
grad_accum_steps = target_examples_per_step // examples_per_step
print0(f"  Gradient accumulation steps: {grad_accum_steps}")

# Calculate iterations
num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
if max_iterations >= 0 and num_iterations > max_iterations:
    num_iterations = max_iterations
print0(f"  Number of iterations: {num_iterations}")

train_loader = rag_data_generator(train_ds, batch_size=device_batch_size)
build_val_loader = lambda: rag_data_generator(val_ds, batch_size=device_batch_size)

# -----------------------------------------------------------------------------
# Initialize optimizer

optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)

# Set initial LR
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"]

# -----------------------------------------------------------------------------
# Training loop

print0("\n" + "="*80)
print0("Starting RAG Fine-Tuning")
print0("="*80 + "\n")

def get_lr_multiplier(it):
    """Linear decay to 0."""
    return 1.0 - it / num_iterations

# Training loop
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
        
        for _ in range(eval_steps):
            val_inputs, val_targets = next(val_iter)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        
        val_loss = torch.stack(losses).mean()
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        val_loss = val_loss.item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        
        print0(f"Step {step:05d} | Val loss: {val_loss:.6f} | Best: {best_val_loss:.6f}")
        wandb_run.log({"step": step, "val_loss": val_loss, "best_val_loss": best_val_loss})
        model.train()
    
    if last_step:
        break
    
    # Training step
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_iter)
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
    
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
        train_loss_item = train_loss.item()
        print0(f"Step {step:05d}/{num_iterations:05d} | Train loss: {train_loss_item:.6f} | LR mult: {lrm:.4f}")
        wandb_run.log({"step": step, "train_loss": train_loss_item, "lrm": lrm})
    
    step += 1

# Save final model
if master_process:
    base_dir = get_base_dir()
    depth = model.config.n_layer
    model_tag_out = f"d{depth}_rag"
    checkpoint_dir = os.path.join(base_dir, "rag_checkpoints", model_tag_out)
    
    model_config_kwargs = {
        k: v for k, v in model.config.__dict__.items()
        if not k.startswith('_')
    }
    
    save_checkpoint(
        checkpoint_dir,
        step,
        orig_model.state_dict(),
        None,
        {
            "step": step,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "model_config": model_config_kwargs,
            "rag_config": {
                "knowledge_base": knowledge_base,
                "retriever_type": retriever_type,
                "top_k": top_k,
                "base_tasks": base_tasks
            }
        }
    )
    print0(f"\n✅ Saved RAG model to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="RAG Fine-Tuning", data=[
    user_config,
    {
        "Training examples": len(train_ds),
        "Number of iterations": num_iterations,
        "Final val loss": val_loss,
        "Best val loss": best_val_loss,
        "Knowledge base": knowledge_base,
        "Retriever type": retriever_type,
        "Top-k documents": top_k
    }
])

print0("\n" + "="*80)
print0("RAG Fine-Tuning Complete!")
print0("="*80)

# Cleanup
wandb_run.finish()
compute_cleanup()

