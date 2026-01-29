"""
Finetune a base model to be a chat model.
Run on one GPU e.g. for debugging:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
"""

import argparse
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import torch
import torch.distributed as dist
from contextlib import nullcontext

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.engine import Engine
from scripts.chat_eval import run_chat_eval

from tasks.common import TaskMixture
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Supervised finetuning for chat")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument("--source", type=str, default="mid", help="base|mid - which checkpoint to load from")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs")
parser.add_argument("--num-iterations", type=int, default=-1, help="override number of iterations (-1 = use num_epochs)")
# Batch sizes
parser.add_argument("--device-batch-size", type=int, default=4, help="per-device batch size")
parser.add_argument("--target-examples-per-step", type=int, default=32, help="target examples per optimization step")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.02, help="initial LR as fraction of base LR")
# Evaluation
parser.add_argument("--eval-every", type=int, default=100, help="evaluate val loss every N steps")
parser.add_argument("--eval-steps", type=int, default=100, help="number of batches for val loss evaluation")
parser.add_argument("--eval-metrics-every", type=int, default=200, help="evaluate accuracy metrics every N steps")
parser.add_argument("--eval-metrics-max-problems", type=int, default=1024, help="max problems per metric evaluation")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=args.run, config=user_config, save_code=True)

# Load the model and tokenizer
model, tokenizer, meta = load_model(args.source, device, phase="train", model_tag=args.model_tag, step=args.model_step)
orig_model = model # original, uncompiled model
# model = torch.compile(model, dynamic=True) # doesn't work super well because of variable lengths of inputs
engine = Engine(model, tokenizer) # will be used for inline model evaluation only

# -----------------------------------------------------------------------------
# Task data mixture we'll train on
identity_conversations_filepath = os.path.join(get_base_dir(), "identity_conversations.jsonl")
train_ds = TaskMixture([
    ARC(subset="ARC-Easy", split="train"), # 2.3K rows
    ARC(subset="ARC-Challenge", split="train"), # 1.1K rows
    GSM8K(subset="main", split="train"), # 8K rows
    SmolTalk(split="train", stop=10_000), # 10K rows of smoltalk
    CustomJSON(filepath=identity_conversations_filepath), # 1K rows of synthetic identity conversations
    SimpleSpelling(size=300, split="train"), # 300 rows of Simple Spelling (e.g. spell the word 'apple')
    SpellingBee(size=300, split="train"), # 300 rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
]) # 2.3K + 1.1K + 8K + 10K + 1K + 0.3K + 0.3K = 23K rows
val_ds = SmolTalk(split="test") # general conversations, 24K rows (though we don't actually use all of it)

# -----------------------------------------------------------------------------
# DataLoader

def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>") # use <|assistant_end|> as the pad token is ok, these positions are masked in the loss
    # prepares a list of tokenized conversations into a batch and yields
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1 # seq of n creates inputs/targets of n-1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long) # -1 is ignore index
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, :n-1] = ids_tensor[:-1]
            # recall -1 is the ignore index, so mask out targets where mask is 0
            row_targets = ids_tensor[1:]
            # mask[1:] omits the mask for the BOS token, which is never a target atm so it's ok
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1 # mask out targets where mask is 0
            targets[i, :n-1] = row_targets
        inputs = inputs.to(device) # move to device
        targets = targets.to(device)
        return inputs, targets
    # iterates over the dataset in epochs, tokenizes
    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

examples_per_step = args.device_batch_size * ddp_world_size
print0(f"Target examples per step: {args.target_examples_per_step}")
print0(f"Device batch size: {args.device_batch_size}")
print0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}")
assert args.target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
grad_accum_steps = args.target_examples_per_step // examples_per_step
print0(f"=> Setting grad accum steps: {grad_accum_steps}")

if args.num_iterations == -1:
    # derive num_iterations from num_epochs and the size of the dataset
    assert args.num_epochs > 0, "num_epochs must be positive if num_iterations is -1"
    num_iterations = (len(train_ds) // args.target_examples_per_step) * args.num_epochs
else:
    num_iterations = args.num_iterations
train_loader = sft_data_generator(train_ds, batch_size=args.device_batch_size)
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=args.device_batch_size)

# -----------------------------------------------------------------------------
# Initialize the Optimizer

optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)
# Set the initial learning rate as a fraction of the base learning rate
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# -----------------------------------------------------------------------------
# Training loop

# Learning rate scheduler
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_iterations
    return lrm

# Go!
step = 0
for step in range(num_iterations):
    last_step = step == num_iterations - 1

    # evaluate the validation loss
    if last_step or step % args.eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        losses = []
        for _ in range(args.eval_steps):
            val_inputs, val_targets = next(val_loader)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        val_loss = torch.stack(losses).mean() # average over eval_steps
        if ddp:
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG) # average over ranks
        val_loss = val_loss.item()
        print0(f"Step {step:05d} | Validation loss: {val_loss:.6f}")
        wandb_run.log({
            "step": step,
            "val_loss": val_loss,
        })
        model.train()

    # evaluate accuracy of the multiple choice tasks (which are quick to run)
    if last_step or (step > 0 and step % args.eval_metrics_every == 0):
        model.eval()
        metrics = {}
        with torch.no_grad(), autocast_ctx:
            # note that because these are inside no_grad, we can usually afford to at least ~2X the batch size
            metrics["mmlu_acc"] = run_chat_eval("MMLU", model, tokenizer, engine, batch_size=args.device_batch_size*2, max_problems=args.eval_metrics_max_problems)
            metrics["arc_easy_acc"] = run_chat_eval("ARC-Easy", model, tokenizer, engine, batch_size=args.device_batch_size*2, max_problems=args.eval_metrics_max_problems)
        metrics_str = ', '.join(f'{k}: {v:.6f}' for k, v in metrics.items())
        print0(f"Step {step:05d} | {metrics_str}")
        wandb_run.log({
            "step": step,
            **metrics,
        })
        model.train()

    if last_step:
        break

    # evaluate the gradient
    steps = [next(train_loader) for _ in range(grad_accum_steps)]
    num_tokens = torch.tensor(0, device=device) # the number of "active" tokens of supervision seen
    num_tokens += sum((targets >= 0).sum() for _, targets in steps)

    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM) # sum over ranks

    for micro_step, (train_inputs, train_targets) in enumerate(steps):
        with autocast_ctx:
            loss = model(train_inputs, train_targets, loss_reduction='sum')
        loss = loss / num_tokens # normalize loss here
        train_loss = loss.detach() # for logging
        loss.backward() # accumulate the gradient

    # learning rate scheduler
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    # step the optimizer
    optimizer.step()
    model.zero_grad(set_to_none=True)

    # logging
    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()
    print0(f"Step {step:05d}/{num_iterations:05d} | Training loss: {train_loss_item:.6f}| lrm: {lrm:.6f}| num_tokens: {num_tokens_item:,}")
    wandb_run.log({
        "step": step,
        "lrm": lrm,
        "train_loss": train_loss_item,
        "num_tokens": num_tokens_item,
    })
    step += 1

# Save the model at the end of the run
if master_process:
    base_dir = get_base_dir()
    depth = model.config.n_layer
    output_dirname = args.model_tag if args.model_tag else f"d{depth}" # e.g. d12
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
    model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        None, # note: we don't bother to save the optimizer state
        {
            "step": step,
            "val_loss": val_loss,
            **metrics,
            "model_config": model_config_kwargs,
        }
    )
    print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Chat SFT", data=[
    user_config, # CLI args
    {
        "Training rows": len(train_ds),
        "Number of iterations": num_iterations,
        "Training loss": train_loss_item,
        "Validation loss": val_loss,
    },
])

# Cleanup
wandb_run.finish()
compute_cleanup()
