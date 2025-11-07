"""
Finetune a base model to be a chat model.
Run on one GPU e.g. for debugging:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import wandb
import torch
import torch.distributed as dist
from contextlib import nullcontext

from nanochat.common import compute_init, compute_cleanup, get_base_dir, print0, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import HuggingFaceTokenizer
from scripts.chat_eval import run_chat_eval

from tasks.common import TaskMixture
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# -----------------------------------------------------------------------------
# SFT specific functions

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
]

def render_conversation(tokenizer, doc):
    ids = [tokenizer.encode("<|bos|>")[0]]
    mask = [0] # BOS token is never a target
    for message in doc["messages"]:
        if message["role"] == "user":
            ids.extend(tokenizer.encode("<|user_start|>"))
            mask.extend([0] * len(tokenizer.encode("<|user_start|>")))
            ids.extend(tokenizer.encode(message["content"]))
            mask.extend([0] * len(tokenizer.encode(message["content"])))
            ids.extend(tokenizer.encode("<|user_end|>"))
            mask.extend([0] * len(tokenizer.encode("<|user_end|>")))
        elif message["role"] == "assistant":
            ids.extend(tokenizer.encode("<|assistant_start|>"))
            mask.extend([0] * len(tokenizer.encode("<|assistant_start|>")))
            ids.extend(tokenizer.encode(message["content"]))
            mask.extend([1] * len(tokenizer.encode(message["content"])))
            ids.extend(tokenizer.encode("<|assistant_end|>"))
            mask.extend([0] * len(tokenizer.encode("<|assistant_end|>")))
        elif message["role"] == "python":
            ids.extend(tokenizer.encode("<|python_start|>"))
            mask.extend([0] * len(tokenizer.encode("<|python_start|>")))
            ids.extend(tokenizer.encode(message["content"]))
            mask.extend([0] * len(tokenizer.encode(message["content"])))
            ids.extend(tokenizer.encode("<|python_end|>"))
            mask.extend([0] * len(tokenizer.encode("<|python_end|>")))
        elif message["role"] == "output":
            ids.extend(tokenizer.encode("<|output_start|>"))
            mask.extend([0] * len(tokenizer.encode("<|output_start|>")))
            ids.extend(tokenizer.encode(message["content"]))
            mask.extend([0] * len(tokenizer.encode(message["content"])))
            ids.extend(tokenizer.encode("<|output_end|>"))
            mask.extend([0] * len(tokenizer.encode("<|output_end|>")))
    return ids, mask

# -----------------------------------------------------------------------------
# SFT Hyperparameters
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# input model options
source = "mid" # base|mid , which checkpoint to load the model from (base model or midtrained model)
model_tag = None # model tag to load the model from (base model or midtrained model)
step = None # step to load the model from (base model or midtrained model)
# compute/precision
device_type = "" # cuda|cpu|mps (empty => autodetect)
dtype = "bfloat16"
device_batch_size = 4 # max to avoid OOM
# optimization
num_epochs = 1
num_iterations = -1 # override number of iterations (-1 = disable, use num_epochs to derive it)
target_examples_per_step = 32
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.02
# evaluation and logging there of
eval_every = 100
eval_steps = 100
eval_metrics_every = 200
eval_metrics_max_problems = 1024
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # possibly useful for logging
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-sft", name=run, config=user_config, save_code=True)

# Load the model and tokenizer
model_dir = {
    "base": "base_checkpoints",
    "mid": "mid_checkpoints",
    "sft": "chatsft_checkpoints",
    "rl": "chatrl_checkpoints",
}[source]
base_dir = get_base_dir()
checkpoints_dir = os.path.join(base_dir, model_dir)

if os.path.exists(checkpoints_dir) and len(os.listdir(checkpoints_dir)) > 0:
    model, tokenizer, meta = load_model(source, device, phase="train", model_tag=model_tag, step=step)
else:
    print0(f"No checkpoints found in {checkpoints_dir}. Building a new model.")
    # Initialize a new GPT model and tokenizer
    model_config = GPTConfig()
    model = GPT(model_config)
    tokenizer = HuggingFaceTokenizer.from_pretrained("gpt2") # Using gpt2 as a default pretrained tokenizer
    meta = {"model_config": model_config.__dict__}

orig_model = model # original, uncompiled model
engine = Engine(model_config) # will be used for inline model evaluation only

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
    pad_token_id = tokenizer.encode("<|assistant_end|>")[0] # use <|assistant_end|> as the pad token is ok, these positions are masked in the loss
    # prepares a list of tokenized conversations into a batch and yields
    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1 # seq of n creates inputs/targets of n-1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long) # -1 is ignore index
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            print(f"ids_tensor max: {ids_tensor.max()}, min: {ids_tensor.min()}")
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
            ids, mask = render_conversation(tokenizer, doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []

examples_per_step = device_batch_size * ddp_world_size
print0(f"Target examples per step: {target_examples_per_step}")
print0(f"Device batch size: {device_batch_size}")
print0(f"Examples per step is device_batch_size * ddp_world_size: {examples_per_step}")
assert target_examples_per_step % examples_per_step == 0, "Target examples per step must be divisible by examples per step"
grad_accum_steps = target_examples_per_step // examples_per_step
print0(f"=> Setting grad accum steps: {grad_accum_steps}")

if num_iterations == -1:
    # derive num_iterations from num_epochs and the size of the dataset
    assert num_epochs > 0, "num_epochs must be positive if num_iterations is -1"
    num_iterations = (len(train_ds) // target_examples_per_step) * num_epochs
train_loader = sft_data_generator(train_ds, batch_size=device_batch_size)
build_val_loader = lambda: sft_data_generator(val_ds, batch_size=device_batch_size)

# -----------------------------------------------------------------------------
# Initialize the Optimizer

optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)
# Set the initial learning rate as a fraction of the base learning rate
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"] # save the initial learning so we can decay easily later

# -----------------------------------------------------------------------------
# Training loop

# Learning rate scheduler
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_iterations
    return lrm

# Go!
step = 0
train_iter = iter(train_loader)
for step in range(num_iterations):
    last_step = step == num_iterations - 1

    # evaluate the validation loss
    if last_step or step % eval_every == 0:
        model.eval()
        val_iter = iter(build_val_loader())
        losses = []
        for _ in range(eval_steps):
            val_inputs, val_targets = next(val_iter)
            with torch.no_grad(), autocast_ctx:
                total_loss, lm_loss, recon_loss, energy_loss, _, _ = engine(val_inputs, val_targets)
            losses.append(total_loss)
        val_total_loss = torch.stack(losses).mean() # average over eval_steps
        if ddp:
            dist.all_reduce(val_total_loss, op=dist.ReduceOp.AVG) # average over ranks
        val_total_loss = val_total_loss.item()
        print0(f"Step {step:05d} | Validation total loss: {val_total_loss:.6f}")
        wandb_run.log({
            "step": step,
            "val_total_loss": val_total_loss,
            "val_lm_loss": lm_loss.item(),
            "val_recon_loss": recon_loss.item(),
            "val_energy_loss": energy_loss.item(),
        })
        model.train()

    # evlauate accuracy of the multiple choice tasks (which are quick to run)
    if last_step or (step > 0 and step % eval_metrics_every == 0):
        model.eval()
        metrics = {}
        with torch.no_grad(), autocast_ctx:
            # note that because these are inside no_grad, we can usually afford to at least ~2X the batch size
            metrics["mmlu_acc"] = run_chat_eval("MMLU", model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=eval_metrics_max_problems)
            metrics["arc_easy_acc"] = run_chat_eval("ARC-Easy", model, tokenizer, engine, batch_size=device_batch_size*2, max_problems=eval_metrics_max_problems)
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
    num_tokens = torch.tensor(0, device=device) # the number of "active" tokens of supervision seen
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_iter)
        with autocast_ctx:
            total_loss, lm_loss, recon_loss, energy_loss = engine(train_inputs, train_targets)
        train_total_loss = total_loss.detach() # for logging
        train_lm_loss = lm_loss.detach()
        train_recon_loss = recon_loss.detach()
        train_energy_loss = energy_loss.detach()
        total_loss = total_loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        total_loss.backward() # accumulate the gradient
        num_tokens += (train_targets >= 0).sum()
    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM) # sum over ranks

    # learning rate scheduler
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # step the optimizers
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    # logging
    num_tokens_item = num_tokens.item()
    print0(f"Step {step:05d}/{num_iterations:05d} | Training total loss: {train_total_loss.item():.6f} | LM loss: {train_lm_loss.item():.6f} | Recon loss: {train_recon_loss.item():.6f} | Energy loss: {train_energy_loss.item():.6f} | lrm: {lrm:.6f} | num_tokens: {num_tokens_item:,}")
    wandb_run.log({
        "step": step,
        "lrm": lrm,
        "train_total_loss": train_total_loss.item(),
        "train_lm_loss": train_lm_loss.item(),
        "train_recon_loss": train_recon_loss.item(),
        "train_energy_loss": train_energy_loss.item(),
        "num_tokens": num_tokens_item,
    })
    step += 1

# Save the model at the end of the run
if master_process:
    base_dir = get_base_dir()
    depth = model.config.n_layer
    model_tag = f"d{depth}" # base the model tag on the depth of the base model
    checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", model_tag)
    model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        None, # note: we don't bother to save the optimizer state
        {
            "step": step,
            "val_total_loss": val_total_loss,
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
        "Training total loss": train_total_loss.item(),
        "Validation total loss": val_total_loss,
    },
])

# Cleanup
wandb_run.finish()
compute_cleanup()
