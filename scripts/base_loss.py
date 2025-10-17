"""
Loads a checkpoint, and:
- Evaluates the loss on a larger chunk of train/val splits
- Samples from the model

Example run as:
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
"""
import math
import os
import torch
from nanochat.checkpoint_manager import load_model
from nanochat.common import compute_init, print0, compute_cleanup
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.tokenizer import get_token_bytes
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine

# Configuration
_DEFAULT_DEVICE_BATCH_SIZE = 32
device_batch_size = _DEFAULT_DEVICE_BATCH_SIZE
split_tokens = 20*524288  # number of tokens to evaluate per split
model_tag = None # optional model tag for the output directory name
model_step = None # optional model step for the output directory name
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file

_device_batch_size_overridden = device_batch_size != _DEFAULT_DEVICE_BATCH_SIZE
if not _device_batch_size_overridden:
    env_device_batch_size = os.environ.get("NANOCHAT_DEVICE_BATCH_SIZE")
    if env_device_batch_size:
        device_batch_size = int(env_device_batch_size)
        _device_batch_size_overridden = True

if not _device_batch_size_overridden:
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_mem_gib = props.total_memory / (1024 ** 3)
            if total_mem_gib < 30:
                recommended = 16
            elif total_mem_gib < 60:
                recommended = 24
            else:
                recommended = device_batch_size
            if device_batch_size > recommended:
                print0(f"Auto-adjusting device_batch_size from {device_batch_size} to {recommended} for {total_mem_gib:.1f} GiB GPUs")
                device_batch_size = recommended
    except Exception as exc:
        print0(f"Warning: unable to auto-adjust device_batch_size ({exc})")

# Load the base model and the tokenizer
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=model_tag, step=model_step)
sequence_len = meta["model_config"]["sequence_len"] # could be arbitrary really

# Set up the precision we'll run with
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# Evaluate the loss on each split
tokens_per_step = device_batch_size * sequence_len * ddp_world_size
if split_tokens % tokens_per_step != 0:
    steps = math.ceil(split_tokens / tokens_per_step)
    adjusted_split_tokens = steps * tokens_per_step
    print0(
        f"Adjusting split_tokens from {split_tokens:,} to {adjusted_split_tokens:,} "
        "so it divides evenly across GPUs"
    )
    split_tokens = adjusted_split_tokens
else:
    steps = split_tokens // tokens_per_step
token_bytes = get_token_bytes(device=device)
bpb_results = {}
for split_name in ["train", "val"]:
    loader = tokenizing_distributed_data_loader(device_batch_size, sequence_len, split_name)
    with autocast_ctx:
        bpb = evaluate_bpb(model, loader, steps, token_bytes)
    print0(f"{split_name} bpb: {bpb:.4f}")
    bpb_results[split_name] = bpb

# Master process also samples from the model
samples = []
if ddp_rank == 0:
    prompts = [
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "The opposite of hot is",
        "The planets of the solar system are:",
        "My favorite color is",
        "If 5*x + 3 = 13, then x is",
    ]
    engine = Engine(model, tokenizer)
    for prompt in prompts:
        tokens = tokenizer(prompt, prepend="<|bos|>")
        with autocast_ctx:
            sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
        sample_str = tokenizer.decode(sample[0])
        print0(sample_str)
        samples.append(sample_str)

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model loss", data=[
    {
        "train bpb": bpb_results["train"],
        "val bpb": bpb_results["val"],
    },
    {f"sample {i}": sample for i, sample in enumerate(samples)},
])

# Cleanup
compute_cleanup()
