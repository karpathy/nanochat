from collections import deque
import os
import numpy as np

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files, USE_OPENWEBTEXT
from nanochat.tokenizer import get_tokenizer

# Support for loading openwebtext from HuggingFace datasets
if USE_OPENWEBTEXT:
    from datasets import load_dataset

def bin_data_loader_with_state(B, T, data_dir, split="train", device="cuda", resume_state_dict=None):
    """
    Load data from .bin files (nanoMoE format) and yield training batches.
    Matches nanoMoE's get_batch function exactly.
    
    Args:
        B: batch size
        T: sequence length (block_size)
        data_dir: directory containing train.bin and val.bin files
        split: "train" or "val"
        device: device to move tensors to
        resume_state_dict: optional state dict for resuming training (not used in nanoMoE, kept for compatibility)
    
    Yields:
        inputs, targets, state_dict tuples
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    
    # Get binary file path
    bin_path = os.path.join(data_dir, f'{split}.bin')
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Data file not found: {bin_path}")
    
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    
    while True:  # infinite iteration
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        
        # Sample random positions for this batch (matching nanoMoE exactly: len(data) - T)
        ix = torch.randint(len(data) - T, (B,))
        
        # Convert memmap slices directly to tensors (matching nanoMoE train.py exactly)
        x = torch.stack([torch.from_numpy((data[i:i+T]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+T]).astype(np.int64)) for i in ix])
        
        # Move to device with optional memory pinning (matching nanoMoE exactly)
        if device_type == 'cuda':
            # Try pinning memory, but fall back if it fails
            try:
                x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
            except RuntimeError:
                # Fall back to regular transfer if pin_memory fails
                x, y = x.to(device), y.to(device)
        else:
            x, y = x.to(device), y.to(device)
        
        # Return state_dict for compatibility (nanoMoE doesn't use this, but we keep it)
        state_dict = {"pos": 0}  # Simple placeholder, not used in nanoMoE
        yield x, y, state_dict

def bin_data_loader(*args, **kwargs):
    """Helper function that only emits inputs/targets without state_dict"""
    for inputs, targets, state_dict in bin_data_loader_with_state(*args, **kwargs):
        yield inputs, targets

def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This implementation became a bit more complex because we wish to support approximate resume training.
    Instead of turning this into a Class, we opt to return the state_dict with every batch,
    and then the caller can pass in a state_dict to resume training from a desired point.
    Note that this resumption is atm only *approximate* for simplicity.
    We won't repeat the same documents but we might skip a few.
    The state_dict that is returned can be later passed into this function via `resume_state_dict` to approximately resume.

    Perfect state resumption is possible but would be a lot more bloated, probably not worth it atm.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    def document_batches():
        if USE_OPENWEBTEXT:
            # Load openwebtext dataset from local cache
            # Note: openwebtext only has 'train' split, so we use it for both train and val
            cache_dir = os.environ.get("HF_DATASETS_CACHE", "/root/autodl-tmp/huggingface/datasets")
            dataset = load_dataset("openwebtext", "plain_text", cache_dir=cache_dir, split="train")
            full_dataset_size = len(dataset)
            
            # For validation, use the last 1% of the dataset
            if split == "val":
                dataset_start = int(full_dataset_size * 0.99)
                dataset_size = full_dataset_size - dataset_start
            else:
                dataset_start = 0
                # For training, use 99% of the dataset
                dataset_size = int(full_dataset_size * 0.99)
            
            resume_idx = resume_state_dict.get("rg_idx", 0) if resume_state_dict is not None else 0
            idx = resume_idx
            while True: # iterate infinitely (multi-epoch)
                while idx < dataset_size:
                    # Collect batch_size documents starting from current index with DDP stride
                    batch = []
                    batch_indices = []
                    for i in range(tokenizer_batch_size):
                        if idx >= dataset_size:
                            break
                        # DDP: each rank processes different documents
                        actual_idx = dataset_start + idx
                        if actual_idx < full_dataset_size and idx % ddp_world_size == ddp_rank:
                            text = dataset[actual_idx]["text"]
                            batch.append(text)
                            batch_indices.append(idx)
                        idx += 1
                    if batch:
                        yield batch, (0, idx)  # Use idx as state
                idx = 0  # Reset for next epoch
        else:
            # Original parquet file iteration
            parquet_paths = list_parquet_files()
            parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
            resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
            resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
            pq_idx = resume_pq_idx # we kick off parquet files at the resume index (or by default just 0)
            while True: # iterate infinitely (multi-epoch)
                while pq_idx < len(parquet_paths): # iterate over all parquet files
                    filepath = parquet_paths[pq_idx]
                    pf = pq.ParquetFile(filepath)
                    # Start from resume point if resuming on same file, otherwise from DDP rank
                    # I know this state resumption is a little bit tricky and a little bit hacky... sigh.
                    if resume_rg_idx is not None:
                        base_idx = resume_rg_idx // ddp_world_size # in units of ddp_world_size
                        base_idx += 1 # advance by 1 so that we definitely don't repeat data after resuming
                        rg_idx = base_idx * ddp_world_size + ddp_rank
                        resume_rg_idx = None # set to None as we only want to do this a single time
                    else:
                        rg_idx = ddp_rank
                    while rg_idx < pf.num_row_groups:
                        rg = pf.read_row_group(rg_idx)
                        batch = rg.column('text').to_pylist() # each batch is a parquet group, e.g. 1024 rows
                        # the tokenizer encode might want to go in even smaller batches, e.g. 128 rows
                        for i in range(0, len(batch), tokenizer_batch_size):
                            yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                        rg_idx += ddp_world_size # advance to the next row group (in DDP)
                    pq_idx += 1 # advance to the next parquet file
    batches = document_batches()

    # Now emit batches of tokens.
    needed_tokens = B * T + 1 # +1 is because we also need the target at the last token
    # get the tokenizer and the bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # CUDA supports memory pinning for asynchronous transfers between CPU and GPU
        use_cuda_optimizations = device.type == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations) # in PyTorch, long=int64
        # Create the inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        # For openwebtext, we use rg_idx as the state; for parquet files, we use both pq_idx and rg_idx
        if USE_OPENWEBTEXT:
            state_dict = {"pq_idx": 0, "rg_idx": rg_idx}  # Use rg_idx to track position in dataset
        else:
            state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx} # we need this in case we wish to approximately resume training
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
