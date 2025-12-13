from collections import deque

import torch
import pyarrow.parquet as pq
from tqdm import tqdm

from nanochat_moe.common import get_dist_info, print0
from nanochat_moe.dataset import list_parquet_files
from nanochat_moe.tokenizer import get_tokenizer

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
    show_progress = ddp_rank == 0  # only show progress on rank 0
    
    print0(f"[DataLoader] Initializing dataloader for split={split}, rank={ddp_rank}/{ddp_world_size}")
    
    def document_batches():
        from nanochat_moe.dataset import DATA_DIR
        print0(f"[DataLoader] Listing parquet files from: {DATA_DIR}")
        parquet_paths = list_parquet_files()
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        print0(f"[DataLoader] Found {len(parquet_paths)} parquet files for {split} split")
        if len(parquet_paths) == 0:
            print0(f"[DataLoader] WARNING: No parquet files found! Check if data directory exists and contains .parquet files.")
        
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        pq_idx = resume_pq_idx # we kick off parquet files at the resume index (or by default just 0)
        pbar = None
        epoch = 0
        while True: # iterate infinitely (multi-epoch)
            if show_progress and pbar is None:
                # Use position=0 and leave=True to ensure progress bar displays correctly
                pbar = tqdm(total=len(parquet_paths), desc=f"Tokenizing {split} data (epoch {epoch})", unit="file", leave=True, position=0, file=None)
            while pq_idx < len(parquet_paths): # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                if show_progress:
                    pbar.set_postfix({"file": f"{pq_idx+1}/{len(parquet_paths)}"})
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
                if show_progress:
                    pbar.update(1)
                pq_idx += 1 # advance to the next parquet file
            # Finished one epoch, reset for next epoch
            if show_progress:
                pbar.close()
                pbar = None
            epoch += 1
            pq_idx = 0  # reset to start of files for next epoch
    batches = document_batches()

    # Now emit batches of tokens.
    needed_tokens = B * T + 1 # +1 is because we also need the target at the last token
    # get the tokenizer and the bos token
    print0(f"[DataLoader] Loading tokenizer...")
    tokenizer = get_tokenizer()
    print0(f"[DataLoader] Tokenizer loaded, vocab_size={tokenizer.get_vocab_size()}")
    bos_token = tokenizer.get_bos_token_id()
    print0(f"[DataLoader] Starting to yield batches (needed_tokens={needed_tokens})...")
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
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations) # in PyTorch, long=int64
        # Create the inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to GPU async
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx} # we need this in case we wish to approximately resume training
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
