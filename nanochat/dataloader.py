from collections import deque

import torch

from nanochat.common import get_dist_info
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda"):
    """Stream pretraining text from parquet files, tokenize, yield training batches."""
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1 # +1 is because we also need the target at the last token
    # get the tokenizer and the bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left
    # pin_memory and non_blocking only work on CUDA
    device_type = device if isinstance(device, str) else device.type
    use_cuda_optimizations = device_type == "cuda"

    # infinite iterator over document batches
    def document_batches():
        while True:
            # batch will iterate in group size of the parquet files, usually e.g. 1024 rows
            for batch in parquets_iter_batched(split=split, start=ddp_rank, step=ddp_world_size):
                # for the tokenizer we might want to go in usually smaller batches, e.g. 128 rows
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size]
    batches = document_batches()

    batch_index = 0
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
            batch_index += 1
        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        scratch = torch.tensor(tokens, dtype=torch.int64, pin_memory=use_cuda_optimizations)
        # Create the inputs/targets as 1D tensors
        inputs_cpu = scratch[:-1].to(dtype=torch.int32)
        targets_cpu = scratch[1:]
        # Reshape to 2D and move to device (async on CUDA)
        inputs = inputs_cpu.view(B, T).to(device=device, dtype=torch.int32, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, dtype=torch.int64, non_blocking=use_cuda_optimizations)
        yield inputs, targets
