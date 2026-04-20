import glob
from pathlib import Path
import numpy as np
import torch
from nanochat.common import get_dist_info

def load_pg_shard(file: str | Path) -> torch.Tensor:
    """
    Load a Parameter Golf token shard.

    Format from parameter-golf/train_gpt.py:
    - first 256 int32 values are header
    - token payload starts after that
    - token payload dtype is little-endian uint16
    """
    file = Path(file)
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256:
        raise ValueError(f"Invalid PG shard header in {file}")
    header_bytes = 256 * np.dtype("<i4").itemsize
    num_tokens = int((file.stat().st_size - header_bytes) // np.dtype("<u2").itemsize)
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.int64))

def list_pg_bin_files(data_path: str, split: str):
    assert split in ["train", "val"]
    pattern = str(Path(data_path) / f"fineweb_{split}_*.bin")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No PG shard files found for pattern: {pattern}")
    return files

def pg_token_batch_loader_with_state(*args, **kwargs):
    for x, y in pg_token_batch_loader(*args, **kwargs):
        yield x, y, None

def pg_token_batch_loader(
    data_path: str,
    B: int,
    T: int,
    split: str,
    device: str = "cuda",
):
    """
    Minimal PG token loader for smoke testing.
    Produces contiguous fixed-length (B, T) input/target batches.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    files = list_pg_bin_files(data_path, split)
    while True:
        for f in files:
            tokens = load_pg_shard(f)
            stride = B * T * ddp_world_size
            local_stride = B * T
            start_offset = ddp_rank * local_stride
            max_start = len(tokens) - (B * T + 1)
            if max_start <= start_offset:
                continue
            pos = start_offset
            while pos <= max_start:
                chunk = tokens[pos : pos + (B * T + 1)]
                x = chunk[:-1].view(B, T)
                y = chunk[1:].view(B, T)
                if device != "cpu":
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                yield x, y
                pos += stride
