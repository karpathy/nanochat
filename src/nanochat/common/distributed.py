"""Distributed training utilities and compute initialization."""

import logging
import os
from typing import Tuple

import torch
import torch.distributed as dist

from nanochat.common.io import print0
from nanochat.common.logging import setup_default_logging

logger = logging.getLogger(__name__)


def is_ddp_requested() -> bool:
    """True if launched by torchrun (env present), even before init."""
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))


def is_ddp_initialized() -> bool:
    """True if torch.distributed is available and the process group is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_dist_info() -> Tuple[bool, int, int, int]:
    if is_ddp_requested():
        assert all(var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def autodetect_device_type() -> str:
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type


def compute_init(device_type: str = "cuda") -> Tuple[bool, int, int, int, torch.device]:
    """Basic initialization that we keep doing over and over, so make common."""
    setup_default_logging()

    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), (
            "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
        )
    if device_type == "mps":
        assert torch.backends.mps.is_available(), (
            "Your PyTorch installation is not configured for MPS but device_type is 'mps'"
        )

    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)

    if device_type == "cuda":
        torch.set_float32_matmul_precision(
            "high"
        )  # uses tf32 instead of fp32 for matmuls

    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device


def compute_cleanup() -> None:
    """Companion function to compute_init, to clean things up before script exit."""
    if is_ddp_initialized():
        dist.destroy_process_group()
