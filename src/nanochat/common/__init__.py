"""Common utilities for nanochat."""

from nanochat.common.distributed import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_dist_info,
    is_ddp_initialized,
    is_ddp_requested,
)
from nanochat.common.dtype import get_compute_dtype, get_compute_dtype_reason
from nanochat.common.hardware import get_peak_flops
from nanochat.common.io import download_file_with_lock, get_base_dir, print0, print_banner
from nanochat.common.logging import ColoredFormatter, setup_default_logging
from nanochat.common.wandb import DummyWandb

__all__ = [
    # dtype
    "get_compute_dtype",
    "get_compute_dtype_reason",
    # logging
    "ColoredFormatter",
    "setup_default_logging",
    # distributed
    "is_ddp_requested",
    "is_ddp_initialized",
    "get_dist_info",
    "autodetect_device_type",
    "compute_init",
    "compute_cleanup",
    # io
    "get_base_dir",
    "download_file_with_lock",
    "print0",
    "print_banner",
    # hardware
    "get_peak_flops",
    # wandb
    "DummyWandb",
]
