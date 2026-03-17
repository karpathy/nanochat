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
from nanochat.common.hardware import get_device_sync, get_peak_flops
from nanochat.common.io import download_file_with_lock, download_single_file, print0, print_banner
from nanochat.common.logging import ColoredFormatter, setup_default_logging
from nanochat.common.paths import (
    checkpoint_dir,
    data_dir,
    eval_results_dir,
    eval_tasks_dir,
    get_default_base_dir,
    identity_data_path,
    legacy_data_dir,
    report_dir,
    root_data_dir,
    tokenizer_dir,
)
from nanochat.common.wandb import DummyWandb, LocalWandb, init_wandb

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
    "download_file_with_lock",
    "download_single_file",
    "print0",
    "print_banner",
    # hardware
    "get_device_sync",
    "get_peak_flops",
    # wandb
    "DummyWandb",
    "LocalWandb",
    "init_wandb",
    # paths
    "get_default_base_dir",
    "root_data_dir",
    "data_dir",
    "legacy_data_dir",
    "eval_tasks_dir",
    "eval_results_dir",
    "tokenizer_dir",
    "checkpoint_dir",
    "identity_data_path",
    "report_dir",
]
