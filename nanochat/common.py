"""Common utilities for nanochat."""

from __future__ import annotations

import logging
import os
import re
import urllib.request
from typing import Any
from typing import Callable

import torch
import torch.distributed as dist
from filelock import FileLock


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages.

    Attributes:
        COLORS: A dictionary mapping log level names to ANSI color codes.
        RESET: ANSI code to reset text formatting.
        BOLD: ANSI code for bold text.
    """

    COLORS: dict[str, str] = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET: str = "\033[0m"
    BOLD: str = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record with colored output.

        Args:
            record: The log record to format.

        Returns:
            The formatted log message string with ANSI color codes.
        """
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        message = super().format(record)
        if levelname == "INFO":
            message = re.sub(
                r"(\d+\.?\d*\s*(?:GB|MB|%|docs))",
                rf"{self.BOLD}\1{self.RESET}",
                message,
            )
            message = re.sub(
                r"(Shard \d+)",
                rf"{self.COLORS['INFO']}{self.BOLD}\1{self.RESET}",
                message,
            )
        return message


def setup_default_logging() -> None:
    """Sets up the default logging configuration with colored output."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])


setup_default_logging()
logger = logging.getLogger(__name__)


def get_base_dir() -> str:
    """Gets the base directory for nanochat cached data.

    Co-locates nanochat intermediates with other cached data in ~/.cache
    by default. Can be overridden by setting the NANOCHAT_BASE_DIR
    environment variable.

    Returns:
        The path to the nanochat base directory.
    """
    env_dir = os.environ.get("NANOCHAT_BASE_DIR")
    if env_dir:
        nanochat_dir = env_dir
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir


def download_file_with_lock(
    url: str,
    filename: str,
    postprocess_fn: Callable[[str], None] | None = None,
) -> str:
    """Downloads a file from a URL to a local path in the base directory.

    Uses a lock file to prevent concurrent downloads among multiple ranks.

    Args:
        url: The URL to download the file from.
        filename: The name of the file to save locally.
        postprocess_fn: Optional function to run after downloading the file.
            Receives the file path as its argument.

    Returns:
        The path to the downloaded file.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # All other ranks block until it is released

        # Recheck after acquiring lock
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()  # bytes

        # Write to local file
        with open(file_path, "wb") as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path


def print0(s: str = "", **kwargs: Any) -> None:
    """Prints a message only on rank 0 in distributed training.

    Args:
        s: The string to print.
        **kwargs: Additional keyword arguments to pass to the print function.
    """
    ddp_rank = int(os.environ.get("RANK", 0))
    if ddp_rank == 0:
        print(s, **kwargs)


def print_banner() -> None:
    """Prints the nanochat ASCII art banner on rank 0."""
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
                                                      █████                █████
                                                     ░░███                ░░███
    ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
   ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
    ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
    ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
    ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
   ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)


def is_ddp_requested() -> bool:
    """Checks if distributed training was requested via torchrun.

    Returns:
        True if launched by torchrun (environment variables present),
        even before init. Used to decide whether to initialize a
        process group.
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))


def is_ddp_initialized() -> bool:
    """Checks if torch.distributed is initialized.

    Returns:
        True if torch.distributed is available and the process group
        is initialized. Used at cleanup to avoid destroying a
        non-existent process group.
    """
    return dist.is_available() and dist.is_initialized()


def get_dist_info() -> tuple[bool, int, int, int]:
    """Gets distributed training information from environment variables.

    Returns:
        A tuple containing:
            - is_ddp: Whether distributed training is requested.
            - ddp_rank: The global rank of the current process.
            - ddp_local_rank: The local rank of the current process.
            - ddp_world_size: The total number of processes.
    """
    if is_ddp_requested():
        # We rely on torchrun's env to decide if we SHOULD init.
        # (Initialization itself happens in compute init.)
        assert all(var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def autodetect_device_type() -> str:
    """Auto-detects the best available device type.

    Prefers CUDA if available, otherwise uses MPS, otherwise falls back
    to CPU.

    Returns:
        The device type string: "cuda", "mps", or "cpu".
    """
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type


def compute_init(device_type: str = "cuda") -> tuple[bool, int, int, int, torch.device]:
    """Initializes the compute environment for training.

    Performs basic initialization including setting random seeds, precision
    settings, and distributed training setup if applicable.

    Args:
        device_type: The type of device to use. Must be one of
            "cuda", "mps", or "cpu".

    Returns:
        A tuple containing:
            - is_ddp_requested: Whether distributed training was requested.
            - ddp_rank: The global rank of the current process.
            - ddp_local_rank: The local rank of the current process.
            - ddp_world_size: The total number of processes.
            - device: The torch device to use for computation.

    Raises:
        AssertionError: If the device type is invalid or the corresponding
            backend is not available.
    """
    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), (
            "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
        )
    if device_type == "mps":
        assert torch.backends.mps.is_available(), (
            "Your PyTorch installation is not configured for MPS but device_type is 'mps'"
        )

    # Reproducibility
    # Note that we set the global seeds here, but most of the code uses explicit rng objects.
    # The only place where global rng might be used is nn.Module initialization of the model weights.
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)

    # Precision
    if device_type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "tf32"  # uses tf32 instead of fp32 for matmuls

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    is_ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)  # mps|cpu

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return is_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device


def compute_cleanup() -> None:
    """Cleans up compute resources before script exit.

    Companion function to compute_init. Destroys the distributed process
    group if it was initialized.
    """
    if is_ddp_initialized():
        dist.destroy_process_group()


class DummyWandb:
    """A dummy Weights & Biases client for when wandb is not used.

    Provides the same method signatures as the real wandb client but
    performs no operations. Useful for maintaining consistent code
    structure without requiring wandb.
    """

    def __init__(self) -> None:
        """Initializes a dummy wandb instance."""
        pass

    def log(self, *args: Any, **kwargs: Any) -> None:
        """Logs metrics (no-op).

        Args:
            *args: Positional arguments (ignored).
            **kwargs: Keyword arguments (ignored).
        """
        pass

    def finish(self) -> None:
        """Finishes the wandb run (no-op)."""
        pass


def get_peak_flops(device_name: str) -> float:
    """Gets the hardcoded BF16 peak flops for various GPUs.

    Inspired by torchtitan:
    https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
    and PR: https://github.com/karpathy/nanochat/pull/147

    Args:
        device_name: The name of the GPU device.

    Returns:
        The peak BF16 flops for the device. Returns float('inf') for
        unknown devices so MFU shows as 0% rather than a wrong guess.
    """
    name = device_name.lower()

    # --- NVIDIA Blackwell ---
    if "gb200" in name or "grace blackwell" in name:
        return 2.5e15
    if "b200" in name:
        return 2.25e15
    if "b100" in name:
        return 1.8e15

    # --- NVIDIA Hopper (H100/H200/H800) ---
    if "h200" in name:
        if "nvl" in name or "pcie" in name:
            return 836e12
        return 989e12  # H200 SXM
    if "h100" in name:
        if "nvl" in name:
            return 835e12
        if "pcie" in name:
            return 756e12
        return 989e12  # H100 SXM
    if "h800" in name:
        if "nvl" in name:
            return 989e12
        return 756e12  # H800 PCIe

    # --- NVIDIA Ampere data center ---
    if "a100" in name or "a800" in name:
        return 312e12
    if "a40" in name:
        return 149.7e12
    if "a30" in name:
        return 165e12

    # --- NVIDIA Ada data center ---
    if "l40s" in name or "l40-s" in name or "l40 s" in name:
        return 362e12
    if "l4" in name:
        return 121e12

    # --- AMD CDNA accelerators ---
    if "mi355" in name:
        return 2.5e15
    if "mi325" in name or "mi300x" in name:
        return 1.3074e15
    if "mi300a" in name:
        return 980.6e12
    if "mi250x" in name:
        return 383e12
    if "mi250" in name:
        return 362.1e12

    # --- Intel ---
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # --- Consumer RTX (for hobbyists) ---
    if "5090" in name:
        return 209.5e12
    if "4090" in name:
        return 165.2e12
    if "3090" in name:
        return 71e12

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float("inf")
