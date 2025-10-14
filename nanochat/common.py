"""
Common utilities for nanochat.
"""

import os
import re
import logging
import torch
import torch.distributed as dist

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        message = super().format(record)
        # Add color to specific parts of the message
        if levelname == 'INFO':
            # Highlight numbers and percentages
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    banner = """
                                                   █████                 █████
                                                  ░░███                 ░░███
 ████████    ██████   ████████    ██████   ██████  ░███████    ██████   ███████
░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███ ░░░███░
 ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████   ░███
 ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███   ░███ ███
 ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░████████  ░░█████
░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░    ░░░░░
"""
    print0(banner)

def is_ddp():
    # TODO is there a proper way
    return int(os.environ.get('RANK', -1)) != -1

def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def compute_init():
    """Basic initialization that we keep doing over and over, so make common."""

    # CUDA is currently required
    assert torch.cuda.is_available(), "CUDA is needed for a distributed run atm"

    # Reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Precision
    torch.set_float32_matmul_precision("high") # uses tf32 instead of fp32 for matmuls

    # Distributed setup: Distributed Data Parallel (DDP), optional
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp:
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device) # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device("cuda")

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    if is_ddp():
        dist.destroy_process_group()

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass
