"""
Common utilities for nanochat.
"""

import os
import re
import logging
import urllib.request
import torch
import torch.distributed as dist
from filelock import FileLock

# The dtype used for compute (matmuls, activations). Master weights stay fp32 for optimizer precision.
# Linear layers cast their weights to this dtype in forward, replacing torch.amp.autocast.
# Override with NANOCHAT_DTYPE env var: "bfloat16", "float16", "float32"
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
def _detect_compute_dtype():
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via NANOCHAT_DTYPE={env}"
    if torch.cuda.is_available():
        # bf16 requires SM 80+ (Ampere: A100, A10, etc.)
        # Older GPUs like V100 (SM 70) and T4 (SM 75) only have fp16 tensor cores
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        # fp16 training requires GradScaler (not yet implemented), so fall back to fp32.
        # Users can still force fp16 via NANOCHAT_DTYPE=float16 if they know what they're doing.
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"
COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        message = super().format(record)
        if levelname == 'INFO':
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
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir


def get_checkpoint_base_dir():
    """Base directory for checkpoints (base_checkpoints/, chatsft_checkpoints/, etc.). Defaults to get_base_dir(); override with NANOBOT_BASE_DIR or NANOCHAT_BASE_DIR."""
    env_dir = os.environ.get("NANOBOT_BASE_DIR") or os.environ.get("NANOCHAT_BASE_DIR")
    return os.path.expanduser(env_dir) if env_dir else get_base_dir()


def download_file_with_lock(url: str, filename: str, postprocess_fn=None) -> str:
    base_dir = get_base_dir()
    path = os.path.join(base_dir, filename)
    lock_path = path + ".lock"
    with FileLock(lock_path):
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)
        if postprocess_fn is not None:
            postprocess_fn(path)
    return path


def get_dist_info():
    if dist.is_initialized():
        return True, dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0)), dist.get_world_size()
    return False, 0, 0, 1

def print0(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)


def print_banner():
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    print("=" * 60)
    print("nanochat base_train")
    print("=" * 60)

def autodetect_device_type():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_peak_flops(name: str) -> float:
    flops_map = {
        "H100": 1.98e15,
        "A100": 3.12e14,
        "A10": 1.25e14,
        "RTX 4090": 8.2e13,
    }
    for key, flops in flops_map.items():
        if key in name:
            return flops
    return 1e14

def is_ddp_initialized():
    return dist.is_initialized()

def compute_init(device_type: str):
    if device_type == "cuda" and torch.cuda.is_available():
        if dist.is_initialized():
            rank = dist.get_rank()
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
            device = torch.device("cuda", local_rank)
            return True, rank, local_rank, dist.get_world_size(), device
        return False, 0, 0, 1, torch.device("cuda", 0)
    if device_type == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return False, 0, 0, 1, torch.device("mps")
    return False, 0, 0, 1, torch.device("cpu")

def estimate_model_vram(n_layer, n_embd, n_head, vocab_size, seq_len, batch_size, dtype=torch.float32, training=True):
    """
    Estimate VRAM usage (in GB) for a given GPT configuration.
    This is a rough heuristic that includes weights, gradients, optimizer states, and activations.
    """
    # Bytes per element
    bpe = 2 if dtype in [torch.float16, torch.bfloat16] else 4

    # 1. Parameter memory (weights)
    # roughly 12 * n_layer * n_embd^2
    n_params = 12 * n_layer * n_embd**2 + vocab_size * n_embd
    param_mem = n_params * bpe

    if not training:
        # For inference, we only need weights and KV cache
        # KV cache: 2 * n_layer * batch_size * seq_len * n_embd * bpe
        kv_cache_mem = 2 * n_layer * batch_size * seq_len * n_embd * bpe
        return (param_mem + kv_cache_mem) / 1024**3

    # 2. Gradient memory (same as parameters)
    grad_mem = n_params * bpe

    # 3. Optimizer memory (AdamW stores 2 states per parameter in FP32)
    opt_mem = n_params * 2 * 4

    # 4. Activation memory (Roughly batch * seq * n_layer * n_embd * bytes)
    # This varies a lot with implementation (checkpointing, flash attention)
    act_mem = batch_size * seq_len * n_layer * n_embd * bpe * 4

    total_bytes = param_mem + grad_mem + opt_mem + act_mem
    return total_bytes / 1024**3

def recommend_config(vram_gb, training=True, device_type="cuda"):
    """Recommend model hyperparameters that should fit in the given VRAM."""
    # Start with some base configs and scale down until they fit
    # Note: n_embd should be divisible by n_head (default 6 or 8)
    configs = [
        {"depth": 20, "n_embd": 1280}, # ~1.5B (1280 % 8 == 0)
        {"depth": 16, "n_embd": 1024}, # ~500M (1024 % 8 == 0)
        {"depth": 12, "n_embd": 768},  # ~125M (768 % 8 == 0)
        {"depth": 8, "n_embd": 512},   # ~40M (512 % 8 == 0)
        {"depth": 4, "n_embd": 256},   # ~10M (256 % 8 == 0)
    ]

    for cfg in configs:
        # assume batch_size=1, seq_len=512 for estimation
        est = estimate_model_vram(cfg["depth"], cfg["n_embd"], 8, 32768, 512, 1,
                                  dtype=torch.float16 if device_type=="cuda" else torch.float32,
                                  training=training)
        if est < vram_gb * 0.8: # leave 20% headroom
            return cfg

    return configs[-1] # fallback to smallest

def compute_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

class DummyWandb:
    def log(self, d): pass
    def finish(self): pass
