"""
What the load-bearing code needs from its environment: the compute dtype, the rank
info, print0, and where things live on disk. Everything else about running a script
(distributed init, logging, wandb, hardware tables) is in harness/runtime.py.
"""

import os
import json
import torch

# The dtype used for compute (matmuls, activations). Master weights stay fp32 for optimizer precision.
# Linear layers cast their weights to this dtype in forward, replacing torch.amp.autocast.
# Override with NANOCHAT_DTYPE env var: "bfloat16" or "float32"
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float32": torch.float32}
def _detect_compute_dtype():
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via NANOCHAT_DTYPE={env}"
    if torch.cuda.is_available():
        # bf16 requires SM 80+ (Ampere: A100, A10, etc.)
        # Older GPUs like V100 (SM 70) and T4 (SM 75) have fp16 but no bf16 tensor cores; fp16
        # training needs loss scaling, which nanochat does not do, so they train in fp32
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
    # Note: MPS on recent macOS also handles bf16 fine, opt in via NANOCHAT_DTYPE=bfloat16
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"
COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()


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

def get_experiment_name():
    """The active experiment name, e.g. exported by run.sh."""
    return os.environ.get("NANOCHAT_EXPERIMENT", "default")

def get_experiment_dir(name=None):
    """
    The experiment dir holds everything one experiment produces: the tokenizer,
    checkpoints, stage logs, meta.json. See experiment_refactor.md for the design.
    Lives under the base dir (not the repo, to keep the source tree lean):
    $NANOCHAT_BASE_DIR/experiments/<name>, overridable via NANOCHAT_EXPERIMENTS_DIR.
    """
    name = get_experiment_name() if name is None else name
    default_root = os.path.join(get_base_dir(), "experiments")
    experiments_root = os.environ.get("NANOCHAT_EXPERIMENTS_DIR", default_root)
    experiment_dir = os.path.join(experiments_root, name)
    return experiment_dir

CANONICAL_DATASET = "climbmix"

def get_dataset_name():
    """
    The active dataset name: $NANOCHAT_DATASET if set; else the dataset recorded in the
    active experiment's meta.json, so that downstream stages (evals, chat) resolve the
    tokenizer and data the experiment was trained on; else the canonical dataset.
    """
    if "NANOCHAT_DATASET" in os.environ:
        return os.environ["NANOCHAT_DATASET"]
    meta_path = os.path.join(get_experiment_dir(), "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get("dataset", CANONICAL_DATASET)
    return CANONICAL_DATASET

def get_dataset_dir(name=None):
    """Resolve a dataset name to its directory in the shared store."""
    name = get_dataset_name() if name is None else name
    return os.path.join(get_base_dir(), "datasets", name)

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def is_ddp_requested() -> bool:
    """
    True if launched by torchrun (env present), even before init.
    Used to decide whether we *should* initialize a PG.
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def get_dist_info():
    if is_ddp_requested():
        # We rely on torchrun's env to decide if we SHOULD init.
        # (Initialization itself happens in compute init.)
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1
