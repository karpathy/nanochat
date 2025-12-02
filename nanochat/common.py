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
    if not nanochat_dir.startswith("gs://"):
        os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
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
            content = response.read() # bytes

        # Write to local file
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
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

def autodetect_device_type():
    # prefer to use CUDA if available, otherwise use MPS, otherwise fallback on CPU
    print0(f"DEBUG: torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print0(f"DEBUG: torch.version.cuda: {torch.version.cuda}")
        print0(f"DEBUG: torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")
        print0(f"DEBUG: torch.cuda.device_count(): {torch.cuda.device_count()}")
        print0(f"DEBUG: torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    
    # Print environment variables relevant to CUDA
    env_vars = ["LD_LIBRARY_PATH", "PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES"]
    for var in env_vars:
        print0(f"DEBUG: env {var}: {os.environ.get(var, 'NOT SET')}")

    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type

def compute_init(device_type="cuda"): # cuda|cpu|mps
    """Basic initialization that we keep doing over and over, so make common."""

    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"

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
        torch.set_float32_matmul_precision("high") # uses tf32 instead of fp32 for matmuls

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type) # mps|cpu

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
    def init(self, *args, **kwargs):
        return self
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass

class VertexLogger:
    """Logs metrics to Vertex AI Experiments."""
    def __init__(self, experiment_name, tensorboard_resource_name=None):
        from google.cloud import aiplatform
        self.aiplatform = aiplatform
        self.experiment_name = experiment_name
        self.tensorboard_resource_name = tensorboard_resource_name
        self._run = None

    def init(self, project=None, name=None, config=None, **kwargs):
        # Map wandb 'project' to Vertex 'experiment'
        experiment = project or self.experiment_name
        
        self.aiplatform.init(
            experiment=experiment, 
            experiment_tensorboard=self.tensorboard_resource_name
        )
        try:
            self._run = self.aiplatform.start_run(run=name, resume=True)
        except Exception as e:
            print(f"Could not resume run {name}: {e}. Creating a new run.")
            self._run = self.aiplatform.start_run(run=name, resume=False)
        
        # Initialize TensorBoard SummaryWriter if tensorboard resource is provided
        # We need to write to a GCS bucket that the TensorBoard resource can access.
        # Vertex AI automatically uploads logs if we write to the base_output_directory?
        # Or we can write directly to GCS if we have permissions.
        # Let's try writing to a GCS path derived from the bucket we use for data.
        # Ideally we should pass the bucket name, but let's infer or use a default.
        # Actually, for Custom Jobs, 'base_output_directory' is often set.
        # Let's try to use the GCS bucket passed in args if possible, but here we don't have it easily.
        # However, we can use the 'gs://nzp-nanochat/tensorboard_logs/{name}' path.
        # We'll assume the bucket 'nzp-nanochat' exists as it's hardcoded elsewhere.
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            import os
            # Use AIP_TENSORBOARD_LOG_DIR if available (set by Vertex AI)
            log_dir = os.environ.get('AIP_TENSORBOARD_LOG_DIR')
            if not log_dir:
                # Fallback for local runs or if env var is missing
                log_dir = f"gs://nzp-nanochat/tensorboard_logs/{name}"
                print(f"AIP_TENSORBOARD_LOG_DIR not found. Using fallback: {log_dir}")
            
            self.summary_writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logging enabled to: {log_dir}")
        except Exception as e:
            print(f"Failed to initialize TensorBoard SummaryWriter: {e}")
            self.summary_writer = None

        if config:
            self.aiplatform.log_params(config)
        return self

    def log(self, data, step=None):
        # Only log from rank 0 to avoid concurrency conflicts with Vertex AI Experiments
        import os
        rank = int(os.environ.get('RANK', 0))
        
        # Vertex AI log_metrics doesn't support 'step' directly in the same way.
        # It logs a new data point.
        # We must flatten the dictionary because log_metrics only accepts scalars.
        
        def flatten(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flat_data = flatten(data)
        
        #Extract step for TensorBoard if present in the data
        global_step = flat_data.get('step', step if step is not None else 0)
        
        # Only rank 0 should log to Vertex AI Experiments to prevent etag conflicts
        if rank == 0:
            self.aiplatform.log_metrics(flat_data)
        
        # Log to TensorBoard from all ranks (TensorBoard can handle concurrent writes)
        if self.summary_writer:
            for k, v in flat_data.items():
                if isinstance(v, (int, float)) and k != 'step':  # Don't log 'step' as a metric
                    self.summary_writer.add_scalar(k, v, global_step=global_step)
            self.summary_writer.flush()

    def finish(self):
        if self.summary_writer:
            self.summary_writer.close()
        self.aiplatform.end_run()

def get_experiment_logger(args):
    """Returns a logger compatible with wandb interface."""
    if hasattr(args, 'wandb_run') and args.wandb_run != "dummy":
        import wandb
        return wandb
    elif hasattr(args, 'vertex_experiment') and args.vertex_experiment:
        return VertexLogger(
            experiment_name=args.vertex_experiment,
            tensorboard_resource_name=getattr(args, 'vertex_tensorboard', None)
        )
    else:
        return DummyWandb()
