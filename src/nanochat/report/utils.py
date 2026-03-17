"""
Utilities for generating training report cards. More messy code than usual, will fix.
"""

import datetime
import os
import platform
import socket
import subprocess
from typing import cast

import psutil
import torch


def run_command(cmd: str) -> str | None:
    """Run a shell command and return output, or None if it fails."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        # Return stdout if we got output (even if some files in xargs failed)
        if result.stdout.strip():
            return result.stdout.strip()
        if result.returncode == 0:
            return ""
        return None
    except Exception:
        return None


def get_git_info():
    """Get current git commit, branch, and dirty status."""
    info = {}
    info["commit"] = run_command("git rev-parse --short HEAD") or "unknown"
    info["branch"] = run_command("git rev-parse --abbrev-ref HEAD") or "unknown"

    # Check if repo is dirty (has uncommitted changes)
    status = run_command("git status --porcelain")
    info["dirty"] = bool(status) if status is not None else False

    # Get commit message
    info["message"] = run_command("git log -1 --pretty=%B") or ""
    info["message"] = info["message"].split("\n")[0][:80]  # First line, truncated

    return info


def get_gpu_info():
    """Get GPU information."""
    if not torch.cuda.is_available():
        return {"available": False}

    num_devices = torch.cuda.device_count()
    info = {"available": True, "count": num_devices, "names": [], "memory_gb": []}

    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        info["names"].append(props.name)
        info["memory_gb"].append(props.total_memory / (1024**3))

    # Get CUDA version
    info["cuda_version"] = torch.version.cuda or "unknown"

    return info


def get_system_info():
    """Get system information."""
    info = {}

    # Basic system info
    info["hostname"] = socket.gethostname()
    info["platform"] = platform.system()
    info["python_version"] = platform.python_version()
    info["torch_version"] = torch.__version__

    # CPU and memory
    info["cpu_count"] = psutil.cpu_count(logical=False)
    info["cpu_count_logical"] = psutil.cpu_count(logical=True)
    info["memory_gb"] = psutil.virtual_memory().total / (1024**3)

    # User and environment
    info["user"] = os.environ.get("USER", "unknown")
    info["nanochat_base_dir"] = os.environ.get("NANOCHAT_BASE_DIR", "out")
    info["working_dir"] = os.getcwd()

    return info


def estimate_cost(gpu_info: dict[str, object], runtime_hours: float | None = None) -> dict[str, object] | None:
    """Estimate training cost based on GPU type and runtime."""

    # Rough pricing, from Lambda Cloud
    default_rate = 2.0
    gpu_hourly_rates = {
        "H100": 3.00,
        "A100": 1.79,
        "V100": 0.55,
    }

    if not gpu_info.get("available"):
        return None

    # Try to identify GPU type from name
    hourly_rate = None
    gpu_name = cast(list[object], gpu_info["names"])[0] if gpu_info["names"] else "unknown"
    for gpu_type, rate in gpu_hourly_rates.items():
        if gpu_type in gpu_name:
            hourly_rate = rate * cast(int, gpu_info["count"])
            break

    if hourly_rate is None:
        hourly_rate = default_rate * cast(int, gpu_info["count"])  # Default estimate

    return {
        "hourly_rate": hourly_rate,
        "gpu_type": gpu_name,
        "estimated_total": hourly_rate * runtime_hours if runtime_hours else None,
    }


def generate_header():
    """Generate the header for a training report."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    git_info = get_git_info()
    gpu_info = get_gpu_info()
    sys_info = get_system_info()
    cost_info = estimate_cost(gpu_info)

    header = f"""# nanochat training report

Generated: {timestamp}

## Environment

### Git Information
- Branch: {git_info["branch"]}
- Commit: {git_info["commit"]} {"(dirty)" if git_info["dirty"] else "(clean)"}
- Message: {git_info["message"]}

### Hardware
- Platform: {sys_info["platform"]}
- CPUs: {sys_info["cpu_count"]} cores ({sys_info["cpu_count_logical"]} logical)
- Memory: {sys_info["memory_gb"]:.1f} GB
"""

    if gpu_info.get("available"):
        gpu_names = ", ".join(set(gpu_info["names"]))
        total_vram = sum(gpu_info["memory_gb"])
        header += f"""- GPUs: {gpu_info["count"]}x {gpu_names}
- GPU Memory: {total_vram:.1f} GB total
- CUDA Version: {gpu_info["cuda_version"]}
"""
    else:
        header += "- GPUs: None available\n"

    if cost_info and cost_info["hourly_rate"] > 0:
        header += f"""- Hourly Rate: ${cost_info["hourly_rate"]:.2f}/hour\n"""

    header += f"""
### Software
- Python: {sys_info["python_version"]}
- PyTorch: {sys_info["torch_version"]}

"""

    # bloat metrics: count lines/chars in git-tracked source files only
    extensions = ["py", "md", "rs", "html", "toml", "sh"]
    git_patterns = " ".join(f"'*.{ext}'" for ext in extensions)
    files_output = run_command(f"git ls-files -- {git_patterns}")
    file_list = [f for f in (files_output or "").split("\n") if f]
    num_files = len(file_list)
    num_lines = 0
    num_chars = 0
    if num_files > 0:
        wc_output = run_command(f"git ls-files -- {git_patterns} | xargs wc -lc 2>/dev/null")
        if wc_output:
            total_line = wc_output.strip().split("\n")[-1]
            parts = total_line.split()
            if len(parts) >= 2:
                num_lines = int(parts[0])
                num_chars = int(parts[1])
    num_tokens = num_chars // 4  # assume approximately 4 chars per token

    # count dependencies via uv.lock
    uv_lock_lines = 0
    if os.path.exists("uv.lock"):
        with open("uv.lock", "r", encoding="utf-8") as f:
            uv_lock_lines = len(f.readlines())

    header += f"""
### Bloat
- Characters: {num_chars:,}
- Lines: {num_lines:,}
- Files: {num_files:,}
- Tokens (approx): {num_tokens:,}
- Dependencies (uv.lock lines): {uv_lock_lines:,}

"""
    return header


# -----------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Slugify a text string."""
    return text.lower().replace(" ", "-")


# the expected files and their order
EXPECTED_FILES = [
    "tokenizer-training.md",
    "tokenizer-evaluation.md",
    "base-model-training.md",
    "base-model-loss.md",
    "base-model-evaluation.md",
    "chat-sft.md",
    "chat-evaluation-sft.md",
    "chat-rl.md",
    "chat-evaluation-rl.md",
]
# the metrics we're currently interested in
chat_metrics = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "ChatCORE"]


def extract(section: str, keys: str | list[str]) -> dict[str, str]:
    """simple def to extract a single key from a section"""
    if not isinstance(keys, list):
        keys = [keys]  # convenience
    out = {}
    for line in section.split("\n"):
        for key in keys:
            if key in line:
                out[key] = line.split(":")[1].strip()
    return out


def extract_timestamp(content: str, prefix: str) -> datetime.datetime | None:
    """Extract timestamp from content with given prefix."""
    for line in content.split("\n"):
        if line.startswith(prefix):
            time_str = line.split(":", 1)[1].strip()
            try:
                return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
    return None
