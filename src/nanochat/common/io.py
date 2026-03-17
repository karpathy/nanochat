"""File I/O utilities, base directory resolution, and console output."""

import os
import time
import urllib.request
from typing import Callable, Optional

import requests
from filelock import FileLock

from nanochat.common.paths import root_data_dir


def download_single_file(data_dir: str, url: str, filename: str, max_attempts: int = 5) -> bool:
    """Downloads a single file index, with some backoff"""

    # Construct the local filepath for this file and skip if it already exists
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True
    print(f"Downloading {filename}...")

    # Download with retries
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2**attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


def download_file_with_lock(
    base_dir: str, url: str, filename: str, postprocess_fn: Optional[Callable[[str], None]] = None
) -> str:
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    file_path = os.path.join(root_data_dir(base_dir), filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        if os.path.exists(file_path):
            return file_path

        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()

        with open(file_path, "wb") as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        if postprocess_fn is not None:
            postprocess_fn(file_path)

    os.remove(lock_path)

    return file_path


def print0(s: str = "", **kwargs: object) -> None:
    ddp_rank = int(os.environ.get("RANK", 0))
    if ddp_rank == 0:
        print(s, **kwargs)


def print_banner() -> None:
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
