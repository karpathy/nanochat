"""Shared helpers for locating and building the Swift NanochatMLXStub package.

Both ``nanochat/swift_stub_engine.py`` and ``scripts/mlx_swift_stub.py`` use
these utilities so that the freshness logic and build command stay in one place.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def package_dir(root: Path) -> Path:
    return root / "swift" / "NanochatMLXStub"


def build_products_dir(root: Path) -> Path:
    return root / "swift" / "Build" / "Products" / "Debug"


def stub_binary_path(root: Path) -> Path:
    return build_products_dir(root) / "nanochat-mlx-stub"


def bundle_path(root: Path) -> Path:
    return build_products_dir(root) / "mlx-swift_Cmlx.bundle"


# ---------------------------------------------------------------------------
# Build freshness helpers
# ---------------------------------------------------------------------------

def _stub_build_inputs(root: Path) -> list[Path]:
    package_root = package_dir(root)
    inputs = [package_root / "Package.swift", package_root / "Package.resolved"]
    sources_dir = package_root / "Sources"
    if sources_dir.exists():
        inputs.extend(path for path in sources_dir.rglob("*.swift") if path.is_file())
    return [path for path in inputs if path.exists()]


def _stub_build_is_fresh(root: Path) -> bool:
    binary = stub_binary_path(root)
    bundle = bundle_path(root)
    if not binary.exists() or not bundle.exists():
        return False

    inputs = _stub_build_inputs(root)
    if not inputs:
        return False

    newest_input_mtime = max(path.stat().st_mtime for path in inputs)
    oldest_output_mtime = min(binary.stat().st_mtime, bundle.stat().st_mtime)
    return oldest_output_mtime >= newest_input_mtime


# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------

def ensure_stub_is_built(root: Path, *, rebuild: bool) -> None:
    """Build the Swift stub unless the build is already fresh.

    Forces a ``clean build`` when *rebuild* is True or when any Swift source
    file / Package manifest is newer than the compiled binary or framework
    bundle.
    """
    if not rebuild and _stub_build_is_fresh(root):
        return

    command = [
        "xcodebuild",
        "-scheme",
        "NanochatMLXStub",
        "-destination",
        "platform=macOS",
        "-derivedDataPath",
        ".derived",
        "clean",
        "build",
    ]
    subprocess.run(command, cwd=package_dir(root), check=True)
