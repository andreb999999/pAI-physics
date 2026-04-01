"""Detect platform capabilities: OS, Python, conda/venv, SLURM, LaTeX."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass, field


@dataclass
class PlatformInfo:
    os_name: str = ""            # linux, darwin, windows
    os_version: str = ""
    arch: str = ""               # x86_64, arm64
    python_version: str = ""
    python_path: str = ""
    has_conda: bool = False
    conda_path: str = ""
    active_conda_env: str = ""
    has_venv: bool = False       # currently in a venv
    has_slurm: bool = False
    has_pdflatex: bool = False
    has_git: bool = False
    has_playwright: bool = False
    wsl: bool = False
    missing: list[str] = field(default_factory=list)


def detect() -> PlatformInfo:
    """Detect the current platform and available tools."""
    info = PlatformInfo()

    # OS
    info.os_name = platform.system().lower()
    info.os_version = platform.version()
    info.arch = platform.machine()

    # WSL detection
    if info.os_name == "linux":
        try:
            with open("/proc/version", "r") as f:
                if "microsoft" in f.read().lower():
                    info.wsl = True
        except OSError:
            pass

    # Python
    info.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    info.python_path = sys.executable

    if sys.version_info < (3, 10):
        info.missing.append("Python >= 3.10 required")

    # Conda
    info.conda_path = shutil.which("conda") or ""
    info.has_conda = bool(info.conda_path)
    info.active_conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")

    # Venv
    info.has_venv = sys.prefix != sys.base_prefix

    # SLURM
    info.has_slurm = bool(shutil.which("sbatch"))

    # LaTeX
    info.has_pdflatex = bool(shutil.which("pdflatex"))

    # Git
    info.has_git = bool(shutil.which("git"))

    # Playwright (check importability)
    try:
        import playwright  # noqa: F401
        info.has_playwright = True
    except ImportError:
        info.has_playwright = False

    return info


def detect_consortium() -> tuple[bool, str]:
    """Check if consortium is importable and return its location."""
    try:
        import consortium  # noqa: F401
        path = getattr(consortium, "__file__", "unknown")
        return True, os.path.dirname(path) if path != "unknown" else "unknown"
    except ImportError:
        return False, ""
