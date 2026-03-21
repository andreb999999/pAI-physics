"""
Deployment mode resolution for PoggioAI/MSc.

Supports three modes:
  - local:  Laptop-only, CPU experiments, no external compute
  - tinker: Local orchestrator + Tinker API for GPU training
  - hpc:    HPC cluster with SLURM job scheduler

Usage:
    from consortium.mode import resolve_mode, load_mode_config, apply_mode_defaults
    mode = resolve_mode(args)
    mode_config = load_mode_config(mode)
    apply_mode_defaults(args, mode_config)
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

VALID_MODES = ("local", "tinker", "hpc")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODES_CONFIG_PATH = _REPO_ROOT / "config" / "modes.yaml"


def resolve_mode(args: Any) -> str:
    """Determine the deployment mode from CLI flag or environment auto-detection.

    Priority:
      1. Explicit --mode flag
      2. CONSORTIUM_MODE env var
      3. Auto-detection from environment
    """
    # 1. Explicit CLI flag
    explicit = getattr(args, "mode", None)
    if explicit:
        if explicit not in VALID_MODES:
            print(
                f"[PoggioAI] ERROR: Unknown mode '{explicit}'. "
                f"Valid modes: {', '.join(VALID_MODES)}",
                file=sys.stderr,
            )
            sys.exit(1)
        return explicit

    # 2. Env var
    env_mode = os.environ.get("CONSORTIUM_MODE", "").lower()
    if env_mode in VALID_MODES:
        return env_mode

    # 3. Auto-detect
    if os.environ.get("CONSORTIUM_SLURM_ENABLED", "0") in ("1", "true", "yes"):
        return "hpc"
    if shutil.which("sbatch") is not None:
        return "hpc"
    if os.environ.get("TINKER_API_KEY"):
        return "tinker"

    return "local"


def load_mode_config(mode: str) -> dict:
    """Load the mode profile from config/modes.yaml."""
    if not _MODES_CONFIG_PATH.exists():
        print(
            f"[PoggioAI] WARNING: {_MODES_CONFIG_PATH} not found, using built-in defaults.",
            file=sys.stderr,
        )
        return _builtin_defaults(mode)

    with open(_MODES_CONFIG_PATH) as f:
        all_modes = yaml.safe_load(f) or {}

    config = all_modes.get(mode)
    if not config:
        print(
            f"[PoggioAI] WARNING: Mode '{mode}' not defined in modes.yaml, "
            f"using built-in defaults.",
            file=sys.stderr,
        )
        return _builtin_defaults(mode)

    return config


def apply_mode_defaults(args: Any, mode_config: dict) -> None:
    """Apply mode defaults to args and environment.

    CLI flags always win — mode defaults only fill in unset values.
    """
    mode = mode_config.get("description", "unknown")

    # --- Set environment variables from mode profile ---
    for key, value in mode_config.get("env", {}).items():
        if key not in os.environ:
            os.environ[key] = str(value)

    # --- Validate required env vars ---
    missing = []
    for key in mode_config.get("required_env", []):
        if not os.environ.get(key):
            missing.append(key)
    if missing:
        print(
            f"[PoggioAI] ERROR: Mode requires these environment variables: "
            f"{', '.join(missing)}\n"
            f"  Set them in your .env file or shell environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Apply default output format (only if user didn't set it) ---
    default_fmt = mode_config.get("default_output_format")
    if default_fmt and getattr(args, "output_format", None) == "latex":
        # Only override if user didn't explicitly pass --output-format
        # argparse default is "latex", so we check if it was explicitly set
        if not _was_explicitly_set(args, "output_format"):
            args.output_format = default_fmt

    # --- Store mode in env for downstream components ---
    os.environ["CONSORTIUM_MODE"] = mode_config.get("experiment_backend", "local")

    # --- Set experiment timeout if not overridden ---
    timeout_key = "CONSORTIUM_EXPERIMENT_TIMEOUT"
    if timeout_key not in os.environ:
        timeout = mode_config.get("experiment_timeout")
        if timeout:
            os.environ[timeout_key] = str(timeout)


def _was_explicitly_set(args: Any, attr: str) -> bool:
    """Heuristic: check if an argparse attribute was explicitly passed on CLI."""
    # We inspect sys.argv for --output-format or --output_format
    flag_variants = [f"--{attr}", f"--{attr.replace('_', '-')}"]
    return any(flag in sys.argv for flag in flag_variants)


def _builtin_defaults(mode: str) -> dict:
    """Hardcoded fallback defaults when modes.yaml is missing."""
    defaults = {
        "local": {
            "description": "Laptop-only mode",
            "experiment_backend": "local",
            "experiment_device": "cpu",
            "experiment_timeout": 1800,
            "default_output_format": "markdown",
            "env": {"CONSORTIUM_SLURM_ENABLED": "0", "CUDA_VISIBLE_DEVICES": ""},
        },
        "tinker": {
            "description": "Tinker API mode",
            "experiment_backend": "tinker",
            "experiment_device": "gpu",
            "experiment_timeout": 7200,
            "default_output_format": "latex",
            "env": {"CONSORTIUM_SLURM_ENABLED": "0"},
            "required_env": ["TINKER_API_KEY"],
        },
        "hpc": {
            "description": "HPC/SLURM mode",
            "experiment_backend": "slurm",
            "experiment_device": "gpu",
            "experiment_timeout": 25200,
            "default_output_format": "latex",
            "env": {"CONSORTIUM_SLURM_ENABLED": "1"},
        },
    }
    return defaults.get(mode, defaults["local"])
