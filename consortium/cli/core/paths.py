"""Shared path utilities for the msc CLI."""

from __future__ import annotations

from pathlib import Path


def find_results_dir() -> Path | None:
    """Find the results directory by checking common locations."""
    candidates = [
        Path.cwd() / "results",
        Path.cwd().parent / "results",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None
