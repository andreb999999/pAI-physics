"""Shared path utilities for the msc CLI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable


def _unique_paths(paths: Iterable[Path | None]) -> list[Path]:
    """Deduplicate paths while preserving order."""
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        if path is None:
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(path)
    return ordered


def find_project_root(start: Path | None = None) -> Path | None:
    """Locate the source checkout that backs the current CLI install."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "consortium").is_dir():
            return candidate
        if (candidate / "scripts" / "campaign_cli.py").is_file() and (candidate / "consortium").is_dir():
            return candidate

    try:
        import consortium

        package_root = Path(consortium.__file__).resolve().parent.parent
    except (ImportError, AttributeError, OSError):
        return None

    if (package_root / "pyproject.toml").is_file() and (package_root / "consortium").is_dir():
        return package_root
    return None


def find_script_path(name: str, *, start: Path | None = None) -> Path | None:
    """Find a repo-managed helper script by absolute path."""
    project_root = find_project_root(start)
    if project_root is None:
        return None
    script_path = project_root / "scripts" / name
    return script_path if script_path.is_file() else None


def build_runner_argv(args: list[str]) -> list[str]:
    """Build a package-native consortium runner invocation."""
    runner_args = list(args)
    if runner_args and runner_args[0] == "consortium":
        runner_args = runner_args[1:]
    return [sys.executable, "-m", "consortium.runner", *runner_args]


def find_results_dir(start: Path | None = None) -> Path | None:
    """Find the results directory by checking supported locations."""
    search_root = (start or Path.cwd()).resolve()
    project_root = find_project_root(search_root)
    candidates = _unique_paths(
        [
            search_root / "results",
            search_root.parent / "results",
            project_root / "results" if project_root else None,
        ]
    )
    for c in candidates:
        if c.is_dir():
            return c
    return None
