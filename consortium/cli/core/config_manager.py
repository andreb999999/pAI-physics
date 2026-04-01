"""Read, write, and merge msc configuration from multiple layers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_DIR = Path.home() / ".msc"
PROJECT_CONFIG_DIR = ".msc"

DEFAULTS: dict[str, Any] = {
    "model": "claude-sonnet-4-6",
    "tier": "medium",
    "preset": "medium",  # backward compat alias for tier
    "output_format": "latex",
    "budget_usd": 200,
    "mode": "auto",
    "autonomous_mode": True,
    "enable_counsel": False,
    "enable_math_agents": False,
    "enable_tree_search": False,
    "notifications": {
        "telegram": {"enabled": False},
        "slack": {"enabled": False},
    },
}


def get_config_dir(override: str | None = None) -> Path:
    """Return the global config directory, creating it if needed."""
    d = Path(override) if override else DEFAULT_CONFIG_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning empty dict if missing or invalid."""
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except (yaml.YAMLError, OSError):
        return {}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively merge overlay into base, overlay wins on conflicts."""
    result = base.copy()
    for k, v in overlay.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_dir_override: str | None = None) -> dict[str, Any]:
    """Load merged config: defaults < global < project-local.

    Resolution order (highest priority wins):
    1. Project-local .msc/config.yaml
    2. Global ~/.msc/config.yaml
    3. Built-in defaults
    """
    config_dir = get_config_dir(config_dir_override)

    # Start with defaults
    merged = DEFAULTS.copy()

    # Layer global config
    global_cfg = _load_yaml(config_dir / "config.yaml")
    merged = _deep_merge(merged, global_cfg)

    # Layer project-local config
    project_cfg = _load_yaml(Path.cwd() / PROJECT_CONFIG_DIR / "config.yaml")
    merged = _deep_merge(merged, project_cfg)

    return merged


def save_config(data: dict[str, Any], config_dir_override: str | None = None) -> Path:
    """Save config to the global config directory."""
    config_dir = get_config_dir(config_dir_override)
    path = config_dir / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return path


def get_value(key: str, config_dir_override: str | None = None) -> Any:
    """Get a config value by dot-separated key (e.g. 'notifications.telegram.enabled')."""
    cfg = load_config(config_dir_override)
    parts = key.split(".")
    current: Any = cfg
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def set_value(key: str, value: str, config_dir_override: str | None = None) -> None:
    """Set a config value by dot-separated key. Writes to global config."""
    config_dir = get_config_dir(config_dir_override)
    cfg = _load_yaml(config_dir / "config.yaml")

    # Parse value type
    parsed: Any = value
    if value.lower() in ("true", "yes"):
        parsed = True
    elif value.lower() in ("false", "no"):
        parsed = False
    elif value.isdigit():
        parsed = int(value)
    else:
        try:
            parsed = float(value)
        except ValueError:
            pass

    # Navigate and set
    parts = key.split(".")
    current = cfg
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = parsed

    save_config(cfg, config_dir_override)
