"""Read, write, and merge msc configuration from multiple layers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from consortium.cli.core.presets import TIERS, resolve_tier_name

DEFAULT_CONFIG_DIR_NAME = ".msc"
PROJECT_CONFIG_DIR = ".msc"

BASE_DEFAULTS: dict[str, Any] = {
    "tier": "medium",
    "preset": "medium",  # backward compat alias for tier
    "mode": "auto",
    "autonomous_mode": True,
    "notifications": {
        "telegram": {"enabled": False},
        "slack": {"enabled": False},
    },
}

_TIER_DERIVED_KEYS = (
    "model",
    "output_format",
    "budget_usd",
    "enable_counsel",
    "enable_math_agents",
    "enable_tree_search",
    "adversarial_verification",
    "enable_planning",
    "enforce_paper_artifacts",
    "enforce_editorial_artifacts",
    "autonomous_mode",
    "enable_ensemble_review",
    "followup_max_iterations",
    "max_rebuttal_iterations",
    "min_review_score",
    "manager_max_steps",
    "theory_repair_max_attempts",
    "duality_max_attempts",
    "persona_post_vote_retries",
    "max_validation_retries",
    "tree_max_breadth",
    "tree_max_depth",
    "tree_max_parallel",
    "tree_pruning_threshold",
)


def get_config_dir(override: str | None = None) -> Path:
    """Return the global config directory, creating it if needed."""
    d = Path(override) if override else Path.home() / DEFAULT_CONFIG_DIR_NAME
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


def _tier_defaults(tier_name: str | None) -> dict[str, Any]:
    """Return config fields derived from a tier definition."""
    if not tier_name:
        return {}
    try:
        tier = TIERS[resolve_tier_name(str(tier_name))]
    except (KeyError, ValueError):
        return {}

    return {
        "tier": tier.name,
        "preset": tier.name,
        "model": tier.model,
        "output_format": tier.output_format,
        "budget_usd": tier.budget_usd,
        "enable_counsel": tier.enable_counsel,
        "enable_math_agents": tier.enable_math_agents,
        "enable_tree_search": tier.enable_tree_search,
        "adversarial_verification": tier.adversarial_verification,
        "enable_planning": tier.enable_planning,
        "enforce_paper_artifacts": tier.enforce_paper_artifacts,
        "enforce_editorial_artifacts": tier.enforce_editorial_artifacts,
        "autonomous_mode": tier.autonomous_mode,
        "enable_ensemble_review": tier.enable_ensemble_review,
        "followup_max_iterations": tier.followup_max_iterations,
        "max_rebuttal_iterations": tier.max_rebuttal_iterations,
        "min_review_score": tier.min_review_score,
        "manager_max_steps": tier.manager_max_steps,
        "theory_repair_max_attempts": tier.theory_repair_max_attempts,
        "duality_max_attempts": tier.duality_max_attempts,
        "persona_post_vote_retries": tier.persona_post_vote_retries,
        "max_validation_retries": tier.max_validation_retries,
        "tree_max_breadth": tier.tree_max_breadth,
        "tree_max_depth": tier.tree_max_depth,
        "tree_max_parallel": tier.tree_max_parallel,
        "tree_pruning_threshold": tier.tree_pruning_threshold,
    }


def _sync_tier_aliases(cfg: dict[str, Any]) -> dict[str, Any]:
    """Keep tier/preset aliases consistent in legacy and new configs."""
    normalized = cfg.copy()
    tier_name = normalized.get("tier") or normalized.get("preset")
    if tier_name is None:
        return normalized

    try:
        resolved = resolve_tier_name(str(tier_name))
    except ValueError:
        return normalized
    normalized["tier"] = resolved
    normalized["preset"] = resolved
    return normalized


def _strip_tier_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    """Remove tier-derived values so they do not masquerade as overrides."""
    normalized = _sync_tier_aliases(cfg)
    defaults = _tier_defaults(normalized.get("tier"))
    for key in _TIER_DERIVED_KEYS:
        if key in normalized and defaults.get(key) == normalized[key]:
            normalized.pop(key)
    return normalized


def _load_config_layers(
    config_dir_override: str | None = None,
    project_dir: Path | None = None,
    *,
    strip_tier_defaults: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the global and project config layers."""
    config_dir = get_config_dir(config_dir_override)
    project_root = project_dir or Path.cwd()

    global_cfg = _load_yaml(config_dir / "config.yaml")
    project_cfg = _load_yaml(project_root / PROJECT_CONFIG_DIR / "config.yaml")

    if strip_tier_defaults:
        global_cfg = _strip_tier_defaults(global_cfg)
        project_cfg = _strip_tier_defaults(project_cfg)
    else:
        global_cfg = _sync_tier_aliases(global_cfg)
        project_cfg = _sync_tier_aliases(project_cfg)

    return global_cfg, project_cfg


def load_explicit_config(
    config_dir_override: str | None = None,
    project_dir: Path | None = None,
) -> dict[str, Any]:
    """Load merged config without built-in defaults."""
    global_cfg, project_cfg = _load_config_layers(
        config_dir_override,
        project_dir,
        strip_tier_defaults=True,
    )
    merged: dict[str, Any] = {}
    merged = _deep_merge(merged, global_cfg)
    merged = _deep_merge(merged, project_cfg)
    return merged


def load_config(config_dir_override: str | None = None) -> dict[str, Any]:
    """Load merged config: defaults < global < project-local.

    Resolution order (highest priority wins):
    1. Project-local .msc/config.yaml
    2. Global ~/.msc/config.yaml
    3. Built-in defaults
    """
    explicit_cfg = load_explicit_config(config_dir_override)
    tier_name = explicit_cfg.get("tier") or explicit_cfg.get("preset") or BASE_DEFAULTS["tier"]

    merged = _deep_merge(BASE_DEFAULTS, _tier_defaults(str(tier_name)))
    merged = _deep_merge(merged, explicit_cfg)
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

    if key in {"tier", "preset"}:
        resolved = resolve_tier_name(value)
        cfg["tier"] = resolved
        cfg["preset"] = resolved
        save_config(cfg, config_dir_override)
        return

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
