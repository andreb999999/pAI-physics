"""Shared model-policy helpers for tier-generated CLI config."""

from __future__ import annotations

from typing import Any

CLAUDE_OPENROUTER_MAX_BUDGET_TOKENS = 31_999
DEFAULT_PERSONA_NAMES = (
    "practical_compass",
    "rigor_novelty",
    "narrative_architect",
)


def _normalize_thinking_budget(value: Any) -> int | None:
    if value is None:
        return None
    budget = int(value)
    return min(budget, CLAUDE_OPENROUTER_MAX_BUDGET_TOKENS)


def normalize_model_settings(model: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Normalize provider-specific settings before they reach runtime."""
    normalized = dict(settings)

    if "claude" in model:
        for key in ("budget_tokens", "thinking_budget"):
            if key in normalized and normalized[key] is not None:
                normalized[key] = _normalize_thinking_budget(normalized[key])

        model_kwargs = normalized.get("model_kwargs")
        if isinstance(model_kwargs, dict):
            normalized_kwargs = dict(model_kwargs)
            for key in ("budget_tokens", "thinking_budget"):
                if key in normalized_kwargs and normalized_kwargs[key] is not None:
                    normalized_kwargs[key] = _normalize_thinking_budget(normalized_kwargs[key])
            normalized["model_kwargs"] = normalized_kwargs

    return normalized


def build_default_persona_specs(
    *,
    model: str,
    reasoning_effort: str,
    budget_tokens: int | None,
) -> list[dict[str, Any]]:
    """Generate a low-surprise persona council that stays inside the tier model."""
    extras: dict[str, Any] = {}
    if "gemini" in model:
        if budget_tokens is not None:
            extras["thinking_budget"] = int(budget_tokens)
    else:
        extras["reasoning_effort"] = reasoning_effort
        if "claude" in model and budget_tokens is not None:
            extras["budget_tokens"] = int(budget_tokens)

    return [
        normalize_model_settings(model, {"persona": persona_name, "model": model, **extras})
        for persona_name in DEFAULT_PERSONA_NAMES
    ]


def persona_spec_to_runtime_spec(raw_spec: dict[str, Any]) -> dict[str, Any]:
    """Convert preset persona specs into the runtime format expected by persona_council."""
    spec = dict(raw_spec)
    model = str(spec["model"])
    runtime = {
        "persona": spec.get("persona") or spec.get("name"),
        "model": model,
    }
    model_kwargs = spec.get("model_kwargs") if isinstance(spec.get("model_kwargs"), dict) else {}
    runtime.update(model_kwargs)

    for key, value in spec.items():
        if key in {"persona", "name", "model", "model_kwargs"}:
            continue
        runtime[key] = value

    normalized = normalize_model_settings(model, runtime)
    if not normalized.get("persona"):
        raise ValueError(f"Persona spec missing persona/name field: {raw_spec}")
    return normalized
