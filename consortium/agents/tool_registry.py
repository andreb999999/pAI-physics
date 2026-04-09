"""Singleton tool registry — caches tool instances per (class, workspace_dir, model_id).

Avoids re-instantiating the same heavy tools (OpenRouterDeepResearchTool,
FetchArxivPapersTool, CitationSearchTool, etc.) across the 22+ agents in
the pipeline.  Thread-safe.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Type

from langchain_core.tools import BaseTool

_REGISTRY: Dict[tuple, BaseTool] = {}
_LOCK = threading.Lock()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(v) for v in value)
    return value


def get_or_create(
    tool_cls: Type[BaseTool],
    *,
    working_dir: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    model: Optional[str] = None,
    model_name: Optional[str] = None,
    **extra_kwargs: Any,
) -> BaseTool:
    """Return a cached tool instance, creating one if needed.

    The cache key is ``(tool_cls, working_dir or workspace_dir, model or model_name)``.
    Extra kwargs (e.g., ``authorized_imports``) are passed to the constructor
    on first creation but are NOT part of the cache key — tools with the same
    class + directory + model share an instance.
    """
    dir_key = working_dir or workspace_dir or ""
    model_key = model or model_name or ""
    key = (tool_cls, dir_key, model_key, _freeze(extra_kwargs))

    with _LOCK:
        if key in _REGISTRY:
            return _REGISTRY[key]

    # Build constructor kwargs from the non-None params
    kwargs: dict[str, Any] = {}
    if working_dir is not None:
        kwargs["working_dir"] = working_dir
    if workspace_dir is not None:
        kwargs["workspace_dir"] = workspace_dir
    if model is not None:
        kwargs["model"] = model
    if model_name is not None:
        kwargs["model_name"] = model_name
    kwargs.update(extra_kwargs)

    try:
        instance = tool_cls(**kwargs) if kwargs else tool_cls()
    except TypeError:
        # Fallback: some tools have unusual constructors
        instance = tool_cls()

    with _LOCK:
        # Double-check after acquiring lock
        if key not in _REGISTRY:
            _REGISTRY[key] = instance
        return _REGISTRY[key]


def clear() -> None:
    """Clear the tool cache (useful between test runs)."""
    with _LOCK:
        _REGISTRY.clear()
