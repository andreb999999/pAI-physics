"""
Context compaction middleware for LangGraph agent nodes (Phase 6a).

Replaces ContextMonitoringCallback (smolagents step callback) with a
trim_messages-based approach that operates on LangGraph state messages.

Usage — wrap any specialist node with compact_context_middleware():

    from consortium.context_compaction import compact_context_middleware
    ideation_node = compact_context_middleware(raw_ideation_node, model_id="claude-opus-4-6")
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, trim_messages


# ---------------------------------------------------------------------------
# Context limit table (mirrors agents/base_agent.py)
# ---------------------------------------------------------------------------

MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-sonnet-4-5-20250929": 200_000,
    "anthropic/claude-opus-4-6": 200_000,
    "anthropic/claude-sonnet-4-6": 200_000,
    "gpt-5": 256_000,
    "gpt-5-mini": 256_000,
    "gpt-5-nano": 256_000,
    "gpt-5.4": 1_050_000,
    "gpt-5.3-codex": 200_000,
    "gpt-4o": 128_000,
    "o3-2025-04-16": 200_000,
    "o4-mini-2025-04-16": 128_000,
    "gemini-2.5-pro": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
    "deepseek-chat": 64_000,
    "deepseek-coder": 64_000,
    "grok-4-0709": 128_000,
}
_DEFAULT_LIMIT = 128_000
_SAFETY_MARGIN = 0.75
# Rough chars-per-token estimate used for fast threshold checking
_CHARS_PER_TOKEN = 4


def _context_limit(model_id: str) -> int:
    return MODEL_CONTEXT_LIMITS.get(model_id, _DEFAULT_LIMIT)


def _estimate_tokens(messages: List[BaseMessage]) -> int:
    """Fast token count estimate from message content length."""
    chars = sum(len(str(getattr(m, "content", ""))) for m in messages)
    return chars // _CHARS_PER_TOKEN + 3_500  # constant overhead


# ---------------------------------------------------------------------------
# Backup helper
# ---------------------------------------------------------------------------

def _backup_messages(messages: List[BaseMessage], workspace_dir: str) -> None:
    """Append compacted message summaries to a JSONL backup file."""
    try:
        backup_dir = os.path.join(workspace_dir, "memory_backup")
        os.makedirs(backup_dir, exist_ok=True)
        backup_file = os.path.join(backup_dir, "full_conversation_backup.jsonl")
        with open(backup_file, "a", encoding="utf-8") as f:
            for msg in messages:
                entry = {
                    "timestamp": time.time(),
                    "role": msg.__class__.__name__,
                    "content": str(getattr(msg, "content", ""))[:500],
                }
                f.write(json.dumps(entry, default=str) + "\n")
    except Exception as exc:
        print(f"⚠️ Failed to backup messages: {exc}")


# ---------------------------------------------------------------------------
# Core compaction function
# ---------------------------------------------------------------------------

def maybe_compact_messages(
    messages: List[BaseMessage],
    model_id: str = "",
    workspace_dir: Optional[str] = None,
    safety_margin: float = _SAFETY_MARGIN,
    keep_last_n: int = 6,
) -> List[BaseMessage]:
    """
    If the estimated token count exceeds the safety threshold, trim older
    messages using LangChain's trim_messages utility, preserving the system
    prompt (first message) and the most recent `keep_last_n` messages.

    Returns the (possibly trimmed) message list.
    """
    limit = _context_limit(model_id)
    threshold = int(limit * safety_margin)
    estimated = _estimate_tokens(messages)

    if estimated <= threshold:
        return messages

    print(f"🧠 Context compaction triggered: ~{estimated:,} tokens > {threshold:,} threshold")

    if workspace_dir:
        # Backup the messages being dropped
        n_drop = max(0, len(messages) - keep_last_n - 1)
        if n_drop > 0:
            _backup_messages(messages[1:n_drop + 1], workspace_dir)

    # Preserve system message + last keep_last_n messages
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    kept = non_system[-keep_last_n:] if len(non_system) > keep_last_n else non_system

    compacted = [
        HumanMessage(
            content=(
                f"[CONTEXT COMPACTED — earlier conversation summarised]\n"
                f"Approx {estimated:,} tokens were truncated. "
                f"Continuing from the most recent {keep_last_n} exchanges."
            )
        )
    ]

    result = system_msgs + compacted + kept
    new_est = _estimate_tokens(result)
    print(f"✅ Compaction complete: {estimated:,} → ~{new_est:,} tokens")
    return result


# ---------------------------------------------------------------------------
# Middleware wrapper
# ---------------------------------------------------------------------------

def compact_context_middleware(
    node_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    model_id: str = "",
    workspace_dir: Optional[str] = None,
    safety_margin: float = _SAFETY_MARGIN,
    keep_last_n: int = 6,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Wrap a LangGraph node function to trim messages before invoking it.

    Args:
        node_fn:        The node callable to wrap.
        model_id:       Model name (used to look up context limit).
        workspace_dir:  If provided, dropped messages are backed up here.
        safety_margin:  Fraction of context window to use before compacting.
        keep_last_n:    Number of recent messages to keep after compaction.

    Returns:
        A wrapped node callable with the same signature.
    """

    def wrapped(state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        compacted = maybe_compact_messages(
            messages=messages,
            model_id=model_id,
            workspace_dir=workspace_dir,
            safety_margin=safety_margin,
            keep_last_n=keep_last_n,
        )
        if compacted is not messages:
            state = {**state, "messages": compacted}
        return node_fn(state)

    wrapped.__name__ = getattr(node_fn, "__name__", "wrapped_node")
    return wrapped
