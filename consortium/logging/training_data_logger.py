"""
Training Data Logger — captures all LLM input/output pairs for post-training.

Writes OpenAI fine-tuning compatible JSONL with a ``messages`` array and
``_metadata`` sidecar.  Three call paths feed into the single logger:

1. **litellm success_callback** — covers all litellm.completion() calls
   (LangGraph agents via ChatLiteLLM, counsel debate/synthesis, persona council).
2. **Explicit VLM instrumentation** — direct Anthropic/OpenAI client calls in llm.py.
3. **Thread-local agent context** — agent nodes set their name before invoke so
   the litellm callback can tag records with the originating agent.

Usage::

    # In runner.py, once at startup:
    initialize_training_data_logger(workspace_dir, run_id, enabled=True)

    # The litellm callback calls log_completion() automatically.
    # VLM code calls log_completion() explicitly.

    # Agent nodes set context:
    set_current_agent_name("literature_review_agent")
    try:
        ...
    finally:
        set_current_agent_name(None)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-local agent name context
# ---------------------------------------------------------------------------

_agent_context = threading.local()


def set_current_agent_name(name: Optional[str]) -> None:
    """Set the agent name for the current thread (used by litellm callback)."""
    _agent_context.agent_name = name


def get_current_agent_name() -> Optional[str]:
    """Return the agent name set for the current thread, or None."""
    return getattr(_agent_context, "agent_name", None)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_global_logger: Optional[TrainingDataLogger] = None  # type: ignore[name-defined]  # forward ref
_init_lock = threading.Lock()


def initialize_training_data_logger(
    workspace_dir: str,
    run_id: str,
    enabled: bool = True,
    include_tool_calls: bool = True,
) -> TrainingDataLogger:
    """Create and store the global TrainingDataLogger singleton."""
    global _global_logger
    with _init_lock:
        _global_logger = TrainingDataLogger(
            workspace_dir=workspace_dir,
            run_id=run_id,
            enabled=enabled,
            include_tool_calls=include_tool_calls,
        )
    return _global_logger


def get_training_data_logger() -> Optional[TrainingDataLogger]:
    """Return the global logger, or None if not initialised / disabled."""
    return _global_logger


# ---------------------------------------------------------------------------
# Global ledger path
# ---------------------------------------------------------------------------

_GLOBAL_LEDGER_DIR = os.path.join(".local", "private_training_data")
_GLOBAL_LEDGER_FILE = "training_data.jsonl"


def _global_ledger_path() -> str:
    return os.path.join(_GLOBAL_LEDGER_DIR, _GLOBAL_LEDGER_FILE)


# ---------------------------------------------------------------------------
# Core logger
# ---------------------------------------------------------------------------


class TrainingDataLogger:
    """Captures LLM input/output pairs and writes them as fine-tuning JSONL."""

    def __init__(
        self,
        workspace_dir: str,
        run_id: str,
        enabled: bool = True,
        include_tool_calls: bool = True,
    ):
        self.enabled = enabled
        self.run_id = run_id
        self.include_tool_calls = include_tool_calls
        self._lock = threading.Lock()

        self._per_run_path = os.path.abspath(
            os.path.join(workspace_dir, "training_data.jsonl")
        )
        self._global_path = os.path.abspath(_global_ledger_path())

        if enabled:
            os.makedirs(os.path.dirname(self._per_run_path), exist_ok=True)
            os.makedirs(os.path.dirname(self._global_path), exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_completion(
        self,
        messages: Any,
        response_content: str,
        model_id: str = "unknown",
        source: str = "unknown",
        agent_name: Optional[str] = None,
        token_usage: Optional[Dict[str, int]] = None,
        call_id: Optional[str] = None,
        duration_ms: Optional[int] = None,
        tool_calls: Optional[List[Dict]] = None,
    ) -> None:
        """Log a single LLM call in OpenAI fine-tuning JSONL format.

        All errors are swallowed — logging must never crash a run.
        """
        if not self.enabled:
            return
        try:
            self._log_completion_inner(
                messages, response_content, model_id, source,
                agent_name, token_usage, call_id, duration_ms, tool_calls,
            )
        except Exception:
            logger.debug("Training data logging failed", exc_info=True)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_completion_inner(
        self,
        messages: Any,
        response_content: str,
        model_id: str,
        source: str,
        agent_name: Optional[str],
        token_usage: Optional[Dict[str, int]],
        call_id: Optional[str],
        duration_ms: Optional[int],
        tool_calls: Optional[List[Dict]],
    ) -> None:
        normalised = self._normalize_messages(messages)

        # Append assistant response
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": response_content or "",
        }
        if self.include_tool_calls and tool_calls:
            assistant_msg["tool_calls"] = self._sanitize_tool_calls(tool_calls)
        normalised.append(assistant_msg)

        # Resolve agent name: explicit arg > thread-local context
        resolved_agent = agent_name or get_current_agent_name()

        # Build token_usage dict
        usage = {}
        if token_usage:
            usage["input_tokens"] = token_usage.get(
                "input_tokens", token_usage.get("prompt_tokens", 0)
            )
            usage["output_tokens"] = token_usage.get(
                "output_tokens", token_usage.get("completion_tokens", 0)
            )

        entry = {
            "messages": normalised,
            "_metadata": {
                "call_id": call_id or str(uuid.uuid4()),
                "run_id": self.run_id,
                "model_id": model_id,
                "agent_name": resolved_agent,
                "source": source,
                "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                **usage,
                **({"duration_ms": duration_ms} if duration_ms is not None else {}),
            },
        }
        self._write(entry)

    def _normalize_messages(self, messages: Any) -> List[Dict[str, str]]:
        """Convert various message formats to OpenAI ``[{role, content}]``.

        Handles:
        - List[dict] with ``role``/``content`` keys (OpenAI / litellm format)
        - LangChain BaseMessage subclasses
        - Nested list-of-lists (batch format from LangChain callbacks)
        - Single string (treated as user message)

        Base64 image data is replaced with ``[IMAGE]`` to keep files small.
        """
        if messages is None:
            return []
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        result: List[Dict[str, str]] = []

        items = messages
        # Flatten one level of nesting (LangChain batch format)
        if items and isinstance(items, list) and isinstance(items[0], list):
            flat: list = []
            for batch in items:
                flat.extend(batch)
            items = flat

        for msg in items:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                content = self._scrub_images(content)
                result.append({"role": role, "content": content})
            elif hasattr(msg, "content") and hasattr(msg, "type"):
                # LangChain BaseMessage
                role = _langchain_role(msg)
                raw_content = getattr(msg, "content", "")
                content = self._scrub_images(raw_content)
                result.append({"role": role, "content": content})
            else:
                result.append({"role": "user", "content": str(msg)})

        return result

    def _scrub_images(self, content: Any) -> str:
        """Replace base64 image data with a placeholder."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Anthropic / OpenAI multimodal content blocks
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(block.get("text", ""))
                    elif btype in ("image", "image_url"):
                        parts.append("[IMAGE]")
                    else:
                        parts.append(str(block))
                else:
                    parts.append(str(block))
            return "\n".join(parts)
        return str(content)

    def _sanitize_tool_calls(self, tool_calls: Any) -> List[Dict]:
        """Best-effort serialisation of tool_calls for the JSONL record."""
        sanitised: List[Dict] = []
        if not tool_calls:
            return sanitised
        for tc in tool_calls:
            if isinstance(tc, dict):
                sanitised.append(tc)
            elif hasattr(tc, "model_dump"):
                sanitised.append(tc.model_dump())
            elif hasattr(tc, "__dict__"):
                sanitised.append(tc.__dict__)
            else:
                sanitised.append({"raw": str(tc)})
        return sanitised

    def _write(self, entry: Dict[str, Any]) -> None:
        """Append a JSONL line to both per-run and global ledger files."""
        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            self._append(self._per_run_path, line)
            self._append(self._global_path, line)

    @staticmethod
    def _append(path: str, line: str) -> None:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as exc:
            logger.debug("Failed to write training data to %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _langchain_role(msg: Any) -> str:
    """Map a LangChain message type to an OpenAI role string."""
    type_name = getattr(msg, "type", "") or msg.__class__.__name__
    mapping = {
        "human": "user",
        "HumanMessage": "user",
        "ai": "assistant",
        "AIMessage": "assistant",
        "system": "system",
        "SystemMessage": "system",
        "tool": "tool",
        "ToolMessage": "tool",
        "function": "function",
        "FunctionMessage": "function",
    }
    return mapping.get(type_name, "user")
