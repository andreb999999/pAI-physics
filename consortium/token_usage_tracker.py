"""
Run-scoped token usage tracker.

This module stores cumulative input/output token usage for the *current launcher run*
in a JSON file under the active workspace. It is designed to aggregate usage from:
- Budgeted model calls (LiteLLMModel wrapper path)
- Direct OpenAI/Anthropic client calls used by helper/tool modules
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional

from .workflow_utils import safe_int as _safe_int

_LOCK = threading.Lock()

ENV_TOKEN_FILE = "CONSORTIUM_RUN_TOKEN_FILE"
ENV_RUN_ID = "CONSORTIUM_RUN_ID"
ENV_PRIVATE_TOKEN_LEDGER = "CONSORTIUM_PRIVATE_TOKEN_LEDGER"
ENV_PRIVATE_TOKEN_TEXT = "CONSORTIUM_PRIVATE_TOKEN_TEXT"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _private_token_dir() -> str:
    return os.path.join(_project_root(), ".local", "private_token_usage")


def _private_ledger_path() -> str:
    env_path = os.getenv(ENV_PRIVATE_TOKEN_LEDGER)
    if env_path:
        return os.path.abspath(env_path)
    return os.path.join(_private_token_dir(), "api_token_calls.jsonl")


def _private_text_path() -> str:
    env_path = os.getenv(ENV_PRIVATE_TOKEN_TEXT)
    if env_path:
        return os.path.abspath(env_path)
    return os.path.join(_private_token_dir(), "api_token_calls.txt")


def _read_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_state(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _append_private_token_entry(
    prompt_tokens: int,
    completion_tokens: int,
    source: str,
    model_id: Optional[str],
    run_id: Optional[str],
    tracker_path: Optional[str],
) -> None:
    """
    Append per-call token usage to a local private ledger and text log.
    Never raise (tracking should not break model execution paths).
    """
    try:
        ts = _now_iso()
        pt = _safe_int(prompt_tokens)
        ct = _safe_int(completion_tokens)
        total = pt + ct
        row = {
            "timestamp": ts,
            "run_id": run_id or "unknown_run",
            "source": source or "unknown",
            "model_id": model_id or "",
            "input_tokens": pt,
            "output_tokens": ct,
            "total_tokens": total,
            "run_token_file": tracker_path or "",
        }
        _append_jsonl(_private_ledger_path(), row)

        txt_path = _private_text_path()
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(
                f"{ts}\trun={row['run_id']}\tsource={row['source']}\t"
                f"model={row['model_id'] or 'unknown'}\t"
                f"input={pt}\toutput={ct}\ttotal={total}\n"
            )
    except Exception:
        # Local private logging is best-effort only.
        return


def initialize_run_token_tracker(workspace_dir: str, run_id: str, reset: bool = True) -> str:
    """
    Initialize per-launch token tracking file.

    Args:
        workspace_dir: Active workspace directory for this run.
        run_id: Unique run identifier (typically launcher timestamp).
        reset: If True, overwrite any existing tracker file for a fresh run.

    Returns:
        Absolute path to the tracker JSON file.
    """
    path = os.path.abspath(os.path.join(workspace_dir, "run_token_usage.json"))
    os.environ[ENV_TOKEN_FILE] = path
    os.environ[ENV_RUN_ID] = run_id
    os.environ.setdefault(ENV_PRIVATE_TOKEN_LEDGER, _private_ledger_path())
    os.environ.setdefault(ENV_PRIVATE_TOKEN_TEXT, _private_text_path())

    with _LOCK:
        if reset or not os.path.exists(path):
            state = {
                "run_id": run_id,
                "started_at": _now_iso(),
                "updated_at": _now_iso(),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "by_source": {},
                "by_model": {},
            }
            _write_state(path, state)
        else:
            state = _read_state(path)
            state["run_id"] = run_id
            state["updated_at"] = _now_iso()
            _write_state(path, state)
    return path


@contextmanager
def token_tracker_scope(
    workspace_dir: str, run_id: str
) -> Generator[str, None, None]:
    """Context manager that temporarily redirects token tracking to a scoped file.

    Saves and restores the parent run's token tracker env vars on exit,
    preventing subprocess/sandbox calls from clobbering the main run's tracker.

    Usage::

        with token_tracker_scope(sandbox_dir, "counsel_debate"):
            # all token recording in this block goes to sandbox_dir
            ...
        # parent tracker is restored automatically
    """
    saved_token = os.environ.get(ENV_TOKEN_FILE)
    saved_run_id = os.environ.get(ENV_RUN_ID)
    try:
        path = initialize_run_token_tracker(workspace_dir, run_id, reset=False)
        yield path
    finally:
        if saved_token is not None:
            os.environ[ENV_TOKEN_FILE] = saved_token
        else:
            os.environ.pop(ENV_TOKEN_FILE, None)
        if saved_run_id is not None:
            os.environ[ENV_RUN_ID] = saved_run_id
        else:
            os.environ.pop(ENV_RUN_ID, None)


def record_token_usage(
    prompt_tokens: int,
    completion_tokens: int,
    source: str = "unknown",
    model_id: Optional[str] = None,
) -> None:
    """
    Add token usage to current run totals.
    """
    pt = _safe_int(prompt_tokens)
    ct = _safe_int(completion_tokens)
    if pt == 0 and ct == 0:
        return

    path = os.getenv(ENV_TOKEN_FILE)
    run_id = os.getenv(ENV_RUN_ID)
    if not path:
        _append_private_token_entry(
            prompt_tokens=pt,
            completion_tokens=ct,
            source=source,
            model_id=model_id,
            run_id=run_id,
            tracker_path=None,
        )
        return

    with _LOCK:
        state = _read_state(path)
        if not state:
            state = {
                "run_id": run_id or "unknown_run",
                "started_at": _now_iso(),
                "updated_at": _now_iso(),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "by_source": {},
                "by_model": {},
            }

        state["run_id"] = run_id or state.get("run_id", "unknown_run")
        state["input_tokens"] = _safe_int(state.get("input_tokens")) + pt
        state["output_tokens"] = _safe_int(state.get("output_tokens")) + ct
        state["total_tokens"] = state["input_tokens"] + state["output_tokens"]
        state["updated_at"] = _now_iso()

        by_source = state.setdefault("by_source", {})
        by_source[source] = _safe_int(by_source.get(source)) + pt + ct

        if model_id:
            by_model = state.setdefault("by_model", {})
            by_model[model_id] = _safe_int(by_model.get(model_id)) + pt + ct

        _write_state(path, state)
        _append_private_token_entry(
            prompt_tokens=pt,
            completion_tokens=ct,
            source=source,
            model_id=model_id,
            run_id=run_id,
            tracker_path=path,
        )


def get_run_token_totals() -> Optional[Dict[str, int]]:
    """
    Return current run token totals, or None if tracker isn't initialized.
    """
    path = os.getenv(ENV_TOKEN_FILE)
    if not path:
        return None
    state = _read_state(path)
    if not state:
        return None
    return {
        "input_tokens": _safe_int(state.get("input_tokens")),
        "output_tokens": _safe_int(state.get("output_tokens")),
        "total_tokens": _safe_int(state.get("total_tokens")),
    }

