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
from datetime import datetime
from typing import Any, Dict, Optional

_LOCK = threading.Lock()

ENV_TOKEN_FILE = "FREEPHDLABOR_RUN_TOKEN_FILE"
ENV_RUN_ID = "FREEPHDLABOR_RUN_ID"


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _safe_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except Exception:
        return 0


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


def record_token_usage(
    prompt_tokens: int,
    completion_tokens: int,
    source: str = "unknown",
    model_id: Optional[str] = None,
) -> None:
    """
    Add token usage to current run totals.
    """
    path = os.getenv(ENV_TOKEN_FILE)
    run_id = os.getenv(ENV_RUN_ID)
    if not path:
        return

    pt = _safe_int(prompt_tokens)
    ct = _safe_int(completion_tokens)
    if pt == 0 and ct == 0:
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


def patch_smolagents_monitoring() -> None:
    """
    Monkey-patch smolagents Monitor.update_metrics to display run-wide totals
    from this tracker. Falls back to original per-step accumulation if tracker
    isn't initialized.
    """
    from smolagents.monitoring import Monitor, Text

    if getattr(Monitor, "_freephdlabor_run_token_patch", False):
        return

    original_update_metrics = Monitor.update_metrics

    def _patched_update_metrics(self, step_log):
        totals = get_run_token_totals()
        if totals is None:
            # Fallback to default behavior when tracker is unavailable
            return original_update_metrics(self, step_log)

        step_duration = step_log.timing.duration if step_log.timing.duration is not None else 0.0
        self.step_durations.append(step_duration)
        # Keep Monitor state fields consistent for any downstream consumers.
        self.total_input_token_count = totals["input_tokens"]
        self.total_output_token_count = totals["output_tokens"]
        console_outputs = f"[Step {len(self.step_durations)}: Duration {step_duration:.2f} seconds"
        console_outputs += (
            f"| Input tokens: {totals['input_tokens']:,} | Output tokens: {totals['output_tokens']:,}"
        )

        console_outputs += "]"
        self.logger.log(Text(console_outputs, style="dim"), level=1)

    Monitor.update_metrics = _patched_update_metrics
    Monitor._freephdlabor_run_token_patch = True
