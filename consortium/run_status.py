"""Helpers for persisting live and terminal run status artifacts."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUS_JSON = "run_status.json"
STATUS_TEXT = "STATUS.txt"


def status_json_path(workspace_dir: str | os.PathLike[str]) -> Path:
    return Path(workspace_dir) / STATUS_JSON


def status_text_path(workspace_dir: str | os.PathLike[str]) -> Path:
    return Path(workspace_dir) / STATUS_TEXT


def read_run_status(workspace_dir: str | os.PathLike[str]) -> dict[str, Any]:
    path = status_json_path(workspace_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _render_status_text(data: dict[str, Any]) -> str:
    lines = [str(data.get("status", "unknown")).upper()]
    stage = data.get("current_stage")
    if stage:
        lines.append(f"Stage: {stage}")
    reason = data.get("status_reason")
    if reason:
        lines.append(f"Reason: {reason}")
    last_activity_at = data.get("last_activity_at")
    if last_activity_at:
        lines.append(f"Last Activity: {last_activity_at}")
    pid = data.get("pid")
    if pid:
        lines.append(f"PID: {pid}")
    return "\n".join(lines) + "\n"


def write_run_status(
    workspace_dir: str | os.PathLike[str],
    *,
    status: str,
    current_stage: str | None = None,
    status_reason: str | None = None,
    pid: int | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    last_activity_at: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist run_status.json and STATUS.txt atomically."""
    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)

    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    existing = read_run_status(workspace)
    data = dict(existing)
    data["status"] = status
    data["updated_at"] = now_iso
    data["last_activity_at"] = last_activity_at or now_iso

    if current_stage is not None:
        data["current_stage"] = current_stage
    if status_reason is not None:
        data["status_reason"] = status_reason
    if pid is not None:
        data["pid"] = pid
    if started_at is not None:
        data["started_at"] = started_at
    elif "started_at" not in data:
        data["started_at"] = now_iso
    if finished_at is not None:
        data["finished_at"] = finished_at
    elif status not in {"completed", "failed", "stalled"}:
        data.pop("finished_at", None)
    if extra:
        data.update(extra)

    json_path = status_json_path(workspace)
    text_path = status_text_path(workspace)
    json_tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    text_tmp = text_path.with_suffix(text_path.suffix + ".tmp")

    json_tmp.write_text(json.dumps(data, indent=2))
    os.replace(json_tmp, json_path)
    text_tmp.write_text(_render_status_text(data))
    os.replace(text_tmp, text_path)
    return data
