"""Shared helpers for inspecting live and completed run workspaces."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from consortium.run_status import read_run_status

ACTIVE_THRESHOLD_SECONDS = 30 * 60
STALLED_THRESHOLD_SECONDS = 10 * 60
FINAL_PAPER_CANDIDATES = (
    "paper_workspace/final_paper.pdf",
    "paper_workspace/final_paper.tex",
    "paper_workspace/final_paper.md",
    "final_paper.pdf",
    "final_paper.tex",
    "final_paper.md",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _read_jsonl_last(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {}


def _is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _humanize_age(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        minutes = int(seconds // 60)
        remainder = int(seconds % 60)
        return f"{minutes}m {remainder}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def _iso_or_none(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _find_final_paper(run_dir: Path) -> str | None:
    for candidate in FINAL_PAPER_CANDIDATES:
        if (run_dir / candidate).exists():
            return candidate
    return None


def inspect_run(run_dir: Path) -> dict[str, Any]:
    """Return a normalized summary for one run workspace."""
    now = time.time()
    heartbeat_path = run_dir / ".progress_heartbeat"
    budget_state_path = run_dir / "budget_state.json"
    budget_ledger_path = run_dir / "budget_ledger.jsonl"
    summary_path = run_dir / "run_summary.json"
    metadata_path = run_dir / "experiment_metadata.json"

    heartbeat = _read_json(heartbeat_path)
    run_status = read_run_status(run_dir)
    summary = _read_json(summary_path)
    metadata = _read_json(metadata_path)
    budget_state = _read_json(budget_state_path)
    budget_ledger_last = _read_jsonl_last(budget_ledger_path)

    pid = run_status.get("pid") or heartbeat.get("pid")
    pid_running = bool(pid and _is_pid_running(int(pid)))

    timestamps = [
        p.stat().st_mtime
        for p in (heartbeat_path, budget_state_path, budget_ledger_path, summary_path, metadata_path)
        if p.exists()
    ]
    last_activity_ts = max(timestamps) if timestamps else None
    last_activity_age = (now - last_activity_ts) if last_activity_ts is not None else None

    total_cost = None
    for source in (budget_state, summary, budget_ledger_last):
        total_cost = _maybe_float(source.get("total_usd", source.get("total_cost_usd")))
        if total_cost is not None:
            break

    stage = (
        run_status.get("current_stage")
        or heartbeat.get("stage")
        or summary.get("current_stage")
        or summary.get("stage")
        or "-"
    )
    status_reason = run_status.get("status_reason") or summary.get("status_reason")
    final_paper = summary.get("final_paper") or _find_final_paper(run_dir)
    model = summary.get("model") or metadata.get("model") or "-"
    task = summary.get("task") or metadata.get("task_preview") or "-"

    status = str(run_status.get("status") or "").lower()
    if not status:
        if summary.get("status"):
            status = str(summary["status"]).lower()
        elif final_paper:
            status = "completed"
        elif pid_running or (heartbeat_path.exists() and (last_activity_age or 0) < ACTIVE_THRESHOLD_SECONDS):
            status = "active"
        elif status_reason:
            status = "failed"
        elif any(p.exists() for p in (budget_state_path, budget_ledger_path, metadata_path, summary_path)):
            status = "partial"
        else:
            status = "unknown"

    if status == "running":
        status = "active"
    if status == "active" and pid and not pid_running:
        status = "failed" if status_reason else "partial"
    if status == "active" and pid_running and last_activity_age is not None and last_activity_age > STALLED_THRESHOLD_SECONDS:
        status = "stalled"
    if status in {"partial", "unknown"} and final_paper:
        status = "completed"

    started_at = run_status.get("started_at") or summary.get("started_at")
    last_activity_at = _iso_or_none(last_activity_ts) or run_status.get("last_activity_at")

    return {
        "run_id": run_dir.name,
        "status": status,
        "status_reason": status_reason,
        "current_stage": stage,
        "last_activity_at": last_activity_at,
        "last_activity_age_seconds": last_activity_age,
        "elapsed": _humanize_age(last_activity_age),
        "budget_usd": total_cost,
        "model": model,
        "task": task,
        "final_paper": final_paper,
        "pid": int(pid) if pid else None,
        "pid_running": pid_running,
        "started_at": started_at,
        "workspace": str(run_dir),
    }
