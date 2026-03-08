"""
Campaign status — read/write campaign_status.json and check stage completion.

campaign_status.json schema:
{
  "campaign_name": str,
  "spec_file": str,
  "stages": {
    "<stage_id>": {
      "status": "pending" | "in_progress" | "completed" | "failed",
      "workspace": str | null,
      "pid": int | null,
      "started_at": ISO str | null,
      "completed_at": ISO str | null,
      "missing_artifacts": [str],
      "fail_reason": str | null
    }
  }
}
"""

from __future__ import annotations

import json
import os
import signal
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

try:
    import filelock as _filelock_module
    _FILELOCK_AVAILABLE = True
except ImportError:
    _FILELOCK_AVAILABLE = False

from .spec import CampaignSpec, Stage

STATUS_FILE = "campaign_status.json"
_LOCK_TIMEOUT = 30  # seconds


@contextmanager
def _status_lock(campaign_dir: str):
    """Context manager that holds an exclusive file lock on the status file.

    Falls back to a no-op context manager if filelock is not installed.
    """
    lock_path = os.path.join(campaign_dir, "campaign_status.lock")
    if _FILELOCK_AVAILABLE:
        lock = _filelock_module.FileLock(lock_path, timeout=_LOCK_TIMEOUT)
        with lock:
            yield
    else:
        yield

# Stage status constants
PENDING = "pending"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"
FAILED = "failed"


class CampaignStatus:
    """Thin wrapper around the status dict with typed accessors."""

    def __init__(self, data: dict):
        self._d = data

    # ------------------------------------------------------------------
    # Stage accessors
    # ------------------------------------------------------------------

    def stage_status(self, stage_id: str) -> str:
        return self._d["stages"].get(stage_id, {}).get("status", PENDING)

    def stage_workspace(self, stage_id: str) -> Optional[str]:
        return self._d["stages"].get(stage_id, {}).get("workspace")

    def stage_pid(self, stage_id: str) -> Optional[int]:
        pid = self._d["stages"].get(stage_id, {}).get("pid")
        return int(pid) if pid is not None else None

    def is_complete(self, stage_id: str) -> bool:
        return self.stage_status(stage_id) == COMPLETED

    def is_in_progress(self, stage_id: str) -> bool:
        return self.stage_status(stage_id) == IN_PROGRESS

    def all_complete(self, stage_ids: List[str]) -> bool:
        return all(self.is_complete(sid) for sid in stage_ids)

    def campaign_finished(self, spec: CampaignSpec) -> bool:
        return all(self.is_complete(s.id) for s in spec.stages)

    def campaign_failed(self, spec: CampaignSpec) -> bool:
        return any(self.stage_status(s.id) == FAILED for s in spec.stages)

    # ------------------------------------------------------------------
    # Mutations (return new status dict — callers must call write_status)
    # ------------------------------------------------------------------

    def mark_in_progress(self, stage_id: str, workspace: str, pid: int) -> "CampaignStatus":
        self._d["stages"].setdefault(stage_id, {})
        self._d["stages"][stage_id].update({
            "status": IN_PROGRESS,
            "workspace": workspace,
            "pid": pid,
            "started_at": _now(),
            "completed_at": None,
            "missing_artifacts": [],
            "fail_reason": None,
        })
        return self

    def mark_completed(self, stage_id: str) -> "CampaignStatus":
        self._d["stages"].setdefault(stage_id, {})
        self._d["stages"][stage_id].update({
            "status": COMPLETED,
            "pid": None,
            "completed_at": _now(),
            "missing_artifacts": [],
        })
        return self

    def mark_failed(self, stage_id: str, reason: str, missing: List[str]) -> "CampaignStatus":
        self._d["stages"].setdefault(stage_id, {})
        self._d["stages"][stage_id].update({
            "status": FAILED,
            "pid": None,
            "completed_at": _now(),
            "missing_artifacts": missing,
            "fail_reason": reason,
        })
        return self

    def raw(self) -> dict:
        return self._d


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

def read_status(campaign_dir: str) -> CampaignStatus:
    """Read campaign_status.json under an exclusive lock, returning empty status if not found."""
    path = os.path.join(campaign_dir, STATUS_FILE)
    with _status_lock(campaign_dir):
        if os.path.exists(path):
            with open(path) as f:
                return CampaignStatus(json.load(f))
    return CampaignStatus({"campaign_name": "", "spec_file": "", "stages": {}})


def write_status(campaign_dir: str, status: CampaignStatus) -> None:
    """Write campaign_status.json atomically under an exclusive lock."""
    os.makedirs(campaign_dir, exist_ok=True)
    path = os.path.join(campaign_dir, STATUS_FILE)
    tmp_path = path + ".tmp"
    with _status_lock(campaign_dir):
        with open(tmp_path, "w") as f:
            json.dump(status.raw(), f, indent=2)
        os.replace(tmp_path, path)  # atomic on POSIX


def init_status(campaign_dir: str, spec: "CampaignSpec", spec_file: str) -> CampaignStatus:
    """Create or update the status file for a campaign, preserving existing stage data."""
    os.makedirs(campaign_dir, exist_ok=True)
    path = os.path.join(campaign_dir, STATUS_FILE)
    tmp_path = path + ".tmp"
    with _status_lock(campaign_dir):
        existing_data: dict = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    existing_data = json.load(f)
            except Exception:
                existing_data = {}
        existing_stages = existing_data.get("stages", {})
        data = {
            "campaign_name": spec.name,
            "spec_file": os.path.abspath(spec_file),
            "stages": {},
        }
        for stage in spec.stages:
            existing_stage = existing_stages.get(stage.id, {})
            data["stages"][stage.id] = existing_stage or {
                "status": PENDING,
                "workspace": None,
                "pid": None,
                "started_at": None,
                "completed_at": None,
                "missing_artifacts": [],
                "fail_reason": None,
            }
        status = CampaignStatus(data)
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    return status


# ------------------------------------------------------------------
# Artifact + process checking
# ------------------------------------------------------------------

def check_stage_artifacts(workspace: str, stage: Stage) -> Tuple[bool, List[str]]:
    """
    Check whether all required artifacts for a stage exist in workspace.

    Returns (all_present: bool, list_of_missing_paths).
    """
    missing = []
    for artifact in stage.success_artifacts.get("required", []):
        full = os.path.join(workspace, artifact)
        # Support both files and directories (trailing /)
        if artifact.endswith("/"):
            if not os.path.isdir(full.rstrip("/")):
                missing.append(artifact)
        else:
            if not os.path.exists(full):
                missing.append(artifact)
    return len(missing) == 0, missing


def is_pid_alive(pid: int) -> bool:
    """Return True if a process with the given PID is currently running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't have permission to signal it
        return True


def infer_workspace_from_status_txt(candidate_dir: str) -> Optional[str]:
    """
    If a stage workspace is not yet set, try to find the most recent
    results subdirectory that contains a STATUS.txt file.
    """
    if not os.path.isdir(candidate_dir):
        return None
    subdirs = sorted(
        (d for d in os.listdir(candidate_dir) if os.path.isdir(os.path.join(candidate_dir, d))),
        reverse=True,
    )
    for d in subdirs:
        full = os.path.join(candidate_dir, d)
        if os.path.exists(os.path.join(full, "STATUS.txt")):
            return full
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
