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
      "slurm_job_id": int | null,
      "attempt_id": int | null,
      "stdout_log": str | null,
      "stderr_log": str | null,
      "started_at": ISO str | null,
      "completed_at": ISO str | null,
      "missing_artifacts": [str],
      "fail_reason": str | null
    }
  }
}
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import signal
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

try:
    import filelock as _filelock_module
    _FILELOCK_AVAILABLE = True
except ImportError:
    _FILELOCK_AVAILABLE = False

from .spec import CampaignSpec, Stage

_logger = logging.getLogger(__name__)

STATUS_FILE = "campaign_status.json"
_LOCK_TIMEOUT = 30  # seconds


@contextmanager
def _status_lock(campaign_dir: str):
    """Context manager that holds an exclusive file lock on the status file.

    Uses filelock if available, otherwise falls back to fcntl.flock() on POSIX.
    """
    lock_path = os.path.join(campaign_dir, "campaign_status.lock")
    if _FILELOCK_AVAILABLE:
        lock = _filelock_module.FileLock(lock_path, timeout=_LOCK_TIMEOUT)
        with lock:
            yield
    else:
        # Fallback: use fcntl.flock() (available on all POSIX systems)
        _logger.warning(
            "filelock not installed — using fcntl.flock() fallback. "
            "Install filelock for better cross-platform locking: pip install filelock"
        )
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

# Stage status constants
PENDING = "pending"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"
FAILED = "failed"
REPAIRING = "repairing"
PLANNING = "planning"  # Waiting for human review of dynamic campaign plan


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

    def stage_slurm_job_id(self, stage_id: str) -> Optional[int]:
        jid = self._d["stages"].get(stage_id, {}).get("slurm_job_id")
        return int(jid) if jid is not None else None

    def stage_attempt_id(self, stage_id: str) -> Optional[int]:
        attempt_id = self._d["stages"].get(stage_id, {}).get("attempt_id")
        return int(attempt_id) if attempt_id is not None else None

    def stage_stdout_log(self, stage_id: str) -> Optional[str]:
        return self._d["stages"].get(stage_id, {}).get("stdout_log")

    def stage_stderr_log(self, stage_id: str) -> Optional[str]:
        return self._d["stages"].get(stage_id, {}).get("stderr_log")

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

    def mark_in_progress(
        self, stage_id: str, workspace: str, pid: int,
        slurm_job_id: Optional[int] = None,
        attempt_id: Optional[int] = None,
        stdout_log: Optional[str] = None,
        stderr_log: Optional[str] = None,
    ) -> "CampaignStatus":
        self._d["stages"].setdefault(stage_id, {})
        self._d["stages"][stage_id].update({
            "status": IN_PROGRESS,
            "workspace": workspace,
            "pid": pid,
            "slurm_job_id": slurm_job_id,
            "attempt_id": attempt_id,
            "stdout_log": stdout_log,
            "stderr_log": stderr_log,
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

    def mark_repairing(self, stage_id: str) -> "CampaignStatus":
        """Transition a failed stage into 'repairing' status."""
        self._d["stages"].setdefault(stage_id, {})
        self._d["stages"][stage_id]["status"] = REPAIRING
        return self

    def mark_pending_retry(self, stage_id: str) -> "CampaignStatus":
        """Reset a repaired stage to pending so it can be relaunched."""
        self._d["stages"].setdefault(stage_id, {})
        self._d["stages"][stage_id].update({
            "status": PENDING,
            "pid": None,
            "slurm_job_id": None,
            "stdout_log": None,
            "stderr_log": None,
            "completed_at": None,
            "missing_artifacts": [],
            "fail_reason": None,
        })
        return self

    # ------------------------------------------------------------------
    # Repair tracking
    # ------------------------------------------------------------------

    def repair_attempt_count(self, stage_id: str) -> int:
        """Number of repair attempts already made for this stage."""
        return len(self._d["stages"].get(stage_id, {}).get("repair_log", []))

    def add_repair_attempt(
        self,
        stage_id: str,
        success: bool,
        diagnosis: str,
        actions: List[str],
        duration: float,
        error: Optional[str] = None,
    ) -> "CampaignStatus":
        """Record a repair attempt in the stage's repair_log."""
        self._d["stages"].setdefault(stage_id, {})
        stage = self._d["stages"][stage_id]
        stage.setdefault("repair_log", [])
        stage["repair_log"].append({
            "attempt": len(stage["repair_log"]) + 1,
            "timestamp": _now(),
            "success": success,
            "diagnosis": diagnosis[:500],
            "actions": actions[:20],
            "duration_seconds": round(duration, 1),
            "error": error,
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
        # Preserve ALL existing stage entries (including dynamically injected
        # stages not present in the current spec).  This fixes the critical
        # bug where init_status() was called every heartbeat tick with a
        # freshly-loaded spec that only contained the YAML-defined stages,
        # silently dropping dynamically generated stages from status tracking.
        spec_ids = {s.id for s in spec.stages}
        for sid, sdata in existing_stages.items():
            if sid not in spec_ids:
                data["stages"][sid] = sdata
        for stage in spec.stages:
            existing_stage = existing_stages.get(stage.id, {})
            data["stages"][stage.id] = existing_stage or {
                "status": PENDING,
                "workspace": None,
                "pid": None,
                "slurm_job_id": None,
                "attempt_id": None,
                "stdout_log": None,
                "stderr_log": None,
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


def is_slurm_job_alive(job_id: int) -> Optional[bool]:
    """Check if a SLURM job is still active (PENDING, RUNNING, etc.).

    Returns:
        True  — job is confirmed alive (PENDING/RUNNING/etc.)
        False — job is confirmed dead (not in squeue, exit code available)
        None  — unknown (squeue timed out or unavailable; don't assume dead)

    Uses squeue for fast lookup — only lists active jobs.
    Works across nodes (unlike PID-based checking).
    """
    if not job_id or job_id <= 0:
        return False
    try:
        result = subprocess.run(
            ["squeue", "-j", str(job_id), "--noheader", "--format=%T"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        state = result.stdout.strip()
        if state in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED"):
            return True
        # Empty output or terminal state = job finished
        return False
    except subprocess.TimeoutExpired:
        _logger.warning("squeue timed out checking job %d — treating as unknown", job_id)
        return None
    except FileNotFoundError:
        _logger.warning("squeue not found — treating SLURM job %d as unknown", job_id)
        return None
    except Exception as exc:
        _logger.warning("squeue failed for job %d (%s) — treating as unknown", job_id, exc)
        return None


def is_stage_alive(status: "CampaignStatus", stage_id: str) -> Optional[bool]:
    """Check if a stage is still running, using SLURM job ID or PID.

    Returns True (alive), False (dead), or None (unknown — skip this tick).
    Prefers SLURM job ID check (works across nodes) over PID check (local only).
    """
    slurm_jid = status.stage_slurm_job_id(stage_id)
    if slurm_jid:
        return is_slurm_job_alive(slurm_jid)
    pid = status.stage_pid(stage_id)
    if pid:
        return is_pid_alive(pid)
    return False


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


def clean_orphaned_stages(campaign_dir: str, status: CampaignStatus) -> list[str]:
    """Remove status entries whose workspace directory does not exist.

    Prevents stale entries from crashed/aborted campaigns from accumulating.
    Only removes stages that are in 'pending' or 'failed' state and whose
    workspace was set but no longer exists on disk.

    Returns list of removed stage IDs.
    """
    removed = []
    stages = status.raw().get("stages", {})
    for sid, sdata in list(stages.items()):
        workspace = sdata.get("workspace")
        if not workspace:
            continue  # no workspace assigned yet — not orphaned
        st = sdata.get("status", PENDING)
        if st in (IN_PROGRESS, REPAIRING, COMPLETED):
            continue  # don't touch active or completed stages
        full_path = os.path.join(campaign_dir, workspace) if not os.path.isabs(workspace) else workspace
        if not os.path.isdir(full_path):
            _logger.info("Removing orphaned stage '%s' (workspace %s missing)", sid, workspace)
            del stages[sid]
            removed.append(sid)
    return removed


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
