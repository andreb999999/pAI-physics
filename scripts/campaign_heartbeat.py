#!/usr/bin/env python3
"""
Campaign heartbeat — OpenClaw's entrypoint for autonomous campaign management.

Call this script on a regular interval (every N minutes) to:
  1. Check whether the current in-progress stage has completed or failed.
  2. Advance the campaign to the next pending stage when ready.
  3. Distill completed stage artifacts into cross-run memory.
  4. Push a status summary notification.

Usage:
    python scripts/campaign_heartbeat.py --campaign campaign.yaml
    python scripts/campaign_heartbeat.py --campaign campaign.yaml --campaign-dir results/my_campaign
    python scripts/campaign_heartbeat.py --campaign campaign.yaml --force-advance
    python scripts/campaign_heartbeat.py --campaign campaign.yaml --status

Exit codes:
    0  — campaign is fully complete (all stages done)
    1  — campaign is in progress (normal heartbeat tick, nothing to do)
    2  — campaign has a failed stage (human attention required)
    3  — campaign was just advanced (new stage launched this tick)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure tex-live is available for paper/editorial stages
_TEX_LIVE_BIN = "/orcd/software/community/001/pkg/tex-live/20251104/bin/x86_64-linux"
if _TEX_LIVE_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _TEX_LIVE_BIN + ":" + os.environ.get("PATH", "")

# Ensure consortium is importable from repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from consortium.campaign.spec import load_spec
from consortium.campaign.status import (
    COMPLETED, FAILED, IN_PROGRESS, PENDING, REPAIRING,
    check_stage_artifacts, init_status, is_pid_alive,
    read_status, write_status,
)
from consortium.campaign.runner import launch_stage
from consortium.campaign.memory import distill_stage_memory
from consortium.campaign.notify import (
    notify, notify_campaign_complete, notify_heartbeat,
    notify_stage_complete, notify_stage_failed, notify_stage_launched,
    notify_repair_started, notify_repair_succeeded, notify_repair_failed,
)
from consortium.campaign.repair_agent import (
    attempt_repair, submit_slurm_repair, poll_slurm_repair,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Campaign heartbeat for consortium.")
    p.add_argument("--campaign", required=True, help="Path to campaign.yaml")
    p.add_argument(
        "--campaign-dir",
        help="Directory for campaign state files (default: workspace_root from spec).",
    )
    p.add_argument(
        "--force-advance",
        action="store_true",
        help="Force-advance to next stage even if no in-progress stage is detected.",
    )
    p.add_argument(
        "--status",
        action="store_true",
        help="Print current campaign status and exit.",
    )
    p.add_argument(
        "--init",
        action="store_true",
        help="Initialise campaign status file and exit (safe to re-run).",
    )
    return p.parse_args()


def print_status_report(spec, status, campaign_dir: str) -> None:
    print(f"\n{'='*60}")
    print(f"Campaign: {spec.name}")
    print(f"Status file: {os.path.join(campaign_dir, 'campaign_status.json')}")
    print(f"{'='*60}")
    for stage in spec.stages:
        sid = stage.id
        st = status.stage_status(sid)
        ws = status.stage_workspace(sid) or "(none)"
        pid = status.stage_pid(sid)
        pid_str = f" PID={pid}" if pid else ""
        missing = status.raw()["stages"].get(sid, {}).get("missing_artifacts", [])
        missing_str = f" MISSING={missing}" if missing else ""
        print(f"  [{st:12s}] {sid:30s} {ws}{pid_str}{missing_str}")
    print()


def run_heartbeat(
    spec,
    status,
    campaign_dir: str,
    force_advance: bool = False,
) -> int:
    """
    Core heartbeat logic. Returns exit code.
    """
    # ----------------------------------------------------------------
    # 0. Check any stage with a SLURM repair job in progress
    # ----------------------------------------------------------------
    for stage in spec.stages:
        sid = stage.id
        if status.stage_status(sid) == REPAIRING:
            print(f"[heartbeat] Stage '{sid}' has a repair in progress — polling...")
            repair_exit = _try_repair(stage, spec, status, campaign_dir)
            if repair_exit is not None:
                return repair_exit
            # If None, repair exhausted — fall through to normal failure handling

    # ----------------------------------------------------------------
    # 1. Check any in-progress stage
    # ----------------------------------------------------------------
    for stage in spec.stages:
        sid = stage.id
        if not status.is_in_progress(sid):
            continue

        pid = status.stage_pid(sid)
        workspace = status.stage_workspace(sid) or ""
        alive = is_pid_alive(pid) if pid else False

        if alive and not force_advance:
            # Stage still running — check artifacts speculatively
            all_present, missing = check_stage_artifacts(workspace, stage)
            summary = _build_summary(spec, status)
            notify_heartbeat(summary, spec.notification)
            if not all_present:
                print(
                    f"[heartbeat] Stage '{sid}' in progress (PID {pid}). "
                    f"Missing artifacts: {missing}"
                )
            else:
                print(f"[heartbeat] Stage '{sid}' in progress (PID {pid}). Artifacts look complete.")
            return 1

        # Process is dead (or force_advance). Check artifacts.
        all_present, missing = check_stage_artifacts(workspace, stage)

        if all_present:
            print(f"[heartbeat] Stage '{sid}' completed.")
            status.mark_completed(sid)
            write_status(campaign_dir, status)
            distill_stage_memory(stage, workspace, campaign_dir)
            notify_stage_complete(sid, workspace, spec.notification)
        else:
            reason = f"Process ended but required artifacts missing: {missing}"
            print(f"[heartbeat] Stage '{sid}' FAILED: {reason}")
            status.mark_failed(sid, reason, missing)
            write_status(campaign_dir, status)

            # --- Autonomous repair attempt ---
            repair_exit = _try_repair(stage, spec, status, campaign_dir)
            if repair_exit is not None:
                return repair_exit

            # No repair attempted or repair exhausted — escalate
            notify_stage_failed(sid, reason, spec.notification)
            return 2

        break  # Only process one in-progress stage per tick

    # ----------------------------------------------------------------
    # 2. Check for overall failure (with repair opportunity)
    # ----------------------------------------------------------------
    if status.campaign_failed(spec):
        # Find the failed stage and try repair before giving up
        for stage in spec.stages:
            if status.stage_status(stage.id) == FAILED:
                repair_exit = _try_repair(stage, spec, status, campaign_dir)
                if repair_exit is not None:
                    return repair_exit
                break  # only handle one failed stage per tick
        print("[heartbeat] Campaign has a failed stage. Human attention required.")
        return 2

    # ----------------------------------------------------------------
    # 3. Check if campaign is fully complete
    # ----------------------------------------------------------------
    if status.campaign_finished(spec):
        print(f"[heartbeat] Campaign '{spec.name}' is complete.")
        notify_campaign_complete(spec.name, spec.notification)
        _write_summary_md(spec, status, campaign_dir)
        return 0

    # ----------------------------------------------------------------
    # 4. Find next pending stage to launch
    # ----------------------------------------------------------------
    for stage in spec.stages:
        sid = stage.id
        if status.stage_status(sid) != PENDING:
            continue
        if not status.all_complete(stage.depends_on):
            continue

        # Ready to launch
        proc = launch_stage(stage, spec, status, campaign_dir)
        workspace = status.stage_workspace(stage.depends_on[0]) if stage.depends_on else \
            os.path.join(spec.workspace_root, stage.id)
        # Runner writes the actual workspace; read it back from the proc's args if needed.
        # For now, update status with the workspace the runner chose.
        from consortium.campaign.runner import build_stage_workspace
        workspace = build_stage_workspace(stage, spec, status)
        status.mark_in_progress(sid, workspace, proc.pid)
        write_status(campaign_dir, status)
        notify_stage_launched(sid, proc.pid, workspace, spec.notification)
        return 3

    # No stage to launch and no active stage — waiting for a dependency
    print("[heartbeat] No action taken — waiting for dependencies.")
    return 1


def _try_repair(
    stage,
    spec,
    status,
    campaign_dir: str,
) -> "int | None":
    """
    Attempt autonomous repair of a failed stage.

    Supports two launcher modes:
      - "local": runs Claude Code inline (blocks this tick, ~10min max)
      - "slurm": submits a SLURM job and returns immediately; the next
        heartbeat tick polls for the result.

    Returns:
        - 3 if the stage was repaired and relaunched (same as "stage launched")
        - 1 if repair is in progress (SLURM job running, or retrying next tick)
        - None if repair is not enabled, exhausted, or failed
    """
    import time as _time
    from consortium.campaign.status import is_slurm_job_alive

    sid = stage.id
    repair_config = spec.repair

    if not repair_config.enabled:
        return None

    # ------------------------------------------------------------------
    # SLURM mode: check if a repair job is already running
    # ------------------------------------------------------------------
    if repair_config.launcher == "slurm":
        stage_data = status.raw()["stages"].get(sid, {})
        repair_slurm_jid = stage_data.get("repair_slurm_job_id")

        if repair_slurm_jid and status.stage_status(sid) == REPAIRING:
            # A repair job was previously submitted — poll for result
            if is_slurm_job_alive(repair_slurm_jid):
                print(
                    f"[heartbeat] Repair SLURM job {repair_slurm_jid} for '{sid}' "
                    f"still running. Will check next tick."
                )
                return 1  # come back later

            # Job finished — poll the sentinel
            result = poll_slurm_repair(stage, status, campaign_dir)
            if result is None:
                print(
                    f"[heartbeat] Repair SLURM job {repair_slurm_jid} for '{sid}' "
                    f"finished but no sentinel found. Treating as failure."
                )
                result_success = False
                result_diagnosis = "SLURM repair job completed but produced no result sentinel."
                result_actions = []
                result_duration = 0.0
                result_error = "No sentinel file"
            else:
                result_success = result.success
                result_diagnosis = result.diagnosis
                result_actions = result.actions_taken
                result_duration = result.duration_seconds
                result_error = result.error

            # Record the attempt
            status.add_repair_attempt(
                sid,
                success=result_success,
                diagnosis=result_diagnosis,
                actions=result_actions,
                duration=result_duration,
                error=result_error,
            )

            return _handle_repair_result(
                sid, result_success, result_diagnosis, result_actions,
                result_error, status, spec, stage, campaign_dir,
            )

    # ------------------------------------------------------------------
    # Check attempt budget
    # ------------------------------------------------------------------
    attempts_so_far = status.repair_attempt_count(sid)
    if attempts_so_far >= repair_config.max_attempts:
        print(
            f"[heartbeat] Repair attempts exhausted for '{sid}' "
            f"({attempts_so_far}/{repair_config.max_attempts})."
        )
        return None

    attempt_num = attempts_so_far + 1
    print(
        f"[heartbeat] Attempting autonomous repair for '{sid}' "
        f"(attempt {attempt_num}/{repair_config.max_attempts})..."
    )

    # Notify
    notify_repair_started(sid, attempt_num, repair_config.max_attempts, spec.notification)

    # Mark as repairing
    status.mark_repairing(sid)
    write_status(campaign_dir, status)

    # ------------------------------------------------------------------
    # SLURM mode: submit job and return immediately
    # ------------------------------------------------------------------
    if repair_config.launcher == "slurm":
        job_id = submit_slurm_repair(stage, spec, status, campaign_dir)
        if job_id is None:
            print(f"[heartbeat] SLURM repair submission failed for '{sid}'. Falling back to local.")
            # Fall through to local mode
        else:
            # Store SLURM job ID in status so we can poll it next tick
            status.raw()["stages"][sid]["repair_slurm_job_id"] = job_id
            write_status(campaign_dir, status)
            print(
                f"[heartbeat] Repair SLURM job {job_id} submitted for '{sid}'. "
                f"Will poll on next heartbeat tick."
            )
            return 1  # in progress — come back next tick

    # ------------------------------------------------------------------
    # Local mode: run Claude Code inline (blocking)
    # ------------------------------------------------------------------
    result = attempt_repair(stage, spec, status, campaign_dir)

    # Record the attempt
    status.add_repair_attempt(
        sid,
        success=result.success,
        diagnosis=result.diagnosis,
        actions=result.actions_taken,
        duration=result.duration_seconds,
        error=result.error,
    )

    return _handle_repair_result(
        sid, result.success, result.diagnosis, result.actions_taken,
        result.error, status, spec, stage, campaign_dir,
    )


def _handle_repair_result(
    sid: str,
    success: bool,
    diagnosis: str,
    actions: list,
    error: "str | None",
    status,
    spec,
    stage,
    campaign_dir: str,
) -> "int | None":
    """
    Common handler for repair results (both local and SLURM modes).

    Returns exit code for the heartbeat.
    """
    import time as _time

    repair_config = spec.repair
    attempt_num = status.repair_attempt_count(sid)  # already incremented

    if success:
        print(f"[heartbeat] Repair succeeded for '{sid}'. Relaunching stage.")
        notify_repair_succeeded(sid, attempt_num, diagnosis, spec.notification)

        # Reset to pending and relaunch
        status.mark_pending_retry(sid)
        write_status(campaign_dir, status)

        # Brief pause before retry
        if repair_config.retry_delay_seconds > 0:
            _time.sleep(repair_config.retry_delay_seconds)

        # Relaunch the stage
        proc = launch_stage(stage, spec, status, campaign_dir)
        from consortium.campaign.runner import build_stage_workspace
        workspace = build_stage_workspace(stage, spec, status)
        status.mark_in_progress(sid, workspace, proc.pid)
        write_status(campaign_dir, status)
        notify_stage_launched(sid, proc.pid, workspace, spec.notification)
        return 3  # stage relaunched

    else:
        print(f"[heartbeat] Repair failed for '{sid}': {error or diagnosis[:200]}")
        notify_repair_failed(
            sid, attempt_num, repair_config.max_attempts,
            error or diagnosis[:150],
            spec.notification,
        )

        # Mark back as failed (repair didn't help)
        stage_data = status.raw()["stages"].get(sid, {})
        status.mark_failed(
            sid,
            stage_data.get("fail_reason", "Repair failed"),
            stage_data.get("missing_artifacts", []),
        )
        write_status(campaign_dir, status)

        # If we still have attempts left, return 1 so the next heartbeat
        # tick can try again
        if attempt_num < repair_config.max_attempts:
            print(
                f"[heartbeat] Will retry repair on next heartbeat tick "
                f"({repair_config.max_attempts - attempt_num} attempt(s) remaining)."
            )
            return 1  # in progress — come back next tick

        return None  # exhausted


def _build_summary(spec, status) -> str:
    lines = [f"Campaign '{spec.name}' heartbeat:"]
    for stage in spec.stages:
        sid = stage.id
        st = status.stage_status(sid)
        pid = status.stage_pid(sid)
        pid_str = f" (PID {pid})" if pid and st == IN_PROGRESS else ""
        lines.append(f"  {sid}: {st}{pid_str}")
    return " | ".join(lines[1:])


def _write_summary_md(spec, status, campaign_dir: str) -> None:
    path = os.path.join(campaign_dir, "CAMPAIGN_COMPLETE.md")
    lines = [f"# Campaign Complete: {spec.name}\n"]
    for stage in spec.stages:
        sid = stage.id
        ws = status.stage_workspace(sid) or "(unknown)"
        completed_at = status.raw()["stages"].get(sid, {}).get("completed_at", "")
        lines.append(f"- **{sid}**: {ws} (completed {completed_at})")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[heartbeat] Written: {path}")


def main() -> int:
    args = parse_args()

    spec = load_spec(args.campaign)
    campaign_dir = args.campaign_dir or spec.workspace_root
    os.makedirs(campaign_dir, exist_ok=True)

    if args.init:
        init_status(campaign_dir, spec, args.campaign)
        print(f"[heartbeat] Campaign status initialised at {campaign_dir}/campaign_status.json")
        return 0

    # Ensure status file exists with all stage entries
    status = init_status(campaign_dir, spec, args.campaign)

    if args.status:
        print_status_report(spec, status, campaign_dir)
        return 0

    return run_heartbeat(spec, status, campaign_dir, force_advance=args.force_advance)


if __name__ == "__main__":
    sys.exit(main())
