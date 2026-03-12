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
import math
import os
import sys
import time as _time
from datetime import datetime, timezone

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


def _check_campaign_wall_time(spec, status, campaign_dir: str) -> bool:
    """Return True if campaign has exceeded max_campaign_hours. Marks all as failed."""
    if spec.max_campaign_hours <= 0:
        return False  # unlimited
    # Find earliest started_at across all stages
    earliest = None
    for stage in spec.stages:
        started = status.raw()["stages"].get(stage.id, {}).get("started_at")
        if started:
            try:
                dt = datetime.fromisoformat(started)
                if earliest is None or dt < earliest:
                    earliest = dt
            except (ValueError, TypeError):
                pass
    if earliest is None:
        return False
    elapsed_hours = (datetime.now(timezone.utc) - earliest).total_seconds() / 3600
    if elapsed_hours > spec.max_campaign_hours:
        print(
            f"[heartbeat] Campaign wall time exceeded: {elapsed_hours:.1f}h > "
            f"{spec.max_campaign_hours}h limit. Halting."
        )
        return True
    return False


def _load_idle_tick_count(campaign_dir: str) -> int:
    """Load the consecutive idle tick counter from a sidecar file."""
    path = os.path.join(campaign_dir, ".idle_ticks")
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return 0


def _save_idle_tick_count(campaign_dir: str, count: int) -> None:
    """Persist the consecutive idle tick counter."""
    path = os.path.join(campaign_dir, ".idle_ticks")
    with open(path, "w") as f:
        f.write(str(count))


def _check_repairing_timeout(stage, status, spec, campaign_dir: str) -> bool:
    """If a stage has been in REPAIRING status too long, time it out and mark failed."""
    sid = stage.id
    if status.stage_status(sid) != REPAIRING:
        return False
    stage_data = status.raw()["stages"].get(sid, {})
    # Use completed_at as the time repair started (it was set when the stage first failed)
    repair_started = stage_data.get("completed_at")
    if not repair_started:
        return False
    try:
        dt = datetime.fromisoformat(repair_started)
    except (ValueError, TypeError):
        return False
    elapsed = (datetime.now(timezone.utc) - dt).total_seconds()
    timeout = spec.repair.repairing_timeout_seconds
    if elapsed > timeout:
        print(
            f"[heartbeat] Stage '{sid}' stuck in REPAIRING for {elapsed:.0f}s "
            f"(timeout={timeout}s). Marking as failed."
        )
        status.mark_failed(sid, f"Repair timed out after {elapsed:.0f}s", stage_data.get("missing_artifacts", []))
        write_status(campaign_dir, status)
        return True
    return False


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
    # -1. Campaign wall-time check (P3-11)
    # ----------------------------------------------------------------
    if _check_campaign_wall_time(spec, status, campaign_dir):
        notify(
            f"Campaign '{spec.name}' exceeded wall time limit "
            f"({spec.max_campaign_hours}h). Halting.",
            spec.notification,
        )
        return 2

    # ----------------------------------------------------------------
    # 0. Check any stage with a SLURM repair job in progress
    # ----------------------------------------------------------------
    for stage in spec.stages:
        sid = stage.id
        if status.stage_status(sid) == REPAIRING:
            # Check REPAIRING timeout first (P0 fix)
            if _check_repairing_timeout(stage, status, spec, campaign_dir):
                continue  # now FAILED, will be handled below
            print(f"[heartbeat] Stage '{sid}' has a repair in progress — polling...")
            repair_exit = _try_repair(stage, spec, status, campaign_dir)
            if repair_exit is not None:
                _save_idle_tick_count(campaign_dir, 0)
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
            _save_idle_tick_count(campaign_dir, 0)
            return 1

        # Process is dead (or force_advance). Check artifacts.
        all_present, missing = check_stage_artifacts(workspace, stage)

        # P1-5: Validate artifact content, not just existence
        if all_present:
            validation_errors = _validate_artifact_content(workspace, stage)
            if validation_errors:
                all_present = False
                missing = validation_errors

        if all_present:
            print(f"[heartbeat] Stage '{sid}' completed.")
            status.mark_completed(sid)
            write_status(campaign_dir, status)
            distill_stage_memory(stage, workspace, campaign_dir)
            notify_stage_complete(sid, workspace, spec.notification)
        else:
            reason = f"Process ended but required artifacts missing/invalid: {missing}"
            print(f"[heartbeat] Stage '{sid}' FAILED: {reason}")
            status.mark_failed(sid, reason, missing)
            write_status(campaign_dir, status)

            # --- Autonomous repair attempt ---
            repair_exit = _try_repair(stage, spec, status, campaign_dir)
            if repair_exit is not None:
                _save_idle_tick_count(campaign_dir, 0)
                return repair_exit

            # No repair attempted or repair exhausted — escalate
            notify_stage_failed(sid, reason, spec.notification)
            return 2

        _save_idle_tick_count(campaign_dir, 0)
        break  # Only process one in-progress stage per tick

    # ----------------------------------------------------------------
    # 2. Check for overall failure (with repair opportunity)
    # ----------------------------------------------------------------
    if status.campaign_failed(spec):
        # Find the failed stage and try repair before giving up
        for stage in spec.stages:
            sid = stage.id
            if status.stage_status(sid) == FAILED:
                repair_exit = _try_repair(stage, spec, status, campaign_dir)
                if repair_exit is not None:
                    _save_idle_tick_count(campaign_dir, 0)
                    return repair_exit

                # P3-10: Escalation timeout — if repair is exhausted and
                # the failure has been sitting for longer than
                # escalation_timeout_minutes, auto-retry (reset attempts).
                repair_config = spec.repair
                if repair_config.auto_retry_on_timeout and repair_config.escalation_timeout_minutes > 0:
                    stage_data = status.raw()["stages"].get(sid, {})
                    failed_at = stage_data.get("completed_at")
                    if failed_at:
                        try:
                            failed_dt = datetime.fromisoformat(failed_at)
                            elapsed_min = (datetime.now(timezone.utc) - failed_dt).total_seconds() / 60
                            if elapsed_min >= repair_config.escalation_timeout_minutes:
                                print(
                                    f"[heartbeat] Escalation timeout for '{sid}': "
                                    f"{elapsed_min:.0f}min elapsed (limit={repair_config.escalation_timeout_minutes}min). "
                                    f"Auto-resetting repair attempts."
                                )
                                # Clear repair log to allow fresh attempts
                                stage_data["repair_log"] = []
                                write_status(campaign_dir, status)
                                # Try repair again with fresh budget
                                retry_exit = _try_repair(stage, spec, status, campaign_dir)
                                if retry_exit is not None:
                                    _save_idle_tick_count(campaign_dir, 0)
                                    return retry_exit
                        except (ValueError, TypeError):
                            pass

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
        _save_idle_tick_count(campaign_dir, 0)
        return 3

    # ----------------------------------------------------------------
    # 5. Idle tick circuit breaker (P0-1 / P3-12)
    # ----------------------------------------------------------------
    idle_ticks = _load_idle_tick_count(campaign_dir) + 1
    _save_idle_tick_count(campaign_dir, idle_ticks)

    if spec.max_idle_ticks > 0 and idle_ticks >= spec.max_idle_ticks:
        print(
            f"[heartbeat] {idle_ticks} consecutive idle ticks (limit={spec.max_idle_ticks}). "
            f"Campaign is stuck or done. Exiting with code 0 to stop cron."
        )
        return 0

    print(f"[heartbeat] No action taken — waiting for dependencies. (idle tick {idle_ticks}/{spec.max_idle_ticks})")
    return 1


def _validate_artifact_content(workspace: str, stage) -> list:
    """
    P1-5: Validate artifact content beyond mere file existence.

    Uses per-artifact validators from stage.artifact_validators, plus
    built-in heuristics for known file types.

    Returns a list of validation error strings (empty = all valid).
    """
    errors = []
    validators = getattr(stage, "artifact_validators", {})

    for artifact in stage.success_artifacts.get("required", []):
        if artifact.endswith("/"):
            continue  # skip directory artifacts
        full = os.path.join(workspace, artifact)
        if not os.path.exists(full):
            continue  # already caught by check_stage_artifacts

        # Check validator rules if defined
        rules = validators.get(artifact, {})

        try:
            file_size = os.path.getsize(full)
        except OSError:
            continue

        # min_size_bytes check
        min_size = rules.get("min_size_bytes", 0)
        if file_size < min_size:
            errors.append(f"{artifact}: too small ({file_size}B < {min_size}B)")
            continue

        # Content checks (for text files)
        if any(artifact.endswith(ext) for ext in (".json", ".md", ".txt", ".tex", ".csv")):
            try:
                with open(full) as f:
                    content = f.read(100_000)  # cap at 100KB
            except Exception:
                continue

            # must_contain checks
            for required_str in rules.get("must_contain", []):
                if required_str not in content:
                    errors.append(f"{artifact}: missing required content '{required_str}'")

            # must_not_contain checks (detect hollow artifacts)
            for forbidden_str in rules.get("must_not_contain", []):
                if forbidden_str in content:
                    errors.append(f"{artifact}: contains forbidden content '{forbidden_str}'")

            # Built-in heuristic: JSON files with "not_executed" are hollow
            if artifact.endswith(".json") and '"status": "not_executed"' in content:
                errors.append(f"{artifact}: hollow artifact (status=not_executed)")

    return errors


def _compute_backoff_seconds(repair_config, attempt_num: int) -> float:
    """P0-4: Compute exponential backoff for repair attempt N."""
    base = repair_config.backoff_base_seconds
    cap = repair_config.backoff_max_seconds
    delay = min(base * (2 ** (attempt_num - 1)), cap)
    return delay


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

    # P0-4: Exponential backoff — check if enough time has elapsed since last attempt
    if attempts_so_far > 0:
        last_attempt = status.raw()["stages"].get(sid, {}).get("repair_log", [])
        if last_attempt:
            last_ts = last_attempt[-1].get("timestamp", "")
            try:
                last_dt = datetime.fromisoformat(last_ts)
                elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
                required_backoff = _compute_backoff_seconds(repair_config, attempts_so_far)
                if elapsed < required_backoff:
                    remaining = required_backoff - elapsed
                    print(
                        f"[heartbeat] Backoff for '{sid}': {remaining:.0f}s remaining "
                        f"(need {required_backoff:.0f}s between attempts). Skipping this tick."
                    )
                    return 1  # come back next tick
            except (ValueError, TypeError):
                pass  # malformed timestamp, proceed anyway

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
            # Record submission failure as an attempt (P0-2: count all failures)
            status.add_repair_attempt(
                sid,
                success=False,
                diagnosis="SLURM repair submission failed",
                actions=[],
                duration=0.0,
                error="submit_slurm_repair returned None",
            )
            status.mark_failed(sid, "SLURM repair submission failed", status.raw()["stages"].get(sid, {}).get("missing_artifacts", []))
            write_status(campaign_dir, status)
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

    # P0-2: Always record the attempt, even if planning failed
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

        if attempt_num < repair_config.max_attempts:
            # P0-4: Backoff is enforced in _try_repair. Signal that we'll retry
            # on a future tick, but the backoff timer must elapse first.
            backoff = _compute_backoff_seconds(repair_config, attempt_num)
            print(
                f"[heartbeat] Will retry repair after {backoff:.0f}s backoff "
                f"({repair_config.max_attempts - attempt_num} attempt(s) remaining)."
            )
            return 1  # come back next tick (backoff enforced in _try_repair)

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
