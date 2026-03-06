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

# Ensure consortium is importable from repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from consortium.campaign.spec import load_spec
from consortium.campaign.status import (
    COMPLETED, FAILED, IN_PROGRESS, PENDING,
    check_stage_artifacts, init_status, is_pid_alive,
    read_status, write_status,
)
from consortium.campaign.runner import launch_stage
from consortium.campaign.memory import distill_stage_memory
from consortium.campaign.notify import (
    notify, notify_campaign_complete, notify_heartbeat,
    notify_stage_complete, notify_stage_failed, notify_stage_launched,
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
            notify_stage_failed(sid, reason, spec.notification)
            return 2

        break  # Only process one in-progress stage per tick

    # ----------------------------------------------------------------
    # 2. Check for overall failure
    # ----------------------------------------------------------------
    if status.campaign_failed(spec):
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
