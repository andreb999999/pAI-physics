#!/usr/bin/env python3
"""
Campaign monitor — unified CLI for checking status, outputs, logs, and steering.

Subcommands:
    dashboard   Rich overview of campaign status, budget, tokens, recent outputs
    outputs     List and preview key outputs from a stage
    logs        Tail recent log lines from a stage
    steer       Send steering commands to a running stage's HTTP API

Usage:
    python scripts/campaign_monitor.py --campaign campaign.yaml dashboard
    python scripts/campaign_monitor.py --campaign campaign.yaml outputs theory
    python scripts/campaign_monitor.py --campaign campaign.yaml logs theory -f
    python scripts/campaign_monitor.py --campaign campaign.yaml steer status
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Ensure consortium is importable from repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from consortium.campaign.spec import CampaignSpec, Stage, load_spec
from consortium.campaign.status import (
    COMPLETED, FAILED, IN_PROGRESS, PENDING,
    init_status, is_stage_alive, read_status,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_HTTP_PORT = 5002


def _read_budget(workspace: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Read total spend and limit from budget_state.json."""
    if not workspace:
        return None, None
    path = os.path.join(workspace, "budget_state.json")
    if not os.path.exists(path):
        return None, None
    try:
        with open(path) as f:
            bd = json.load(f)
        total = bd.get("total_cost_usd", bd.get("total_spent_usd"))
        limit = bd.get("usd_limit")
        if total is not None:
            total = float(total)
        if limit is not None:
            limit = float(limit)
        return total, limit
    except Exception:
        return None, None


def _read_token_usage(workspace: Optional[str]) -> Optional[Dict[str, Any]]:
    """Read token usage from run_token_usage.json."""
    if not workspace:
        return None
    path = os.path.join(workspace, "run_token_usage.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            usage = json.load(f)
        return {
            "prompt": usage.get("total_prompt_tokens", 0),
            "completion": usage.get("total_completion_tokens", 0),
        }
    except Exception:
        return None


def _format_tokens(n: int) -> str:
    """Format token count: 1234567 -> '1.2M', 45600 -> '45.6K'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _compute_elapsed(started_at: Optional[str]) -> Optional[str]:
    """Compute elapsed time string from an ISO timestamp to now."""
    if not started_at:
        return None
    try:
        start = datetime.fromisoformat(started_at)
        now = datetime.now(timezone.utc)
        delta = now - start
        total_secs = int(delta.total_seconds())
        if total_secs < 0:
            return "0m"
        hours, remainder = divmod(total_secs, 3600)
        minutes, _ = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes:02d}m"
        return f"{minutes}m"
    except Exception:
        return None


def _list_recent_files(workspace: Optional[str], n: int = 5) -> List[Dict[str, Any]]:
    """List the N most recently modified files in a workspace."""
    if not workspace or not os.path.isdir(workspace):
        return []
    entries = []
    for root, _dirs, files in os.walk(workspace):
        # Skip hidden dirs and __pycache__
        _dirs[:] = [d for d in _dirs if not d.startswith(".") and d != "__pycache__"]
        for fname in files:
            if fname.startswith("."):
                continue
            fpath = os.path.join(root, fname)
            try:
                st = os.stat(fpath)
                entries.append({
                    "path": os.path.relpath(fpath, workspace),
                    "size": st.st_size,
                    "mtime": st.st_mtime,
                })
            except OSError:
                continue
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return entries[:n]


def _format_size(nbytes: int) -> str:
    """Format file size: 12345 -> '12.1 KB'."""
    if nbytes >= 1_048_576:
        return f"{nbytes / 1_048_576:.1f} MB"
    if nbytes >= 1024:
        return f"{nbytes / 1024:.1f} KB"
    return f"{nbytes} B"


def _format_age(mtime: float) -> str:
    """Format file age: seconds since mtime -> '3m ago'."""
    delta = int(datetime.now().timestamp() - mtime)
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86400}d ago"


def _read_excerpt(path: str, max_chars: int = 500) -> str:
    """Read a file and return a truncated excerpt."""
    try:
        with open(path) as f:
            raw = f.read()
    except Exception as e:
        return f"[Could not read: {e}]"

    if path.endswith(".json"):
        try:
            parsed = json.loads(raw)
            raw = json.dumps(parsed, indent=2)
        except Exception:
            pass

    if len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + f"\n... [truncated, {len(raw):,} chars total]"


def _resolve_steering_host_port(
    status, spec: CampaignSpec, port_override: Optional[int], host_override: Optional[str],
) -> Tuple[str, int]:
    """Determine the HTTP steering host and port for the active stage."""
    host = host_override or "127.0.0.1"
    port = port_override or _DEFAULT_HTTP_PORT

    if port_override:
        return host, port

    # Try to detect callback_port from the active stage's args
    for stage in spec.stages:
        if status.stage_status(stage.id) != IN_PROGRESS:
            continue
        for i, arg in enumerate(stage.args):
            if arg == "--callback_port" and i + 1 < len(stage.args):
                try:
                    port = int(stage.args[i + 1]) + 1
                except ValueError:
                    pass

        # Try SLURM node detection for cross-node steering
        if not host_override:
            slurm_jid = status.stage_slurm_job_id(stage.id)
            if slurm_jid:
                try:
                    result = subprocess.run(
                        ["squeue", "-j", str(slurm_jid), "--noheader", "--format=%N"],
                        capture_output=True, text=True, timeout=10,
                    )
                    node = result.stdout.strip()
                    if node:
                        host = node
                except Exception:
                    pass
        break

    return host, port


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_dashboard(spec: CampaignSpec, status, campaign_dir: str, as_json: bool = False) -> None:
    """Rich overview of campaign status, budget, tokens, and recent outputs."""

    rows = []
    active_stage_id = None

    for stage in spec.stages:
        sid = stage.id
        raw = status.raw()["stages"].get(sid, {})
        st = status.stage_status(sid)
        workspace = status.stage_workspace(sid)
        alive = is_stage_alive(status, sid) if st == IN_PROGRESS else False

        if st == IN_PROGRESS:
            active_stage_id = sid

        elapsed = _compute_elapsed(raw.get("started_at"))
        budget_usd, budget_limit = _read_budget(workspace)
        tokens = _read_token_usage(workspace)

        rows.append({
            "stage": sid,
            "status": st,
            "alive": alive,
            "elapsed": elapsed,
            "budget_usd": budget_usd,
            "budget_limit": budget_limit,
            "tokens": tokens,
            "workspace": workspace,
            "pid": status.stage_pid(sid),
            "slurm_job_id": status.stage_slurm_job_id(sid),
            "started_at": raw.get("started_at"),
            "completed_at": raw.get("completed_at"),
            "fail_reason": raw.get("fail_reason"),
            "missing_artifacts": raw.get("missing_artifacts", []),
        })

    if as_json:
        print(json.dumps(rows, indent=2, default=str))
        return

    # Header
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"Campaign: {spec.name}")
    print(f"Dir: {campaign_dir}")
    print(sep)

    # Stage table
    print("\nSTAGES")
    hdr = f"  {'Stage':<18s} {'Status':<14s} {'Alive':<7s} {'Elapsed':<11s} {'Budget':<16s} {'Tokens (in/out)'}"
    print(hdr)
    print(f"  {'-'*18} {'-'*14} {'-'*7} {'-'*11} {'-'*16} {'-'*18}")

    for r in rows:
        alive_str = "YES" if r["alive"] else "-"
        elapsed_str = r["elapsed"] or "-"

        if r["budget_usd"] is not None:
            budget_str = f"${r['budget_usd']:.2f}"
            if r["budget_limit"] is not None:
                budget_str += f"/${r['budget_limit']:.0f}"
        else:
            budget_str = "-"

        if r["tokens"]:
            tok_str = f"{_format_tokens(r['tokens']['prompt'])} / {_format_tokens(r['tokens']['completion'])}"
        else:
            tok_str = "-"

        status_display = r["status"]
        if r["status"] == FAILED:
            status_display = "FAILED"

        print(f"  {r['stage']:<18s} {status_display:<14s} {alive_str:<7s} {elapsed_str:<11s} {budget_str:<16s} {tok_str}")

    # Failed stage details
    for r in rows:
        if r["status"] == FAILED:
            print(f"\n  FAILURE [{r['stage']}]: {r['fail_reason'] or 'unknown'}")
            if r["missing_artifacts"]:
                print(f"    Missing: {', '.join(r['missing_artifacts'])}")

    # Dependencies
    print("\nDEPENDENCIES")
    dep_parts = []
    for stage in spec.stages:
        if stage.depends_on:
            deps = ", ".join(stage.depends_on)
            dep_parts.append(f"  {stage.id} <- [{deps}]")
        else:
            dep_parts.append(f"  {stage.id} (no dependencies)")
    print("\n".join(dep_parts))

    # Recent outputs for active or most recently completed stage
    show_stage = active_stage_id
    if not show_stage:
        for r in reversed(rows):
            if r["status"] == COMPLETED and r["workspace"]:
                show_stage = r["stage"]
                break

    if show_stage:
        ws = status.stage_workspace(show_stage)
        recent = _list_recent_files(ws, n=5)
        st_label = status.stage_status(show_stage)
        print(f"\nRECENT OUTPUTS ({show_stage} -- {st_label})")
        if recent:
            for entry in recent:
                print(f"  {entry['path']:<40s} [{_format_size(entry['size'])}, {_format_age(entry['mtime'])}]")
        else:
            print("  (no files found)")

    # HTTP steering status (best-effort)
    for stage in spec.stages:
        if status.stage_status(stage.id) == IN_PROGRESS:
            host, port = _resolve_steering_host_port(status, spec, None, None)
            try:
                import urllib.request
                req = urllib.request.Request(f"http://{host}:{port}/status", method="GET")
                with urllib.request.urlopen(req, timeout=3) as resp:
                    data = json.loads(resp.read())
                print(f"\nHTTP STEERING: {host}:{port} -- paused={data.get('paused')}, queue_depth={data.get('queue_depth')}")
            except Exception:
                print(f"\nHTTP STEERING: {host}:{port} -- (not reachable)")
            break

    print(sep)
    print()


def cmd_outputs(
    spec: CampaignSpec, status, campaign_dir: str,
    stage_id: str, preview_chars: int = 500, artifacts_only: bool = False,
) -> None:
    """List and preview key outputs from a stage."""
    stage = spec.stage(stage_id)
    if not stage:
        print(f"Error: unknown stage '{stage_id}'. Valid: {spec.stage_ids()}")
        sys.exit(1)

    workspace = status.stage_workspace(stage_id)
    if not workspace or not os.path.isdir(workspace):
        print(f"Stage '{stage_id}' has no workspace yet (status: {status.stage_status(stage_id)}).")
        return

    st = status.stage_status(stage_id)
    print(f"\n{'='*70}")
    print(f"Stage: {stage_id}  |  Status: {st}  |  Workspace: {workspace}")
    print(f"{'='*70}")

    # Declared artifacts
    required = stage.success_artifacts.get("required", [])
    optional = stage.success_artifacts.get("optional", [])

    if required or optional:
        print("\nDECLARED ARTIFACTS")
        for rel in required:
            full = os.path.join(workspace, rel.rstrip("/"))
            exists = os.path.exists(full)
            marker = "OK" if exists else "MISSING"
            print(f"  [required] [{marker:7s}] {rel}")
            if exists and os.path.isfile(full):
                excerpt = _read_excerpt(full, max_chars=preview_chars)
                for line in excerpt.splitlines()[:10]:
                    print(f"    | {line}")
                lines_total = excerpt.count("\n") + 1
                if lines_total > 10:
                    print(f"    | ... ({lines_total - 10} more lines)")
                print()

        for rel in optional:
            full = os.path.join(workspace, rel.rstrip("/"))
            exists = os.path.exists(full)
            marker = "OK" if exists else "absent"
            print(f"  [optional] [{marker:7s}] {rel}")

    if artifacts_only:
        print()
        return

    # All files sorted by recency
    recent = _list_recent_files(workspace, n=20)
    print(f"\nALL FILES (20 most recent)")
    if recent:
        for entry in recent:
            print(f"  {entry['path']:<50s} [{_format_size(entry['size']):>8s}, {_format_age(entry['mtime'])}]")
    else:
        print("  (no files found)")

    # Budget and token summary
    budget_usd, budget_limit = _read_budget(workspace)
    tokens = _read_token_usage(workspace)
    if budget_usd is not None or tokens:
        print("\nRUN STATS")
        if budget_usd is not None:
            limit_str = f" / ${budget_limit:.0f} limit" if budget_limit else ""
            print(f"  Budget: ${budget_usd:.2f}{limit_str}")
            lock_path = os.path.join(workspace, "budget.lock")
            if os.path.exists(lock_path):
                print(f"  WARNING: budget.lock exists — stage hit cost cap")
        if tokens:
            print(f"  Tokens: {_format_tokens(tokens['prompt'])} prompt / {_format_tokens(tokens['completion'])} completion")

    print()


def cmd_logs(
    campaign_dir: str, stage_id: str,
    lines: int = 50, stderr: bool = False, follow: bool = False,
) -> None:
    """Tail recent log lines from a stage."""
    stream = "stderr" if stderr else "stdout"
    log_path = os.path.join(campaign_dir, "logs", f"{stage_id}_{stream}.log")

    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        # Check if any logs exist for this stage
        log_dir = os.path.join(campaign_dir, "logs")
        if os.path.isdir(log_dir):
            stage_logs = [f for f in os.listdir(log_dir) if f.startswith(stage_id)]
            if stage_logs:
                print(f"Available logs for '{stage_id}': {', '.join(stage_logs)}")
            else:
                print(f"No logs found for stage '{stage_id}'.")
        return

    if follow:
        try:
            subprocess.run(["tail", "-f", "-n", str(lines), log_path])
        except KeyboardInterrupt:
            pass
        return

    # Read last N lines
    try:
        with open(log_path) as f:
            all_lines = f.readlines()
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return

    tail = all_lines[-lines:]
    print(f"--- {log_path} (last {len(tail)} of {len(all_lines)} lines) ---")
    for line in tail:
        print(line, end="")
    print(f"\n--- end ---")


def cmd_steer(
    spec: CampaignSpec, status, campaign_dir: str,
    action: str, text: Optional[str] = None, instr_type: str = "m",
    port_override: Optional[int] = None, host_override: Optional[str] = None,
) -> None:
    """Send steering commands to a running stage's HTTP API."""
    import urllib.request
    import urllib.error

    host, port = _resolve_steering_host_port(status, spec, port_override, host_override)
    base = f"http://{host}:{port}"

    # Check if any stage is actually running
    active = [s.id for s in spec.stages if status.stage_status(s.id) == IN_PROGRESS]
    if not active:
        print("No stage is currently in_progress.")
        return

    print(f"Target: {base}  |  Active stage(s): {', '.join(active)}")

    try:
        if action == "status":
            req = urllib.request.Request(f"{base}/status", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            print(f"  paused: {data.get('paused')}")
            print(f"  queue_depth: {data.get('queue_depth')}")

        elif action == "pause":
            req = urllib.request.Request(f"{base}/interrupt", method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            print(f"  Interrupt sent: {data}")

        elif action == "instruct":
            if not text:
                print("Error: --text is required for 'instruct' action.")
                sys.exit(1)
            payload = json.dumps({"text": text, "type": instr_type}).encode()
            req = urllib.request.Request(
                f"{base}/instruction",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            print(f"  Instruction sent (type={instr_type}): {data}")

    except urllib.error.URLError as e:
        print(f"  Could not reach {base}: {e.reason}")
        print("  The stage may not be running, or the HTTP steering server is not active.")
        slurm_jid = None
        for sid in active:
            slurm_jid = status.stage_slurm_job_id(sid)
            if slurm_jid:
                break
        if slurm_jid and host == "127.0.0.1":
            print(f"  Hint: stage is running as SLURM job {slurm_jid}.")
            print(f"  Try: squeue -j {slurm_jid} --format=%N  to find the compute node,")
            print(f"  then: python scripts/campaign_monitor.py ... steer --host <node> status")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Campaign monitor — unified status, outputs, logs, and steering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --campaign campaign.yaml dashboard\n"
            "  %(prog)s --campaign campaign.yaml dashboard --json\n"
            "  %(prog)s --campaign campaign.yaml outputs theory\n"
            "  %(prog)s --campaign campaign.yaml outputs theory --artifacts-only\n"
            "  %(prog)s --campaign campaign.yaml logs experiments -f\n"
            "  %(prog)s --campaign campaign.yaml logs theory --stderr -n 100\n"
            "  %(prog)s --campaign campaign.yaml steer status\n"
            "  %(prog)s --campaign campaign.yaml steer pause\n"
            "  %(prog)s --campaign campaign.yaml steer instruct --text 'Focus on linear case'\n"
        ),
    )
    p.add_argument("--campaign", required=True, help="Path to campaign.yaml")
    p.add_argument("--campaign-dir", help="Override campaign directory (default: workspace_root from spec)")

    sub = p.add_subparsers(dest="command", required=True)

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Rich overview of campaign status, budget, tokens, recent outputs")
    p_dash.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")

    # outputs
    p_out = sub.add_parser("outputs", help="List and preview key outputs from a stage")
    p_out.add_argument("stage_id", help="Stage ID (e.g., 'theory', 'experiments', 'paper')")
    p_out.add_argument("--preview-chars", type=int, default=500, help="Max chars per file preview (default: 500)")
    p_out.add_argument("--artifacts-only", action="store_true", help="Show only declared success artifacts")

    # logs
    p_log = sub.add_parser("logs", help="Tail recent log lines from a stage")
    p_log.add_argument("stage_id", help="Stage ID")
    p_log.add_argument("-n", "--lines", type=int, default=50, help="Number of lines (default: 50)")
    p_log.add_argument("--stderr", action="store_true", help="Show stderr instead of stdout")
    p_log.add_argument("-f", "--follow", action="store_true", help="Follow mode (like tail -f)")

    # steer
    p_steer = sub.add_parser("steer", help="Send steering commands to a running stage")
    p_steer.add_argument("action", choices=["status", "pause", "instruct"], help="Steering action")
    p_steer.add_argument("--text", help="Instruction text (required for 'instruct')")
    p_steer.add_argument("--type", choices=["m", "n"], default="m", dest="instr_type",
                          help="Instruction type: m=modify current task, n=new task (default: m)")
    p_steer.add_argument("--port", type=int, default=None, help="Override HTTP steering port (default: 5002)")
    p_steer.add_argument("--host", default=None, help="Override HTTP steering host (default: 127.0.0.1)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    spec = load_spec(args.campaign)
    campaign_dir = args.campaign_dir or spec.workspace_root
    os.makedirs(campaign_dir, exist_ok=True)

    # Read or init status
    status = init_status(campaign_dir, spec, args.campaign)

    if args.command == "dashboard":
        cmd_dashboard(spec, status, campaign_dir, as_json=args.as_json)

    elif args.command == "outputs":
        cmd_outputs(spec, status, campaign_dir,
                    stage_id=args.stage_id,
                    preview_chars=args.preview_chars,
                    artifacts_only=args.artifacts_only)

    elif args.command == "logs":
        cmd_logs(campaign_dir, stage_id=args.stage_id,
                 lines=args.lines, stderr=args.stderr, follow=args.follow)

    elif args.command == "steer":
        cmd_steer(spec, status, campaign_dir,
                  action=args.action, text=args.text,
                  instr_type=args.instr_type,
                  port_override=args.port, host_override=args.host)

    return 0


if __name__ == "__main__":
    sys.exit(main())
