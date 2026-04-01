"""msc status — check running pipelines and recent runs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Theme: blue, gray, white ────────────────────────────────────────
_BLUE = "bold blue"
_GRAY = "dim white"
_WHITE = "white"
_GREEN = "bold green"
_RED = "bold red"
_YELLOW = "bold yellow"

from consortium.cli.core.paths import find_results_dir as _find_results_dir

console = Console()

_ACTIVE_THRESHOLD_SECONDS = 30 * 60  # 30 minutes


def _seconds_to_human(seconds: float) -> str:
    """Convert seconds to a human-readable elapsed string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"


def _is_pid_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _detect_run_status(run_dir: Path) -> dict:
    """Detect the status of a single run directory.

    Returns a dict with keys: status, stage, elapsed, budget.
    """
    now = time.time()
    info: dict[str, str] = {
        "status": "unknown",
        "stage": "-",
        "elapsed": "-",
        "budget": "-",
    }

    # --- Check for active indicators ---

    # 1. Progress heartbeat file
    heartbeat = run_dir / ".progress_heartbeat"
    if heartbeat.exists():
        age = now - heartbeat.stat().st_mtime
        if age < _ACTIVE_THRESHOLD_SECONDS:
            info["status"] = "active"
            info["elapsed"] = _seconds_to_human(age)
            # Try to read stage from heartbeat
            try:
                hb_data = json.loads(heartbeat.read_text())
                info["stage"] = hb_data.get("stage", "-")
            except (json.JSONDecodeError, OSError):
                pass

    # 2. PID files
    for pid_file in run_dir.glob("*.pid"):
        try:
            pid = int(pid_file.read_text().strip())
            if _is_pid_running(pid):
                info["status"] = "active"
                break
        except (ValueError, OSError):
            continue

    # 3. Recent consortium_*.out files
    for out_file in run_dir.glob("consortium_*.out"):
        try:
            age = now - out_file.stat().st_mtime
            if age < _ACTIVE_THRESHOLD_SECONDS:
                info["status"] = "active"
                break
        except OSError:
            continue

    # --- Check completion / failure ---
    if info["status"] != "active":
        # Check for final paper
        has_paper = any(
            (run_dir / f"final_paper{ext}").exists()
            for ext in (".pdf", ".md", ".tex")
        )
        if has_paper:
            info["status"] = "completed"
        else:
            # Check run_summary.json
            summary_path = run_dir / "run_summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                    if summary.get("completed"):
                        info["status"] = "completed"
                    elif summary.get("error") or summary.get("failed"):
                        info["status"] = "failed"
                    else:
                        info["status"] = "partial"
                except (json.JSONDecodeError, OSError):
                    pass

    # --- Elapsed time from directory mtime ---
    if info["elapsed"] == "-":
        try:
            dir_age = now - run_dir.stat().st_mtime
            info["elapsed"] = _seconds_to_human(dir_age) + " ago"
        except OSError:
            pass

    # --- Budget from summary or budget_state ---
    for budget_file_name in ("run_summary.json", "budget_state.json"):
        budget_path = run_dir / budget_file_name
        if budget_path.exists():
            try:
                data = json.loads(budget_path.read_text())
                cost = data.get("total_cost_usd")
                if cost is not None:
                    info["budget"] = f"${cost:.2f}"
                    break
            except (json.JSONDecodeError, OSError):
                continue

    # --- Stage from run_summary if not already set ---
    if info["stage"] == "-":
        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
                info["stage"] = summary.get("current_stage", summary.get("stage", "-"))
            except (json.JSONDecodeError, OSError):
                pass

    return info


def _get_slurm_jobs() -> list[dict]:
    """Query SLURM for running consortium jobs. Returns empty list if squeue unavailable."""
    if not shutil.which("squeue"):
        return []

    user = os.environ.get("USER", "")
    if not user:
        return []

    try:
        result = subprocess.run(
            ["squeue", "-u", user, "--format=%i|%j|%T|%M|%P", "--noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    jobs = []
    for line in result.stdout.strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 4:
            job_id, name, state, elapsed = parts[0], parts[1], parts[2], parts[3]
            if "consortium" in name.lower():
                jobs.append({
                    "job_id": job_id.strip(),
                    "name": name.strip(),
                    "state": state.strip(),
                    "elapsed": elapsed.strip(),
                })
    return jobs


def _get_campaign_status(results_dir: Path) -> list[dict]:
    """Find campaign_state.json files and extract campaign status."""
    campaigns = []

    # Check results dir and parent for campaign state files
    search_dirs = [results_dir, results_dir.parent]
    seen: set[str] = set()

    for search_dir in search_dirs:
        for state_file in search_dir.rglob("campaign_state.json"):
            canon = str(state_file.resolve())
            if canon in seen:
                continue
            seen.add(canon)
            try:
                data = json.loads(state_file.read_text())
                campaigns.append({
                    "name": data.get("campaign_name", data.get("slug", state_file.parent.name)),
                    "status": data.get("status", "unknown"),
                    "stage": data.get("current_stage", "-"),
                    "budget": f"${data.get('total_cost_usd', 0):.2f}" if "total_cost_usd" in data else "-",
                })
            except (json.JSONDecodeError, OSError):
                continue

    return campaigns


def _style_status(status: str) -> Text:
    """Return a styled Text object for the given status string."""
    mapping = {
        "active": (_GREEN, "active"),
        "completed": ("green", "completed"),
        "failed": (_RED, "failed"),
        "partial": (_YELLOW, "partial"),
        "unknown": (_GRAY, "unknown"),
    }
    style, label = mapping.get(status, (_GRAY, status))
    return Text(label, style=style)


@click.command()
@click.option("--results-dir", type=click.Path(exists=True), default=None,
              help="Override the results directory path.")
@click.option("--all", "show_all", is_flag=True, help="Show all runs, not just recent ones.")
@click.pass_context
def status(ctx: click.Context, results_dir: str | None, show_all: bool) -> None:
    """Check running pipelines and recent runs.

    \b
    Examples:
      msc status          # Show active and recent runs
      msc status --all    # Show all runs
    """
    rdir = Path(results_dir) if results_dir else _find_results_dir()

    has_any_output = False

    # ── Pipeline runs ───────────────────────────────────────────────
    if rdir and rdir.is_dir():
        run_dirs = sorted(
            [d for d in rdir.iterdir() if d.is_dir() and not d.name.startswith(".")],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )

        if not show_all:
            run_dirs = run_dirs[:15]

        if run_dirs:
            table = Table(
                title="Pipeline Runs",
                title_style=_BLUE,
                border_style=_GRAY,
                header_style=_BLUE,
            )
            table.add_column("Run ID", style=_WHITE, max_width=40)
            table.add_column("Status", min_width=10)
            table.add_column("Stage", style=_WHITE)
            table.add_column("Elapsed", style=_WHITE, justify="right")
            table.add_column("Budget", style=_WHITE, justify="right")

            active_count = 0
            for d in run_dirs:
                info = _detect_run_status(d)
                if info["status"] == "active":
                    active_count += 1
                table.add_row(
                    d.name,
                    _style_status(info["status"]),
                    info["stage"],
                    info["elapsed"],
                    info["budget"],
                )

            console.print()
            console.print(table)
            has_any_output = True

            if active_count > 0:
                console.print(
                    Text(f"\n  {active_count} active run(s)", style=_GREEN)
                )

    # ── SLURM jobs ──────────────────────────────────────────────────
    slurm_jobs = _get_slurm_jobs()
    if slurm_jobs:
        slurm_table = Table(
            title="SLURM Jobs",
            title_style=_BLUE,
            border_style=_GRAY,
            header_style=_BLUE,
        )
        slurm_table.add_column("Job ID", style=_WHITE)
        slurm_table.add_column("Name", style=_WHITE)
        slurm_table.add_column("State")
        slurm_table.add_column("Elapsed", style=_WHITE, justify="right")

        for job in slurm_jobs:
            state_style = _GREEN if job["state"] == "RUNNING" else _YELLOW
            slurm_table.add_row(
                job["job_id"],
                job["name"],
                Text(job["state"], style=state_style),
                job["elapsed"],
            )

        console.print()
        console.print(slurm_table)
        has_any_output = True

    # ── Campaign status ─────────────────────────────────────────────
    if rdir:
        campaigns = _get_campaign_status(rdir)
        if campaigns:
            camp_table = Table(
                title="Campaigns",
                title_style=_BLUE,
                border_style=_GRAY,
                header_style=_BLUE,
            )
            camp_table.add_column("Campaign", style=_WHITE)
            camp_table.add_column("Status")
            camp_table.add_column("Stage", style=_WHITE)
            camp_table.add_column("Budget", style=_WHITE, justify="right")

            for c in campaigns:
                camp_table.add_row(
                    c["name"],
                    _style_status(c["status"]),
                    c["stage"],
                    c["budget"],
                )

            console.print()
            console.print(camp_table)
            has_any_output = True

    # ── Nothing found ───────────────────────────────────────────────
    if not has_any_output:
        console.print()
        console.print(
            Panel(
                Text.from_markup(
                    '[dim]No active runs.[/]\n\n'
                    'Start one with: [bold blue]msc run "your question"[/bold blue]'
                ),
                title="Status",
                title_align="left",
                border_style=_GRAY,
                style=_WHITE,
                padding=(1, 2),
            )
        )
    console.print()
