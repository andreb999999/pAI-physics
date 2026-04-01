"""msc campaign — manage autonomous research campaigns."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def campaign() -> None:
    """Manage multi-stage research campaigns.

    \b
    Campaigns run multiple research stages autonomously with
    heartbeat monitoring, budget tracking, and auto-repair.

    \b
    Examples:
      msc campaign init                  # Create a new campaign
      msc campaign start my_campaign     # Launch it
      msc campaign status my_campaign    # Check progress
    """


@campaign.command("init")
@click.option("--name", prompt="Campaign name", help="Name for the campaign.")
@click.option("--task", prompt="Research task", help="The research question or task.")
@click.option("--budget", type=int, default=100, help="Budget in USD.")
@click.option("--output-dir", type=click.Path(), default=None)
def campaign_init(name: str, task: str, budget: int, output_dir: str | None) -> None:
    """Create a new campaign from an interactive wizard."""
    import yaml

    slug = name.lower().replace(" ", "_").replace("-", "_")
    out_dir = output_dir or f"results/{slug}"

    spec = {
        "name": name,
        "workspace_root": out_dir,
        "heartbeat_interval_minutes": 15,
        "max_idle_ticks": 6,
        "max_campaign_hours": 96,
        "planning": {
            "enabled": True,
            "base_task": task,
            "max_stages": 6,
            "human_review": False,
        },
        "stages": [],
        "budget": {
            "usd_limit": budget,
        },
        "repair": {
            "enabled": True,
            "max_attempts": 2,
            "launcher": "local",
            "two_phase": True,
        },
        "notification": {
            "on_stage_complete": True,
            "on_failure": True,
            "on_heartbeat": False,
        },
    }

    campaign_file = Path(f"{slug}_campaign.yaml")
    with open(campaign_file, "w") as f:
        yaml.dump(spec, f, default_flow_style=False, sort_keys=False)

    console.print(f"[blue]Campaign created:[/] {campaign_file}")
    console.print(f"  Launch with: [bold]msc campaign start {campaign_file}[/]")


@campaign.command("start")
@click.argument("campaign_file", type=click.Path(exists=True))
def campaign_start(campaign_file: str) -> None:
    """Launch a campaign from its YAML spec."""
    console.print(f"Starting campaign from [bold]{campaign_file}[/]...")
    try:
        proc = subprocess.run(
            [sys.executable, "scripts/campaign_heartbeat.py", "--campaign", campaign_file, "--init"],
            cwd=_find_project_root(),
        )
        if proc.returncode == 0:
            console.print("[blue]Campaign initialized and first heartbeat complete.[/]")
        else:
            console.print(f"[red]Campaign start failed[/] (exit code {proc.returncode})")
    except FileNotFoundError:
        console.print("[bold white on red] Error [/] Could not find campaign_heartbeat.py. Are you in the PoggioAI/MSc directory?")


@campaign.command("status")
@click.argument("campaign_file", type=click.Path(exists=True))
def campaign_status(campaign_file: str) -> None:
    """Show campaign progress and budget."""
    try:
        proc = subprocess.run(
            [sys.executable, "scripts/campaign_cli.py", "--campaign", campaign_file, "status"],
            capture_output=True,
            text=True,
            cwd=_find_project_root(),
        )
        if proc.returncode == 0:
            try:
                data = json.loads(proc.stdout)
                _render_campaign_status(data)
            except json.JSONDecodeError:
                console.print(proc.stdout)
        else:
            console.print(f"[bold white on red] Error [/] {proc.stderr}")
    except FileNotFoundError:
        console.print("[bold white on red] Error [/] campaign_cli.py not found.")


@campaign.command("repair")
@click.argument("campaign_file", type=click.Path(exists=True))
@click.argument("stage_id")
def campaign_repair(campaign_file: str, stage_id: str) -> None:
    """Trigger repair on a failed stage."""
    try:
        proc = subprocess.run(
            [sys.executable, "scripts/campaign_cli.py", "--campaign", campaign_file, "repair", stage_id],
            cwd=_find_project_root(),
        )
        if proc.returncode == 0:
            console.print(f"[blue]Repair triggered for stage {stage_id}[/]")
        else:
            console.print(f"[red]Repair failed[/]")
    except FileNotFoundError:
        console.print("[bold white on red] Error [/] campaign_cli.py not found.")


@campaign.command("list")
def campaign_list() -> None:
    """List all campaign YAML files in the current directory."""
    yamls = list(Path.cwd().glob("*_campaign.yaml")) + list(Path.cwd().glob("campaign_*.yaml"))
    if not yamls:
        console.print("[dim]No campaign files found.[/]")
        return

    table = Table(title="Campaigns")
    table.add_column("File", style="bold")
    table.add_column("Size")
    for y in sorted(yamls):
        table.add_row(y.name, f"{y.stat().st_size:,} bytes")
    console.print(table)


def _find_project_root() -> str | None:
    """Try to find the PoggioAI/MSc project root by walking up the directory tree."""
    current = Path.cwd()
    for _ in range(6):  # check CWD + up to 5 parents
        if (current / "scripts" / "campaign_cli.py").exists():
            return str(current)
        parent = current.parent
        if parent == current:  # filesystem root
            break
        current = parent
    return None


def _render_campaign_status(data: dict) -> None:
    """Render campaign status JSON as a Rich table."""
    table = Table(title=f"Campaign: {data.get('name', 'unknown')}")
    table.add_column("Stage", style="bold")
    table.add_column("Status")
    table.add_column("Duration")

    for stage in data.get("stages", []):
        status = stage.get("status", "unknown")
        style = {
            "completed": "[blue]",
            "in_progress": "[yellow]",
            "failed": "[red]",
            "pending": "[dim]",
        }.get(status, "[dim]")
        table.add_row(
            stage.get("id", "?"),
            f"{style}{status}[/]",
            stage.get("duration", "-"),
        )

    console.print(table)

    budget = data.get("budget", {})
    if budget:
        spent = budget.get("spent_usd", 0)
        limit = budget.get("limit_usd", 0)
        console.print(f"\nBudget: ${spent:.2f} / ${limit:.2f}")
