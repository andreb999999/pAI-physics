"""msc campaign — manage autonomous research campaigns."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from consortium.cli.core.env_manager import build_runtime_env
from consortium.cli.core.paths import find_project_root, find_script_path

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
    task_dir = Path("automation_tasks") / "generated"
    task_dir.mkdir(parents=True, exist_ok=True)
    task_file = task_dir / f"{slug}_discovery_task.txt"
    task_file.write_text(task.strip() + "\n")

    spec = {
        "name": name,
        "workspace_root": out_dir,
        "budget_usd": budget,
        "heartbeat_interval_minutes": 15,
        "max_idle_ticks": 6,
        "max_campaign_hours": 96,
        "planning": {
            "enabled": True,
            "base_task_file": task_file.as_posix(),
            "max_stages": 6,
            "human_review": False,
        },
        "stages": [],
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
    console.print(f"  Task file: [bold]{task_file}[/]")
    console.print(f"  Launch with: [bold]msc campaign start {campaign_file}[/]")


@campaign.command("start")
@click.argument("campaign_file", type=click.Path(exists=True))
@click.pass_context
def campaign_start(ctx: click.Context, campaign_file: str) -> None:
    """Launch a campaign from its YAML spec."""
    campaign_file = str(Path(campaign_file).resolve())
    console.print(f"Starting campaign from [bold]{campaign_file}[/]...")
    project_root = find_project_root()
    script_path = find_script_path("campaign_heartbeat.py")
    if script_path is None or project_root is None:
        console.print(
            "[bold white on red] Error [/] Could not find the packaged campaign runtime. "
            "Run this from an installed source checkout."
        )
        raise SystemExit(1)

    env = _campaign_subprocess_env(ctx, project_root)
    proc = subprocess.run(
        [sys.executable, str(script_path), "--campaign", campaign_file, "--init"],
        cwd=project_root,
        env=env,
    )
    if proc.returncode == 0:
        console.print("[blue]Campaign initialized and first heartbeat complete.[/]")
    else:
        console.print(f"[red]Campaign start failed[/] (exit code {proc.returncode})")


@campaign.command("status")
@click.argument("campaign_file", type=click.Path(exists=True))
@click.pass_context
def campaign_status(ctx: click.Context, campaign_file: str) -> None:
    """Show campaign progress and budget."""
    campaign_file = str(Path(campaign_file).resolve())
    project_root = find_project_root()
    script_path = find_script_path("campaign_cli.py")
    if script_path is None or project_root is None:
        console.print("[bold white on red] Error [/] campaign runtime not found.")
        raise SystemExit(1)

    env = _campaign_subprocess_env(ctx, project_root)
    proc = subprocess.run(
        [sys.executable, str(script_path), "--campaign", campaign_file, "status"],
        capture_output=True,
        text=True,
        cwd=project_root,
        env=env,
    )
    if proc.returncode == 0:
        try:
            data = json.loads(proc.stdout)
            _render_campaign_status(data)
        except json.JSONDecodeError:
            console.print(proc.stdout)
    else:
        console.print(f"[bold white on red] Error [/] {proc.stderr}")


@campaign.command("repair")
@click.argument("campaign_file", type=click.Path(exists=True))
@click.argument("stage_id")
@click.pass_context
def campaign_repair(ctx: click.Context, campaign_file: str, stage_id: str) -> None:
    """Trigger repair on a failed stage."""
    campaign_file = str(Path(campaign_file).resolve())
    project_root = find_project_root()
    script_path = find_script_path("campaign_cli.py")
    if script_path is None or project_root is None:
        console.print("[bold white on red] Error [/] campaign runtime not found.")
        raise SystemExit(1)

    env = _campaign_subprocess_env(ctx, project_root)
    proc = subprocess.run(
        [sys.executable, str(script_path), "--campaign", campaign_file, "repair", stage_id],
        cwd=project_root,
        env=env,
    )
    if proc.returncode == 0:
        console.print(f"[blue]Repair triggered for stage {stage_id}[/]")
    else:
        console.print(f"[red]Repair failed[/]")


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


def _campaign_subprocess_env(ctx: click.Context, project_root: Path | None) -> dict[str, str]:
    """Build a subprocess env that mirrors msc + direct-script runtime behavior."""
    return build_runtime_env(
        config_dir_override=ctx.obj.get("config_dir"),
        repo_root=project_root,
    )


def _render_campaign_status(data: dict) -> None:
    """Render campaign status JSON as a Rich table."""
    table = Table(title=f"Campaign: {data.get('name') or data.get('campaign_name', 'unknown')}")
    table.add_column("Stage", style="bold")
    table.add_column("Status")
    table.add_column("Duration")

    stages = data.get("stages", [])
    if isinstance(stages, dict):
        stage_items = []
        for stage_id, stage_data in stages.items():
            if isinstance(stage_data, dict):
                stage_items.append({"id": stage_id, **stage_data})
            else:
                stage_items.append({"id": stage_id, "status": str(stage_data)})
    else:
        stage_items = [stage for stage in stages if isinstance(stage, dict)]

    for stage in stage_items:
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
        spent = budget.get("spent_usd", budget.get("campaign_lifetime_usd", budget.get("current_attempt_usd", 0)))
        limit = budget.get("limit_usd", budget.get("campaign_limit_usd", 0))
        console.print(f"\nBudget: ${spent:.2f} / ${limit:.2f}")
