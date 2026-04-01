"""openpi resume — resume a previous research run from checkpoint."""

from __future__ import annotations

import subprocess
from pathlib import Path

import click
from rich.console import Console

from openpi.core.env_manager import inject_env

console = Console()


def _find_latest_run() -> Path | None:
    """Find the most recent run directory."""
    results = Path.cwd() / "results"
    if not results.is_dir():
        return None
    run_dirs = sorted(
        [d for d in results.iterdir() if d.is_dir() and d.name.startswith("consortium_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


@click.command()
@click.argument("run_id", required=False)
@click.option("--start-from", type=str, default=None, help="Resume from a specific stage.")
@click.option("--task", type=str, default=None, help="Override the task for the resumed run.")
@click.pass_context
def resume(
    ctx: click.Context,
    run_id: str | None,
    start_from: str | None,
    task: str | None,
) -> None:
    """Resume a previous run from its checkpoint.

    \b
    Examples:
      openpi resume                                       # Resume latest run
      openpi resume consortium_20260322_143022_task       # Resume specific run
      openpi resume --start-from experiment               # Resume from a specific stage
    """
    config_dir = ctx.obj.get("config_dir")

    # Resolve run directory
    if run_id:
        run_dir = Path.cwd() / "results" / run_id
        if not run_dir.is_dir():
            # Try as absolute path
            run_dir = Path(run_id)
        if not run_dir.is_dir():
            console.print(f"[red]Error:[/] Run directory not found: {run_id}")
            raise SystemExit(1)
    else:
        run_dir = _find_latest_run()
        if not run_dir:
            console.print("[red]Error:[/] No previous runs found in results/")
            raise SystemExit(1)

    console.print(f"Resuming from [bold]{run_dir.name}[/]")

    # Build argv
    argv = ["consortium", "--resume", str(run_dir)]
    if start_from:
        argv.extend(["--start-from-stage", start_from])
    if task:
        argv.extend(["--task", task])

    # Inject API keys
    inject_env(config_dir)

    try:
        proc = subprocess.run(argv)
        raise SystemExit(proc.returncode)
    except FileNotFoundError:
        console.print("[red]Error:[/] 'consortium' not found. Install the consortium package.")
        raise SystemExit(1)
