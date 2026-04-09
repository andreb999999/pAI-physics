"""msc runs — list past research runs."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from consortium.cli.core.paths import find_results_dir as _find_results_dir
from consortium.cli.core.run_inspector import inspect_run

console = Console()


@click.command()
@click.option("--limit", "-n", type=int, default=10, help="Number of recent runs to show.")
@click.option("--results-dir", type=click.Path(exists=True), default=None)
def runs(limit: int, results_dir: str | None) -> None:
    """List past research runs with status and cost.

    \b
    Examples:
      msc runs           # Show last 10 runs
      msc runs -n 20     # Show last 20 runs
    """
    rdir = Path(results_dir) if results_dir else _find_results_dir()
    if not rdir:
        console.print("[yellow]No results/ directory found.[/] Run a pipeline first.")
        return

    # Find run directories (sorted newest first)
    # Include all subdirectories — runs may have custom names or consortium_ prefix
    run_dirs = sorted(
        [d for d in rdir.iterdir() if d.is_dir() and not d.name.startswith(".")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )[:limit]

    if not run_dirs:
        console.print("[dim]No runs found.[/]")
        return

    table = Table(title=f"Recent Runs ({len(run_dirs)} of {len(list(rdir.iterdir()))})")
    table.add_column("Run ID", style="bold")
    table.add_column("Status")
    table.add_column("Cost")
    table.add_column("Model")
    table.add_column("Task")

    for d in run_dirs:
        info = inspect_run(d)
        status_map = {
            "active": "[green]active[/]",
            "stalled": "[yellow]stalled[/]",
            "completed": "[blue]completed[/]",
            "failed": "[red]failed[/]",
            "partial": "[yellow]partial[/]",
            "unknown": "[dim]unknown[/]",
        }
        cost = f"${info['budget_usd']:.2f}" if info["budget_usd"] is not None else "-"
        task = (info["task"] or "-")[:50]
        table.add_row(
            info["run_id"],
            status_map.get(info["status"], f"[dim]{info['status']}[/]"),
            cost,
            info["model"],
            task,
        )

    console.print(table)
