"""msc budget — view spending summary."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from consortium.cli.core.paths import find_results_dir
from consortium.cli.core.run_inspector import inspect_run

console = Console()


@click.command()
@click.option("--results-dir", type=click.Path(exists=True), default=None)
def budget(results_dir: str | None) -> None:
    """View spending across all runs.

    \b
    Examples:
      msc budget              # Summary of all runs
    """
    rdir = Path(results_dir) if results_dir else find_results_dir()
    if rdir is None or not rdir.is_dir():
        console.print("[dim]No results/ directory found.[/]")
        return

    table = Table(title="Spending Summary")
    table.add_column("Run", style="bold")
    table.add_column("Cost", justify="right")
    table.add_column("Model")

    total = 0.0
    run_dirs = sorted(
        [d for d in rdir.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    for d in run_dirs:
        info = inspect_run(d)
        cost = info["budget_usd"] or 0.0
        if cost > 0:
            total += cost
            table.add_row(d.name, f"${cost:.2f}", info["model"])

    console.print(table)
    console.print(f"\n[bold]Total spend:[/] ${total:.2f}")
