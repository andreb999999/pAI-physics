"""msc budget — view spending summary."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.command()
@click.option("--results-dir", type=click.Path(exists=True), default=None)
def budget(results_dir: str | None) -> None:
    """View spending across all runs.

    \b
    Examples:
      msc budget              # Summary of all runs
    """
    rdir = Path(results_dir) if results_dir else Path.cwd() / "results"
    if not rdir.is_dir():
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
        cost = 0.0
        model = "-"

        for fname in ("budget_state.json", "run_summary.json"):
            fpath = d / fname
            if fpath.exists():
                try:
                    with open(fpath, "r") as f:
                        data = json.load(f)
                    cost = data.get("total_usd", data.get("total_cost_usd", 0.0))
                    model = data.get("model", model)
                    if cost > 0:
                        break
                except (json.JSONDecodeError, OSError):
                    pass

        if cost > 0:
            total += cost
            table.add_row(d.name, f"${cost:.2f}", model)

    console.print(table)
    console.print(f"\n[bold]Total spend:[/] ${total:.2f}")
