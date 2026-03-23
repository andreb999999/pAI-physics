"""msc runs — list past research runs."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


def _find_results_dir() -> Path | None:
    """Find the results directory by checking common locations."""
    candidates = [
        Path.cwd() / "results",
        Path.cwd().parent / "results",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


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
        run_id = d.name
        status = "[dim]unknown[/]"
        cost = "-"
        model = "-"
        task = "-"

        # Try to read run_summary.json
        summary_path = d / "run_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                status = "[blue]complete[/]" if summary.get("completed") else "[yellow]partial[/]"
                cost = f"${summary.get('total_cost_usd', 0):.2f}" if "total_cost_usd" in summary else "-"
                model = summary.get("model", "-")
                task = (summary.get("task", "-") or "-")[:50]
            except (json.JSONDecodeError, OSError):
                pass

        # Check for final paper
        if (d / "final_paper.pdf").exists():
            status = "[blue]complete (PDF)[/]"
        elif (d / "final_paper.md").exists():
            status = "[blue]complete (MD)[/]"
        elif (d / "final_paper.tex").exists():
            status = "[blue]complete (TeX)[/]"

        # Fallback: check budget state
        if cost == "-":
            budget_path = d / "budget_state.json"
            if budget_path.exists():
                try:
                    with open(budget_path, "r") as f:
                        bs = json.load(f)
                    cost = f"${bs.get('total_cost_usd', 0):.2f}"
                except (json.JSONDecodeError, OSError):
                    pass

        table.add_row(run_id, status, cost, model, task)

    console.print(table)
