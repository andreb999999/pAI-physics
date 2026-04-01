"""openpi run — execute a research pipeline."""

from __future__ import annotations

import subprocess
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from openpi.core.env_manager import has_any_llm_key, inject_env
from openpi.core.flag_translator import build_argv
from openpi.core.presets import PRESETS, list_presets

console = Console()


@click.command()
@click.argument("task", required=False)
@click.option(
    "--preset", "-p",
    type=click.Choice(list(PRESETS.keys())),
    default="standard",
    help="Quality preset: quick ($2-5), standard ($10-25), thorough ($40-100), maximum ($80-200).",
)
@click.option("--task-file", "-f", type=click.Path(exists=True), help="Read task from a file.")
@click.option("--model", "-m", type=str, default=None, help="Override the LLM model.")
@click.option("--budget", "-b", type=int, default=None, help="Override the budget in USD.")
@click.option("--output-format", "-o", type=click.Choice(["markdown", "latex"]), default=None)
@click.option("--mode", type=click.Choice(["local", "tinker", "hpc"]), default=None)
@click.option("--dry-run", is_flag=True, help="Validate setup without running (no cost).")
@click.option("--counsel/--no-counsel", default=None, help="Enable/disable multi-model counsel.")
@click.option("--math/--no-math", default=None, help="Enable/disable math agents.")
@click.option("--tree-search/--no-tree-search", default=None, help="Enable/disable tree search.")
@click.option("--max-run-seconds", type=int, default=None, help="Hard timeout in seconds.")
@click.pass_context
def run(
    ctx: click.Context,
    task: str | None,
    preset: str,
    task_file: str | None,
    model: str | None,
    budget: int | None,
    output_format: str | None,
    mode: str | None,
    dry_run: bool,
    counsel: bool | None,
    math: bool | None,
    tree_search: bool | None,
    max_run_seconds: int | None,
) -> None:
    """Run a research pipeline on a question or topic.

    \b
    Examples:
      openpi run "How do transformers handle long-range dependencies?"
      openpi run --preset thorough "Prove the Goldbach conjecture for small primes"
      openpi run --task-file my_task.txt --preset maximum
      openpi run "Quick survey of GAN architectures" --preset quick --dry-run
    """
    # Resolve task
    if task_file and not task:
        with open(task_file, "r") as f:
            task = f.read().strip()
    if not task:
        console.print("[red]Error:[/] Provide a research task as an argument or via --task-file.")
        raise SystemExit(1)

    config_dir = ctx.obj.get("config_dir")
    quiet = ctx.obj.get("quiet", False)

    # Check for API keys
    if not has_any_llm_key(config_dir):
        console.print(
            "[red]Error:[/] No API keys configured. Run [bold]openpi setup[/] first."
        )
        raise SystemExit(1)

    # Build overrides from CLI flags
    overrides: dict[str, object] = {"dry_run": dry_run}
    if model:
        overrides["model"] = model
    if output_format:
        overrides["output_format"] = output_format
    if mode:
        overrides["mode"] = mode
    if max_run_seconds:
        overrides["max_run_seconds"] = max_run_seconds
    if counsel is True:
        overrides["enable_counsel"] = True
        overrides["no_counsel"] = False
    elif counsel is False:
        overrides["enable_counsel"] = False
        overrides["no_counsel"] = True
    if math is True:
        overrides["enable_math_agents"] = True
    elif math is False:
        overrides["enable_math_agents"] = False
    if tree_search is True:
        overrides["enable_tree_search"] = True
    elif tree_search is False:
        overrides["enable_tree_search"] = False

    # Build argv
    argv = build_argv(task, preset, **overrides)

    # Show run summary
    if not quiet:
        p = PRESETS[preset]
        effective_model = model or p.model
        effective_budget = budget or p.budget_usd

        table = Table(title="Run Configuration", show_header=False, padding=(0, 2))
        table.add_column("Key", style="bold cyan")
        table.add_column("Value")
        table.add_row("Task", task[:80] + ("..." if len(task) > 80 else ""))
        table.add_row("Preset", preset)
        table.add_row("Model", effective_model)
        table.add_row("Budget", f"${effective_budget}")
        table.add_row("Output", output_format or p.output_format)
        table.add_row("Counsel", "on" if (counsel if counsel is not None else p.enable_counsel) else "off")
        table.add_row("Dry run", "yes" if dry_run else "no")
        console.print(table)
        console.print()

    # Inject API keys into environment
    inject_env(config_dir)

    # Run consortium
    if not quiet:
        console.print("[bold green]Starting pipeline...[/]")
        console.print(f"[dim]$ {' '.join(argv)}[/dim]\n")

    try:
        proc = subprocess.run(argv, env=dict(**__import__("os").environ))
        raise SystemExit(proc.returncode)
    except FileNotFoundError:
        console.print(
            "[red]Error:[/] 'consortium' command not found. "
            "Install it with: [bold]pip install -e /path/to/OpenPI[/]"
        )
        raise SystemExit(1)
