"""openpi doctor — validate environment and dependencies."""

from __future__ import annotations

import shutil

import click
from rich.console import Console
from rich.table import Table

from openpi.core.env_manager import check_required_keys, has_any_llm_key
from openpi.core.platform_detect import detect, detect_consortium

console = Console()


@click.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """Check your environment for issues.

    Validates Python version, installed packages, API keys,
    and optional dependencies like LaTeX and Playwright.
    """
    config_dir = ctx.obj.get("config_dir")
    all_ok = True

    console.print("[bold]OpenPI Doctor[/]\n")

    table = Table(title="Environment Check", show_lines=True)
    table.add_column("Check", style="bold", min_width=30)
    table.add_column("Status", min_width=12)
    table.add_column("Details")

    # Python version
    info = detect()
    if info.missing:
        table.add_row("Python >= 3.10", "[red]FAIL[/]", info.missing[0])
        all_ok = False
    else:
        table.add_row("Python >= 3.10", "[green]OK[/]", f"{info.python_version}")

    # Consortium package
    has_consortium, consortium_path = detect_consortium()
    if has_consortium:
        table.add_row("Consortium engine", "[green]OK[/]", consortium_path)
    else:
        table.add_row(
            "Consortium engine",
            "[red]MISSING[/]",
            "pip install -e /path/to/OpenPI",
        )
        all_ok = False

    # Consortium CLI command
    if shutil.which("consortium"):
        table.add_row("consortium command", "[green]OK[/]", shutil.which("consortium") or "")
    else:
        table.add_row("consortium command", "[red]MISSING[/]", "Not on PATH")
        all_ok = False

    # Git
    if info.has_git:
        table.add_row("Git", "[green]OK[/]", shutil.which("git") or "")
    else:
        table.add_row("Git", "[yellow]WARN[/]", "Not found — needed for experiment tracking")

    # API Keys
    key_results = check_required_keys(config_dir)
    any_key = has_any_llm_key(config_dir)
    for kr in key_results:
        level = kr["level"]
        configured = kr["configured"]
        name = kr["name"]

        if configured:
            table.add_row(f"API Key: {name}", "[green]OK[/]", kr["env_var"])
        elif level == "required":
            table.add_row(f"API Key: {name}", "[yellow]MISSING[/]", f"{kr['env_var']} ({level})")
        elif level == "recommended":
            table.add_row(f"API Key: {name}", "[dim]SKIP[/]", f"{kr['env_var']} ({level})")
        else:
            table.add_row(f"API Key: {name}", "[dim]SKIP[/]", f"{kr['env_var']} ({level})")

    if not any_key:
        all_ok = False

    # LaTeX
    if info.has_pdflatex:
        table.add_row("LaTeX (pdflatex)", "[green]OK[/]", shutil.which("pdflatex") or "")
    else:
        table.add_row("LaTeX (pdflatex)", "[dim]SKIP[/]", "Optional — needed for PDF/LaTeX output")

    # Playwright
    if info.has_playwright:
        table.add_row("Playwright", "[green]OK[/]", "Installed")
    else:
        table.add_row("Playwright", "[dim]SKIP[/]", "Optional — needed for web research")

    # SLURM
    if info.has_slurm:
        table.add_row("SLURM", "[green]OK[/]", "sbatch available")
    else:
        table.add_row("SLURM", "[dim]SKIP[/]", "Not detected — HPC mode unavailable")

    console.print(table)

    # Summary
    console.print()
    if all_ok:
        console.print("[bold green]All checks passed![/] You're ready to run OpenPI.")
    elif not any_key:
        console.print("[bold red]No LLM API keys configured.[/] Run [bold]openpi setup[/] to add them.")
        raise SystemExit(1)
    else:
        console.print("[bold yellow]Some checks have warnings.[/] See details above.")
        if not has_consortium:
            console.print("  Install consortium: [bold]pip install -e /path/to/OpenPI[/]")
