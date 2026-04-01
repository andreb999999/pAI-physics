"""msc doctor — validate environment and dependencies."""

from __future__ import annotations

import shutil

import click
from rich.console import Console
from rich.table import Table

from consortium.cli.core.env_manager import check_required_keys, has_any_llm_key
from consortium.cli.core.platform_detect import detect, detect_consortium

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

    console.print("[bold blue]PoggioAI/MSc Doctor[/]\n")

    table = Table(
        title="[bold white]Environment Check[/]",
        show_lines=True,
        border_style="bright_black",
    )
    table.add_column("Check", style="bold white", min_width=30)
    table.add_column("Status", min_width=12)
    table.add_column("Details", style="dim white")

    # Python version
    info = detect()
    if info.missing:
        table.add_row("Python >= 3.10", "[bold white on red] FAIL [/]", info.missing[0])
        all_ok = False
    else:
        table.add_row("Python >= 3.10", "[blue]\u2713 OK[/]", f"{info.python_version}")

    # Consortium package
    has_consortium, consortium_path = detect_consortium()
    if has_consortium:
        table.add_row("MSc engine", "[blue]\u2713 OK[/]", consortium_path)
    else:
        table.add_row(
            "MSc engine",
            "[bold white on red] MISSING [/]",
            "pip install -e .",
        )
        all_ok = False

    # Consortium CLI command
    if shutil.which("consortium"):
        table.add_row("consortium command", "[blue]\u2713 OK[/]", shutil.which("consortium") or "")
    else:
        table.add_row("consortium command", "[bold white on red] MISSING [/]", "Not on PATH")
        all_ok = False

    # Git
    if info.has_git:
        table.add_row("Git", "[blue]\u2713 OK[/]", shutil.which("git") or "")
    else:
        table.add_row("Git", "[dim]WARN[/]", "Not found — needed for experiment tracking")

    # API Keys
    key_results = check_required_keys(config_dir)
    any_key = has_any_llm_key(config_dir)
    for kr in key_results:
        level = kr["level"]
        configured = kr["configured"]
        name = kr["name"]

        if configured:
            table.add_row(f"API Key: {name}", "[blue]\u2713 OK[/]", kr["env_var"])
        elif level == "required":
            table.add_row(f"API Key: {name}", "[dim]MISSING[/]", f"{kr['env_var']} ({level})")
        elif level == "recommended":
            table.add_row(f"API Key: {name}", "[dim]SKIP[/]", f"{kr['env_var']} ({level})")
        else:
            table.add_row(f"API Key: {name}", "[dim]SKIP[/]", f"{kr['env_var']} ({level})")

    if not any_key:
        all_ok = False

    # LaTeX
    if info.has_pdflatex:
        table.add_row("LaTeX (pdflatex)", "[blue]\u2713 OK[/]", shutil.which("pdflatex") or "")
    else:
        table.add_row("LaTeX (pdflatex)", "[dim]SKIP[/]", "Optional — needed for PDF/LaTeX output")

    # Playwright
    if info.has_playwright:
        table.add_row("Playwright", "[blue]\u2713 OK[/]", "Installed")
    else:
        table.add_row("Playwright", "[dim]SKIP[/]", "Optional — needed for web research")

    # SLURM
    if info.has_slurm:
        table.add_row("SLURM", "[blue]\u2713 OK[/]", "sbatch available")
    else:
        table.add_row("SLURM", "[dim]SKIP[/]", "Not detected — HPC mode unavailable")

    # .llm_config.yaml
    from pathlib import Path
    llm_cfg = Path(".llm_config.yaml")
    if llm_cfg.exists():
        try:
            import yaml
            with open(llm_cfg) as f:
                cfg = yaml.safe_load(f)
            if cfg and "main_agents" in cfg and "budget" in cfg:
                table.add_row(".llm_config.yaml", "[blue]\u2713 OK[/]", "Valid")
            else:
                table.add_row(".llm_config.yaml", "[dim]WARN[/]", "Missing required sections (main_agents, budget)")
        except Exception as exc:
            table.add_row(".llm_config.yaml", "[bold white on red] FAIL [/]", f"Parse error: {exc}")
    else:
        table.add_row(".llm_config.yaml", "[dim]SKIP[/]", "Not found — msc run will auto-generate from tier")

    # scripts/ directory (needed for campaign commands)
    scripts_dir = Path("scripts")
    if scripts_dir.is_dir() and (scripts_dir / "campaign_cli.py").exists():
        table.add_row("scripts/", "[blue]\u2713 OK[/]", "Campaign scripts found")
    elif scripts_dir.is_dir():
        table.add_row("scripts/", "[dim]WARN[/]", "Directory exists but campaign_cli.py missing")
    else:
        table.add_row("scripts/", "[dim]SKIP[/]", "Not in project root — campaign commands unavailable")

    console.print(table)

    # Summary
    console.print()
    if all_ok:
        console.print("[bold blue]\u2713 All checks passed![/] You're ready to run MSc.")
    elif not any_key:
        console.print("[bold white on red] No LLM API keys configured. [/] Run [bold white]msc setup[/] to add them.")
        raise SystemExit(1)
    else:
        console.print("[dim]Some checks have warnings.[/] See details above.")
        if not has_consortium:
            console.print("  Install engine: [bold white]pip install -e .[/]")
