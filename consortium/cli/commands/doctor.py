"""msc doctor — validate environment and dependencies."""

from __future__ import annotations

import shutil
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from consortium.cli.core.env_manager import check_required_keys, has_required_llm_key
from consortium.cli.core.paths import find_project_root, find_script_path
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
    project_root = find_project_root()
    campaign_cli_path = find_script_path("campaign_cli.py")
    campaign_heartbeat_path = find_script_path("campaign_heartbeat.py")

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

    # Runner entrypoint
    if has_consortium:
        table.add_row("Runner module", "[blue]\u2713 OK[/]", "python -m consortium.runner")
    else:
        table.add_row("Runner module", "[bold white on red] MISSING [/]", "consortium package not importable")
        all_ok = False

    # Git
    if info.has_git:
        table.add_row("Git", "[blue]\u2713 OK[/]", shutil.which("git") or "")
    else:
        table.add_row("Git", "[dim]WARN[/]", "Not found — needed for experiment tracking")

    # API Keys
    key_results = check_required_keys(config_dir, repo_root=project_root)
    has_openrouter = has_required_llm_key(config_dir, repo_root=project_root)
    for kr in key_results:
        level = kr["level"]
        configured = kr["configured"]
        name = kr["name"]
        source = kr.get("source")
        detail = kr["env_var"]
        if source:
            detail = f"{detail} via {source}"

        if configured:
            table.add_row(f"API Key: {name}", "[blue]\u2713 OK[/]", detail)
        elif level == "required":
            table.add_row(f"API Key: {name}", "[dim]MISSING[/]", f"{detail} ({level})")
        elif level == "recommended":
            table.add_row(f"API Key: {name}", "[dim]SKIP[/]", f"{detail} ({level})")
        else:
            table.add_row(f"API Key: {name}", "[dim]SKIP[/]", f"{detail} ({level})")

    if not has_openrouter:
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

    # Campaign runtime
    if campaign_cli_path and campaign_heartbeat_path and project_root:
        table.add_row("Campaign runtime", "[blue]\u2713 OK[/]", str(project_root / "scripts"))
    elif project_root:
        table.add_row("Campaign runtime", "[dim]WARN[/]", f"Incomplete scripts under {project_root / 'scripts'}")
    else:
        table.add_row("Campaign runtime", "[dim]SKIP[/]", "Source checkout not discoverable — campaign commands unavailable")

    console.print(table)

    # Summary
    console.print()
    if all_ok:
        console.print("[bold blue]\u2713 All checks passed![/] You're ready to run MSc.")
    elif not has_openrouter:
        console.print(
            "[bold white on red] OPENROUTER_API_KEY is not configured. [/] "
            "Run [bold white]msc setup[/] or add it to your environment."
        )
        raise SystemExit(1)
    else:
        console.print("[dim]Some checks have warnings.[/] See details above.")
        if not has_consortium:
            console.print("  Install engine: [bold white]pip install -e .[/]")
