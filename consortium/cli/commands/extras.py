"""msc install — install optional dependency groups."""

from __future__ import annotations

import subprocess
import sys

import click
from rich.console import Console

console = Console()

EXTRAS = {
    "docs": "Document parsing (docx, pptx, pdf, images)",
    "web": "Web research (Playwright, crawl4ai, search APIs)",
    "experiment": "ML experiments (PyTorch, HuggingFace, datasets)",
    "observability": "Tracing and monitoring (LangSmith, W&B)",
    "latex": "LaTeX document compilation (requires system texlive)",
    "all": "All optional dependencies",
}


@click.command("install")
@click.argument("extra", type=click.Choice(list(EXTRAS.keys())))
def install(extra: str) -> None:
    """Install optional dependency groups.

    \b
    Available extras:
      docs          Document parsing (docx, pdf, images)
      web           Web research (Playwright, search APIs)
      experiment    ML experiments (PyTorch, HuggingFace)
      observability Tracing (LangSmith, W&B)
      latex         LaTeX compilation (system texlive required)
      all           All optional dependencies

    \b
    Examples:
      msc install web          # Install web research deps
      msc install all          # Install everything
    """
    console.print(f"Installing [bold]{extra}[/]: {EXTRAS[extra]}")

    if extra == "latex":
        console.print("[yellow]Note:[/] LaTeX requires system-level texlive. Install it via:")
        console.print("  macOS:  [bold]brew install --cask mactex-no-gui[/]")
        console.print("  Ubuntu: [bold]sudo apt install texlive-full[/]")
        console.print("  conda:  [bold]conda install -c conda-forge texlive-core[/]")
        return

    # Install via pip
    pip_spec = f"poggio-ai[{extra}]"
    console.print(f"Running: [dim]pip install {pip_spec}[/]")

    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", pip_spec],
    )

    if proc.returncode == 0:
        console.print(f"[blue]Successfully installed {extra} dependencies.[/]")

        # Special post-install for web
        if extra in ("web", "all"):
            console.print("Installing Playwright browsers...")
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"])
    else:
        console.print(f"[red]Installation failed.[/] Check the output above for errors.")
