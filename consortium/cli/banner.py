"""ASCII banner and welcome screen for the msc CLI."""

from __future__ import annotations

import random

from rich.console import Console
from rich.text import Text

from consortium.cli import __version__

# ── ASCII art ────────────────────────────────────────────────────────
# Clean, readable block-letter font.
_BANNER_LINES = [
    " __  __  ____       ",
    "|  \\/  |/ ___|  ___ ",
    "| |\\/| |\\___ \\ / __|",
    "| |  | | ___) | (__ ",
    "|_|  |_||____/ \\___|",
]

_TAGLINE = "PoggioAI/MSc  \u2022  End-to-end Agentic Research  \u2022  Poggio Lab @ MIT"

# ── Rotating tips ────────────────────────────────────────────────────
_TIPS = [
    "Run [bold]msc setup[/bold] to configure API keys and preferences.",
    "Use [bold]msc run \"your question\"[/bold] to kick off a full research pipeline.",
    "Check your environment with [bold]msc doctor[/bold] before your first run.",
    "Set a budget cap with [bold]msc config set budget_usd 50[/bold].",
    "Resume an interrupted run with [bold]msc resume <run-id>[/bold].",
    "View past runs and their status with [bold]msc runs list[/bold].",
]

# ── Theme colors ─────────────────────────────────────────────────────
_BLUE = "bold blue"
_GRAY = "dim white"
_WHITE = "white"


def _build_banner() -> Text:
    """Assemble the styled banner as a Rich Text object."""
    parts: list[Text] = []

    # ASCII art in bold blue
    for line in _BANNER_LINES:
        t = Text(line)
        t.stylize(_BLUE)
        parts.append(t)

    # Blank separator
    parts.append(Text(""))

    # Tagline in gray
    tagline = Text(_TAGLINE)
    tagline.stylize(_GRAY)
    parts.append(tagline)

    # Version in white
    version_text = Text(f"v{__version__}")
    version_text.stylize(_WHITE)
    parts.append(version_text)

    # Join with newlines
    result = Text("\n").join(parts)
    return result


def print_banner(console: Console | None = None) -> None:
    """Print the MSc banner to the console."""
    if console is None:
        console = Console()
    console.print()
    console.print(_build_banner())
    console.print()


def print_welcome(console: Console | None = None) -> None:
    """Print the banner followed by a random quick-start tip."""
    if console is None:
        console = Console()

    print_banner(console)

    tip = random.choice(_TIPS)  # noqa: S311
    console.print(Text("Tip: ", style=_BLUE), end="")
    console.print(tip, highlight=False)
    console.print()
