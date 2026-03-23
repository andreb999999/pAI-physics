"""Custom error handling for the msc CLI with Rich-formatted output."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# ── Theme colors ─────────────────────────────────────────────────────
_BLUE = "blue"
_BOLD_WHITE = "bold white"
_WHITE = "white"


# ── Base exception ───────────────────────────────────────────────────

class MSCException(Exception):
    """Base exception for all msc CLI errors."""

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        super().__init__(message)
        self.suggestion = suggestion


# ── Error → suggestion mapping ───────────────────────────────────────

_ERROR_MAP: dict[str, str] = {
    "api_key": "Run `msc setup` to configure your API keys.",
    "consortium_not_found": "Run `pip install -e .` to install the engine.",
    "budget_exceeded": "Increase budget with `msc config set budget_usd 50`.",
    "network_timeout": "Check your internet connection and API key validity.",
    "missing_config": "Run `msc setup` for first-time configuration.",
}

# Patterns matched against the stringified exception to auto-detect the category.
_PATTERN_TO_KEY: list[tuple[str, str]] = [
    ("api_key", "api_key"),
    ("API key", "api_key"),
    ("OPENAI_API_KEY", "api_key"),
    ("ANTHROPIC_API_KEY", "api_key"),
    ("AuthenticationError", "api_key"),
    ("No module named 'consortium'", "consortium_not_found"),
    ("ModuleNotFoundError", "consortium_not_found"),
    ("ImportError", "consortium_not_found"),
    ("BudgetExceeded", "budget_exceeded"),
    ("budget", "budget_exceeded"),
    ("TimeoutError", "network_timeout"),
    ("ConnectTimeout", "network_timeout"),
    ("ReadTimeout", "network_timeout"),
    ("ConnectionError", "network_timeout"),
    ("config", "missing_config"),
    ("FileNotFoundError", "missing_config"),
    (".msc/config", "missing_config"),
]


def _match_suggestion(error: BaseException) -> str | None:
    """Return a suggestion string if the error matches a known pattern."""
    error_str = f"{type(error).__name__}: {error}"
    for pattern, key in _PATTERN_TO_KEY:
        if pattern.lower() in error_str.lower():
            return _ERROR_MAP.get(key)
    return None


# ── Formatting helpers ───────────────────────────────────────────────

def format_error(error: str | BaseException, suggestion: str | None = None) -> Panel:
    """Build a Rich Panel for a single error with optional suggestion.

    Parameters
    ----------
    error:
        The error message or exception instance.
    suggestion:
        A human-readable recovery hint.  When *None* and *error* is an
        exception, the function will attempt to auto-detect one.
    """
    if isinstance(error, BaseException):
        if suggestion is None:
            suggestion = _match_suggestion(error)
        error_msg = str(error)
    else:
        error_msg = error

    body = Text()
    body.append(error_msg, style=_WHITE)

    if suggestion:
        body.append("\n\n")
        body.append("Suggestion: ", style=_BOLD_WHITE)
        body.append(suggestion, style=_BLUE)

    return Panel(
        body,
        title="[bold white]Error[/bold white]",
        border_style=_BLUE,
        expand=False,
        padding=(1, 2),
    )


# ── Context manager ──────────────────────────────────────────────────

@contextmanager
def error_handler(
    console: Console | None = None,
    exit_code: int = 1,
) -> Generator[None, None, None]:
    """Catch common exceptions and display a Rich error panel.

    Usage::

        with error_handler():
            do_something_that_might_fail()

    Parameters
    ----------
    console:
        Rich Console instance.  A default stderr console is created when
        *None*.
    exit_code:
        Process exit code after displaying the error.  Set to ``0`` to
        suppress ``sys.exit``.
    """
    if console is None:
        console = Console(stderr=True)

    try:
        yield
    except KeyboardInterrupt:
        console.print("\n[dim white]Interrupted.[/dim white]")
        raise SystemExit(130) from None
    except MSCException as exc:
        panel = format_error(exc, suggestion=exc.suggestion)
        console.print(panel)
        if exit_code:
            sys.exit(exit_code)
    except Exception as exc:  # noqa: BLE001
        panel = format_error(exc)
        console.print(panel)
        if exit_code:
            sys.exit(exit_code)
