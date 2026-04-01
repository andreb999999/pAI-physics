"""msc logs — tail pipeline output logs."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# ── Theme: blue, gray, white ────────────────────────────────────────
_BLUE = "bold blue"
_GRAY = "dim white"
_WHITE = "white"

from consortium.cli.core.paths import find_results_dir as _find_results_dir

console = Console()


def _find_most_recent_run(results_dir: Path) -> Path | None:
    """Return the most recently modified run directory."""
    run_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and not d.name.startswith(".")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


def _find_log_files(run_dir: Path, stage: str | None = None) -> list[Path]:
    """Find log files in a run directory, optionally filtered by stage.

    Searches for:
      - results/<run_id>/logs/*.log
      - results/<run_id>/consortium_*.out
      - results/<run_id>/*.log
    """
    log_files: list[Path] = []

    # Logs subdirectory
    logs_subdir = run_dir / "logs"
    if logs_subdir.is_dir():
        for f in sorted(logs_subdir.iterdir()):
            if f.is_file() and f.suffix in (".log", ".out", ".txt"):
                if stage and stage.lower() not in f.stem.lower():
                    continue
                log_files.append(f)

    # consortium_*.out in root of run dir
    for f in sorted(run_dir.glob("consortium_*.out")):
        if f.is_file():
            if stage and stage.lower() not in f.stem.lower():
                continue
            log_files.append(f)

    # Other .log files in run dir root
    for f in sorted(run_dir.glob("*.log")):
        if f.is_file() and f not in log_files:
            if stage and stage.lower() not in f.stem.lower():
                continue
            log_files.append(f)

    return log_files


def _tail_lines(path: Path, n: int) -> list[str]:
    """Read the last n lines from a file efficiently."""
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
        return lines[-n:] if len(lines) > n else lines
    except OSError:
        return []


def _print_log_header(run_id: str, log_path: Path, follow: bool) -> None:
    """Print a styled header showing which logs are being displayed."""
    mode = "following" if follow else "showing"
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                f"[{_WHITE}]{mode} logs for[/] [{_BLUE}]{run_id}[/{_BLUE}]\n"
                f"[{_GRAY}]{log_path}[/{_GRAY}]"
            ),
            title="Logs",
            title_align="left",
            border_style=_GRAY,
            style=_WHITE,
            padding=(0, 2),
        )
    )
    console.print()


def _print_lines(lines: list[str]) -> None:
    """Print log lines with consistent styling."""
    for line in lines:
        stripped = line.rstrip("\n")
        # Apply subtle coloring: lines with ERROR/WARN get highlighted
        if "ERROR" in stripped or "error" in stripped:
            console.print(Text(stripped, style="bold red"))
        elif "WARN" in stripped or "warn" in stripped:
            console.print(Text(stripped, style="yellow"))
        elif "INFO" in stripped:
            console.print(Text(stripped, style=_WHITE))
        else:
            console.print(Text(stripped, style=_GRAY))


def _follow_file(path: Path, initial_lines: int) -> None:
    """Continuously tail a file, printing new lines every 0.5s.

    Handles file truncation (rotation) gracefully.
    """
    try:
        with open(path, "r", errors="replace") as f:
            # Read and print initial tail
            all_lines = f.readlines()
            start = max(0, len(all_lines) - initial_lines)
            _print_lines(all_lines[start:])

            # Now follow
            console.print(Text("--- following (Ctrl+C to stop) ---", style=_GRAY))
            while True:
                line = f.readline()
                if line:
                    _print_lines([line])
                else:
                    # Check if file was truncated (rotated)
                    try:
                        current_pos = f.tell()
                        file_size = path.stat().st_size
                        if file_size < current_pos:
                            # File was truncated, reopen from start
                            f.seek(0)
                            console.print(
                                Text("--- log rotated, reading from start ---", style=_GRAY)
                            )
                            continue
                    except OSError:
                        pass
                    time.sleep(0.5)
    except KeyboardInterrupt:
        console.print()
        console.print(Text("Stopped.", style=_GRAY))


@click.command()
@click.argument("run_id", required=False)
@click.option("--follow", "-f", is_flag=True, help="Follow log output in real time.")
@click.option("--lines", "-n", type=int, default=50, help="Number of lines to show (default: 50).")
@click.option("--stage", type=str, default=None, help="Filter logs by pipeline stage name.")
@click.option("--results-dir", type=click.Path(exists=True), default=None,
              help="Override the results directory path.")
def logs(
    run_id: str | None,
    follow: bool,
    lines: int,
    stage: str | None,
    results_dir: str | None,
) -> None:
    """View pipeline output logs.

    \b
    Examples:
      msc logs                       # Show last 50 lines of most recent run
      msc logs -f                    # Follow the latest run's logs live
      msc logs my-run-id -n 100     # Show last 100 lines of a specific run
      msc logs --stage ideation      # Show only ideation stage logs
    """
    rdir = Path(results_dir) if results_dir else _find_results_dir()
    if not rdir:
        console.print("[yellow]No results/ directory found.[/] Run a pipeline first.")
        raise SystemExit(1)

    # Resolve run directory
    if run_id:
        run_dir = rdir / run_id
        if not run_dir.is_dir():
            # Try partial match
            matches = [
                d for d in rdir.iterdir()
                if d.is_dir() and run_id.lower() in d.name.lower()
            ]
            if len(matches) == 1:
                run_dir = matches[0]
                run_id = run_dir.name
            elif len(matches) > 1:
                console.print(f"[yellow]Ambiguous run ID '{run_id}'. Matches:[/]")
                for m in matches:
                    console.print(f"  {m.name}")
                raise SystemExit(1)
            else:
                console.print(f"[red]Run directory not found:[/] {run_id}")
                raise SystemExit(1)
    else:
        run_dir_candidate = _find_most_recent_run(rdir)
        if not run_dir_candidate:
            console.print("[dim]No runs found in results/ directory.[/]")
            raise SystemExit(1)
        run_dir = run_dir_candidate
        run_id = run_dir.name

    # Find log files
    log_files = _find_log_files(run_dir, stage=stage)

    if not log_files:
        stage_hint = f" for stage '{stage}'" if stage else ""
        console.print(
            Panel(
                Text.from_markup(
                    f"[dim]No logs found for run[/] [{_BLUE}]{run_id}[/{_BLUE}]{stage_hint}.\n\n"
                    f"[{_GRAY}]Is the pipeline running? Check with:[/{_GRAY}] "
                    f"[{_BLUE}]msc status[/{_BLUE}]"
                ),
                title="No Logs",
                title_align="left",
                border_style=_GRAY,
                padding=(1, 2),
            )
        )
        raise SystemExit(1)

    # Use the most recent log file (by mtime)
    target_log = max(log_files, key=lambda f: f.stat().st_mtime)

    # If multiple log files found and stage not specified, list them
    if len(log_files) > 1 and not stage:
        console.print(Text(f"  Found {len(log_files)} log files, showing newest:", style=_GRAY))
        for lf in log_files:
            marker = " *" if lf == target_log else ""
            console.print(Text(f"    {lf.name}{marker}", style=_GRAY))

    _print_log_header(run_id, target_log, follow)

    if follow:
        _follow_file(target_log, lines)
    else:
        tail = _tail_lines(target_log, lines)
        if tail:
            _print_lines(tail)
        else:
            console.print(Text("(empty log file)", style=_GRAY))

    console.print()
