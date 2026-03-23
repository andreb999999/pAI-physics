"""Streaming display for msc run — rich live TUI showing pipeline progress."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections import deque
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.style import Style
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Known pipeline stages (order matters — maps to progress bar advancement)
# ---------------------------------------------------------------------------

# V2 pre-track
_V2_PRE_TRACK = [
    "persona_council",
    "literature_review_agent",
    "brainstorm_agent",
    "formalize_goals_entry",
    "formalize_goals_agent",
    "research_plan_writeup_agent",
]

# Math track (optional, included when --math is on)
_MATH_TRACK = [
    "math_literature_agent",
    "math_proposer_agent",
    "math_prover_agent",
    "math_rigorous_verifier_agent",
    "math_empirical_verifier_agent",
    "proof_transcription_agent",
]

# Experiment track
_EXPERIMENT_TRACK = [
    "experiment_literature_agent",
    "experiment_design_agent",
    "experimentation_agent",
    "experiment_verification_agent",
    "experiment_transcription_agent",
]

# V2 post-track
_V2_POST_TRACK = [
    "formalize_results_agent",
    "resource_preparation_agent",
    "writeup_agent",
    "proofreading_agent",
    "reviewer_agent",
]

# Legacy/V1 names that might appear in output (for backward compat detection)
_LEGACY_NAMES = [
    "manager",
    "research_planner",
    "literature_review",
    "ideation",
    "theory_verification",
    "experiment_design",
    "run_experiment",
    "results_analysis",
    "reviewer",
    "duality_check",
    "writeup",
    "editorial_review",
    "final_compilation",
]

# Combined lookup set for fast matching
_ALL_KNOWN_STAGES: set[str] = set(
    _V2_PRE_TRACK
    + _MATH_TRACK
    + _EXPERIMENT_TRACK
    + _V2_POST_TRACK
    + _LEGACY_NAMES
)

# Ordered list for default progress (no math). Caller can override total_stages.
_DEFAULT_STAGE_ORDER: list[str] = (
    _V2_PRE_TRACK + _EXPERIMENT_TRACK + _V2_POST_TRACK
)

# ---------------------------------------------------------------------------
# Theme constants (blue / gray / white)
# ---------------------------------------------------------------------------

BLUE = Style(color="dodger_blue2")
BLUE_BOLD = Style(color="dodger_blue2", bold=True)
GRAY = Style(color="grey62")
GRAY_DIM = Style(color="grey42")
WHITE = Style(color="white")
WHITE_BOLD = Style(color="white", bold=True)
WARN_BUDGET = Style(color="white", bold=True)
CRIT_BUDGET = Style(color="white", bgcolor="red", bold=True)
BORDER_STYLE = "grey42"

# ---------------------------------------------------------------------------
# Regex patterns for parsing pipeline output
# ---------------------------------------------------------------------------

# Stage transitions: "--- Stage: <name>" or "=== <name> ===" or bare agent name
_RE_STAGE_HEADER = re.compile(
    r"(?:---\s*Stage[:\s]+|===\s*)(\w+)", re.IGNORECASE
)
# Bare agent name at start of line (e.g. "literature_review_agent ...")
_RE_BARE_AGENT = re.compile(
    r"^(" + "|".join(re.escape(s) for s in sorted(_ALL_KNOWN_STAGES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
# Budget: "$3.42" or "cost: $3.42" or "budget: $3.42/$25"
_RE_COST = re.compile(r"\$(\d+(?:\.\d+)?)")
_RE_BUDGET_LINE = re.compile(
    r"(?:cost|budget|spent|total_cost)[:\s]*\$?(\d+(?:\.\d+)?)", re.IGNORECASE
)

# LLM waiting indicators
_RE_LLM_WAIT = re.compile(
    r"(?:calling|invoking|requesting|waiting for)\s+(?:llm|model|api|gpt|claude|gemini)",
    re.IGNORECASE,
)


def _human_stage_name(raw: str) -> str:
    """Turn 'literature_review_agent' into 'Literature Review'."""
    name = raw.replace("_agent", "").replace("_entry", "")
    return name.replace("_", " ").title()


# ---------------------------------------------------------------------------
# StreamingDisplay
# ---------------------------------------------------------------------------


class StreamingDisplay:
    """Wraps a subprocess with a rich live TUI showing progress, output, and budget."""

    def __init__(
        self,
        budget: int = 25,
        total_stages: int | None = None,
    ) -> None:
        self._budget_limit: float = float(budget)
        self._total_stages: int = total_stages or len(_DEFAULT_STAGE_ORDER)
        self._current_stage: int = 0
        self._current_stage_name: str = "Initializing"
        self._budget_spent: float = 0.0
        self._lines: deque[str] = deque(maxlen=15)
        self._start_time: float = 0.0
        self._waiting_for_llm: bool = False
        self._seen_stages: set[str] = set()

    # ---- output parsing ---------------------------------------------------

    def _parse_line(self, line: str) -> None:
        """Extract stage transitions, budget info, and LLM wait state."""
        stripped = line.strip()
        if not stripped:
            return

        # Stage header pattern
        m = _RE_STAGE_HEADER.search(stripped)
        if m:
            self._advance_stage(m.group(1))
            return

        # Bare agent name at start of line
        m = _RE_BARE_AGENT.match(stripped)
        if m:
            self._advance_stage(m.group(1))

        # Budget / cost extraction — take the last dollar amount on a budget line
        m = _RE_BUDGET_LINE.search(stripped)
        if m:
            try:
                self._budget_spent = float(m.group(1))
            except ValueError:
                pass
        else:
            # Fallback: any dollar amount in a line containing "cost" or "budget"
            lower = stripped.lower()
            if "cost" in lower or "budget" in lower or "spent" in lower:
                amounts = _RE_COST.findall(stripped)
                if amounts:
                    try:
                        self._budget_spent = float(amounts[-1])
                    except ValueError:
                        pass

        # LLM wait detection
        if _RE_LLM_WAIT.search(stripped):
            self._waiting_for_llm = True
        else:
            self._waiting_for_llm = False

    def _advance_stage(self, name: str) -> None:
        key = name.lower().strip()
        if key in self._seen_stages:
            # Already visited — just update the display name
            self._current_stage_name = _human_stage_name(key)
            return
        if key in _ALL_KNOWN_STAGES or key in {s.lower() for s in _ALL_KNOWN_STAGES}:
            self._seen_stages.add(key)
            self._current_stage = min(self._current_stage + 1, self._total_stages)
            self._current_stage_name = _human_stage_name(key)

    # ---- layout construction ----------------------------------------------

    def _elapsed_str(self) -> str:
        elapsed = time.time() - self._start_time
        mins, secs = divmod(int(elapsed), 60)
        hours, mins = divmod(mins, 60)
        if hours:
            return f"{hours}h{mins:02d}m{secs:02d}s"
        return f"{mins}m{secs:02d}s"

    def _build_progress_panel(self, progress: Progress) -> Panel:
        return Panel(
            progress,
            title="[dodger_blue2]Progress[/dodger_blue2]",
            border_style=BORDER_STYLE,
            padding=(0, 1),
        )

    def _build_output_panel(self) -> Panel:
        output_text = Text()
        for line in self._lines:
            output_text.append(line + "\n", style=WHITE)
        return Panel(
            output_text,
            title="[dodger_blue2]Output[/dodger_blue2]",
            border_style=BORDER_STYLE,
            padding=(0, 1),
            height=19,
        )

    def _build_status_panel(self) -> Panel:
        table = Table.grid(padding=(0, 3))
        table.add_column(justify="left")
        table.add_column(justify="left")
        table.add_column(justify="right")

        # Budget styling
        ratio = self._budget_spent / self._budget_limit if self._budget_limit else 0
        if ratio < 0.50:
            budget_style = BLUE
        elif ratio < 0.85:
            budget_style = WARN_BUDGET
        else:
            budget_style = CRIT_BUDGET

        budget_text = Text(
            f"Budget: ${self._budget_spent:.2f}/${self._budget_limit:.0f}",
            style=budget_style,
        )

        elapsed_text = Text(f"Elapsed: {self._elapsed_str()}", style=GRAY)

        spinner_text = Text("")
        if self._waiting_for_llm:
            spinner_text = Text(" [waiting for LLM]", style=BLUE)

        table.add_row(budget_text, spinner_text, elapsed_text)

        return Panel(
            table,
            title="[dodger_blue2]Status[/dodger_blue2]",
            border_style=BORDER_STYLE,
            padding=(0, 1),
        )

    def _build_layout(self, progress: Progress) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="progress", size=3),
            Layout(name="output"),
            Layout(name="status", size=3),
        )
        layout["progress"].update(self._build_progress_panel(progress))
        layout["output"].update(self._build_output_panel())
        layout["status"].update(self._build_status_panel())
        return layout

    # ---- main entry points ------------------------------------------------

    def run(self, argv: list[str], env: dict[str, str]) -> int:
        """Run subprocess with live streaming display. Returns exit code."""
        self._start_time = time.time()

        # Progress bar with blue/gray theme
        progress = Progress(
            SpinnerColumn(style=BLUE),
            TextColumn("[dodger_blue2]{task.description}[/dodger_blue2]"),
            BarColumn(
                bar_width=40,
                complete_style=BLUE,
                finished_style=BLUE_BOLD,
                style=GRAY_DIM,
            ),
            TextColumn("[grey62]{task.completed}/{task.total}[/grey62]"),
            TimeElapsedColumn(),
            expand=True,
        )
        task_id: TaskID = progress.add_task(
            f"Stage 0/{self._total_stages}: Initializing",
            total=self._total_stages,
            completed=0,
        )

        console = Console()
        proc: Optional[subprocess.Popen] = None

        try:
            proc = subprocess.Popen(
                argv,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            with Live(
                self._build_layout(progress),
                console=console,
                refresh_per_second=4,
                transient=False,
            ) as live:
                assert proc.stdout is not None
                for raw_line in iter(proc.stdout.readline, ""):
                    line = raw_line.rstrip("\n\r")
                    self._lines.append(line)
                    self._parse_line(line)

                    # Update progress bar
                    progress.update(
                        task_id,
                        completed=self._current_stage,
                        description=(
                            f"Stage {self._current_stage}/{self._total_stages}: "
                            f"{self._current_stage_name}"
                        ),
                    )
                    live.update(self._build_layout(progress))

                # Process ended — drain
                proc.stdout.close()
                exit_code = proc.wait()

                # Final update
                if exit_code == 0:
                    progress.update(
                        task_id,
                        completed=self._total_stages,
                        description=f"Stage {self._total_stages}/{self._total_stages}: Complete",
                    )
                live.update(self._build_layout(progress))

            return exit_code

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Interrupted.[/bold yellow]")
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            # Print partial summary
            self._print_summary(console, interrupted=True)
            return 130  # Standard SIGINT exit code

        except Exception as exc:
            console.print(f"\n[red]Display error:[/red] {exc}")
            if proc is not None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                return proc.returncode or 1
            return 1

    def _print_summary(self, console: Console, interrupted: bool = False) -> None:
        """Print a brief summary after run completes or is interrupted."""
        label = "Partial results" if interrupted else "Run complete"
        style = "yellow" if interrupted else "green"
        console.print(
            Panel(
                f"Stages completed: {self._current_stage}/{self._total_stages}\n"
                f"Budget spent: ${self._budget_spent:.2f}/${self._budget_limit:.0f}\n"
                f"Elapsed: {self._elapsed_str()}",
                title=f"[{style}]{label}[/{style}]",
                border_style=BORDER_STYLE,
            )
        )


# ---------------------------------------------------------------------------
# Simple fallback (non-TTY / --no-stream)
# ---------------------------------------------------------------------------


def run_simple(argv: list[str], env: dict[str, str]) -> int:
    """Fallback runner — plain subprocess.run() for non-TTY or --no-stream."""
    try:
        proc = subprocess.run(argv, env=env)
        return proc.returncode
    except FileNotFoundError:
        Console().print(
            "[bold white on red] Error [/] 'consortium' command not found. "
            "Install it with: [bold]pip install -e /path/to/PoggioAI/MSc[/]"
        )
        return 1
    except KeyboardInterrupt:
        Console().print("\n[bold yellow]Interrupted.[/bold yellow]")
        return 130
