"""msc resume — resume a previous research run from checkpoint."""

from __future__ import annotations

import subprocess
from pathlib import Path
import os

import click
from rich.console import Console

from consortium.cli.core.env_manager import inject_runtime_env
from consortium.cli.core.paths import build_runner_argv, find_project_root, find_results_dir

console = Console()


def _should_use_repo_env(project_root: Path | None) -> bool:
    if project_root is None:
        return False
    cwd = Path.cwd().resolve()
    project_root = project_root.resolve()
    return cwd == project_root or project_root in cwd.parents


def _find_latest_run() -> Path | None:
    """Find the most recent run directory."""
    results = find_results_dir()
    if results is None or not results.is_dir():
        return None
    run_dirs = sorted(
        [d for d in results.iterdir() if d.is_dir() and d.name.startswith("consortium_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return run_dirs[0] if run_dirs else None


@click.command()
@click.argument("run_id", required=False)
@click.option("--start-from", type=str, default=None, help="Resume from a specific stage.")
@click.option("--task", type=str, default=None, help="Override the task for the resumed run.")
@click.pass_context
def resume(
    ctx: click.Context,
    run_id: str | None,
    start_from: str | None,
    task: str | None,
) -> None:
    """Resume a previous run from its checkpoint.

    \b
    Examples:
      msc resume                                       # Resume latest run
      msc resume consortium_20260322_143022_task       # Resume specific run
      msc resume --start-from experiment               # Resume from a specific stage
    """
    config_dir = ctx.obj.get("config_dir")
    project_root = find_project_root()
    allow_repo_env = _should_use_repo_env(project_root)

    # Resolve run directory
    if run_id:
        results_dir = find_results_dir()
        run_dir = (results_dir / run_id) if results_dir else Path(run_id)
        if not run_dir.is_dir():
            # Try as absolute path
            run_dir = Path(run_id)
        if not run_dir.is_dir():
            console.print(f"[bold white on red] Error [/] Run directory not found: {run_id}")
            raise SystemExit(1)
    else:
        run_dir = _find_latest_run()
        if not run_dir:
            console.print("[bold white on red] Error [/] No previous runs found in results/")
            raise SystemExit(1)

    console.print(f"Resuming from [bold]{run_dir.name}[/]")

    # Build argv
    argv = build_runner_argv(["--resume", str(run_dir)])
    if start_from:
        argv.extend(["--start-from-stage", start_from])
    if task:
        argv.extend(["--task", task])

    # Inject API keys
    inject_runtime_env(
        config_dir_override=config_dir,
        repo_root=project_root,
        allow_repo_env=allow_repo_env,
    )
    env = dict(os.environ)
    env["CONSORTIUM_USE_REPO_ENV"] = "1" if allow_repo_env else "0"
    if project_root is not None:
        env["CONSORTIUM_PROJECT_ROOT"] = str(project_root)

    try:
        proc = subprocess.run(argv, env=env)
        raise SystemExit(proc.returncode)
    except FileNotFoundError:
        console.print("[bold white on red] Error [/] Could not launch the consortium runner module.")
        raise SystemExit(1)
