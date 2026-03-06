"""
Campaign runner — launches pipeline stages as subprocesses.

Key responsibilities:
  1. Build enriched task prompts (base task + prior-stage context).
  2. Resolve the workspace directory for the new stage (either fresh or
     --resume an existing workspace from a prior stage that feeds this one).
  3. Launch launch_multiagent.py as a subprocess and return the Popen handle.
  4. Write PID to campaign_dir/pids/<stage_id>.pid for later monitoring.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from typing import List, Optional

from .spec import CampaignSpec, Stage
from .status import CampaignStatus


def build_task_prompt(
    stage: Stage,
    spec: CampaignSpec,
    status: CampaignStatus,
    campaign_dir: str,
) -> str:
    """
    Build the full task prompt for a stage.

    If the stage has context_from entries, prepend a block with the markdown
    summaries distilled from each prior stage's memory file:

        --- Context from <stage_id> ---
        [contents of campaign_dir/memory/<stage_id>_summary.md]
        ---

        [original task text]

    If no prior-stage memory files exist yet, the prompt is returned as-is.
    """
    with open(stage.task_file) as f:
        base_task = f.read().strip()

    context_blocks: List[str] = []
    for ctx_id in stage.context_from:
        memory_path = os.path.join(campaign_dir, "memory", f"{ctx_id}_summary.md")
        if os.path.exists(memory_path):
            with open(memory_path) as f:
                content = f.read().strip()
            if content:
                context_blocks.append(
                    f"--- Context from '{ctx_id}' stage ---\n{content}\n---"
                )

    if context_blocks:
        context_header = "\n\n".join(context_blocks)
        return f"{context_header}\n\n{base_task}"
    return base_task


def build_stage_workspace(
    stage: Stage,
    spec: CampaignSpec,
    status: CampaignStatus,
) -> str:
    """
    Determine the workspace directory for the new stage.

    If the stage has context_from dependencies, we --resume the workspace of
    the first listed dependency so the new stage can see its artifacts directly.
    Otherwise, create a fresh workspace under spec.workspace_root.

    Returns an absolute path (the directory will be created if needed).
    """
    if stage.context_from:
        # Use the first listed context source as the resume workspace
        primary_ctx = stage.context_from[0]
        prior_workspace = status.stage_workspace(primary_ctx)
        if prior_workspace and os.path.isdir(prior_workspace):
            return os.path.abspath(prior_workspace)

    # Fresh workspace: workspace_root/<stage_id>/
    workspace = os.path.join(spec.workspace_root, stage.id)
    os.makedirs(workspace, exist_ok=True)
    return os.path.abspath(workspace)


def _extract_flag_value(args: List[str], flag: str) -> Optional[str]:
    """Return the value passed for a CLI flag, supporting both forms:
    --flag value
    --flag=value
    """
    for idx, arg in enumerate(args):
        if arg == flag and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return None


def _port_is_available(host: str, port: int) -> bool:
    """Return True if host:port can be bound right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _find_available_callback_port(
    host: str = "127.0.0.1",
    start_port: int = 5001,
    max_tries: int = 200,
) -> int:
    """Find a free callback port where both p and p+1 are available.

    launch_multiagent binds callback_port and callback_port+1 (HTTP steering),
    so both ports must be free.
    """
    for port in range(start_port, start_port + max_tries):
        if _port_is_available(host, port) and _port_is_available(host, port + 1):
            return port
    raise RuntimeError(
        f"Could not find a free callback port pair on host {host} "
        f"starting at {start_port}."
    )


def launch_stage(
    stage: Stage,
    spec: CampaignSpec,
    status: CampaignStatus,
    campaign_dir: str,
    launcher: Optional[str] = None,
    extra_env: Optional[dict] = None,
) -> subprocess.Popen:
    """
    Launch a pipeline stage as a background subprocess.

    Args:
        stage:        Stage definition from the campaign spec.
        spec:         Full campaign spec.
        status:       Current campaign status (used to resolve workspaces).
        campaign_dir: Directory where campaign_status.json lives.
        launcher:     Path to launch_multiagent.py. Defaults to auto-resolved.
        extra_env:    Optional extra environment variables for the subprocess.

    Returns:
        subprocess.Popen handle. The caller is responsible for writing the PID
        to status and calling write_status().
    """
    launcher = launcher or _find_launcher()
    workspace = build_stage_workspace(stage, spec, status)
    task_prompt = build_task_prompt(stage, spec, status, campaign_dir)

    cmd: List[str] = [
        sys.executable,
        launcher,
        "--task", task_prompt,
        "--resume", workspace,
    ]

    # Pass stage-specific extra args
    stage_args = list(stage.args)
    callback_host = _extract_flag_value(stage_args, "--callback_host") or "127.0.0.1"
    callback_port = _extract_flag_value(stage_args, "--callback_port")
    if callback_port is None:
        chosen_port = _find_available_callback_port(host=callback_host)
        stage_args.extend(["--callback_port", str(chosen_port)])
        print(
            f"[campaign] Stage '{stage.id}' using callback port {chosen_port} "
            f"(HTTP {chosen_port + 1})."
        )

    cmd.extend(stage_args)

    # Write the enriched task to a sidecar file for auditability
    os.makedirs(os.path.join(campaign_dir, "task_prompts"), exist_ok=True)
    task_file = os.path.join(campaign_dir, "task_prompts", f"{stage.id}_task.txt")
    with open(task_file, "w") as f:
        f.write(task_prompt)

    env = {**os.environ, **(extra_env or {})}

    log_dir = os.path.join(campaign_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    stdout_log = open(os.path.join(log_dir, f"{stage.id}_stdout.log"), "a")
    stderr_log = open(os.path.join(log_dir, f"{stage.id}_stderr.log"), "a")

    print(f"[campaign] Launching stage '{stage.id}': {' '.join(cmd[:5])} ...")
    proc = subprocess.Popen(
        cmd,
        stdout=stdout_log,
        stderr=stderr_log,
        env=env,
        cwd=os.path.dirname(launcher),
    )

    # Write PID file
    pid_dir = os.path.join(campaign_dir, "pids")
    os.makedirs(pid_dir, exist_ok=True)
    with open(os.path.join(pid_dir, f"{stage.id}.pid"), "w") as f:
        f.write(str(proc.pid))

    print(f"[campaign] Stage '{stage.id}' started (PID {proc.pid}), workspace: {workspace}")
    return proc


def _find_launcher() -> str:
    """Find launch_multiagent.py relative to this package."""
    # Walk up from consortium/campaign/ to repo root
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        here = os.path.dirname(here)
        candidate = os.path.join(here, "launch_multiagent.py")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not locate launch_multiagent.py. "
        "Pass launcher= explicitly to launch_stage()."
    )
