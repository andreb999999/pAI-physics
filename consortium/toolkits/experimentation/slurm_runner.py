"""
SLURM job runner for experiment execution on HPC clusters.

Submits experiment scripts as SLURM batch jobs and polls for completion.
Used by RunExperimentTool when CONSORTIUM_SLURM_ENABLED=1.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from typing import Optional, Tuple

import yaml

# Polling configuration
_POLL_INTERVAL = 30       # seconds between sacct checks
_MAX_POLL_TIME = 7 * 3600  # 7 hours default (generous for pi_tpoggio 7-day limit)

# SLURM terminal states (job is done, no more polling needed)
_TERMINAL_STATES = frozenset({
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
    "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY",
    "CANCELLED+", "DEADLINE",
})


def is_slurm_enabled() -> bool:
    """Check if SLURM-based experiment execution is enabled."""
    return os.environ.get("CONSORTIUM_SLURM_ENABLED", "0") in ("1", "true", "yes")


def load_engaging_config() -> dict:
    """Load the Engaging cluster configuration from engaging_config.yaml."""
    config_path = os.environ.get("ENGAGING_CONFIG")
    if not config_path:
        # Walk up from this file to find repo root
        here = os.path.dirname(os.path.abspath(__file__))
        for _ in range(5):
            candidate = os.path.join(here, "engaging_config.yaml")
            if os.path.exists(candidate):
                config_path = candidate
                break
            here = os.path.dirname(here)

    if config_path and os.path.exists(config_path):
        from ...workflow_utils import expand_env_vars
        with open(config_path) as f:
            raw = f.read()
        # Expand ${VAR:-default} patterns before parsing YAML
        raw = expand_env_vars(raw)
        return yaml.safe_load(raw) or {}
    return {}


def _get_cluster_config(config: dict) -> dict:
    """Extract the cluster section from the config."""
    return config.get("cluster", {})


def submit_experiment_job(
    cmd: list[str],
    run_dir: str,
    job_name: str = "consortium_exp",
    partition: Optional[str] = None,
    time_limit: Optional[str] = None,
    gpus: Optional[str] = None,
    mem: Optional[str] = None,
    cpus: Optional[int] = None,
) -> Tuple[int, str]:
    """
    Submit an experiment as a SLURM batch job.

    Generates an sbatch script that sets up the environment (modules, conda)
    and runs the provided command, then submits it via sbatch.

    Args:
        cmd: Command to execute (e.g., ["python", "launch_scientist_bfts.py", ...])
        run_dir: Working directory for the experiment
        job_name: SLURM job name
        partition: Override SLURM partition (default from config)
        time_limit: Override time limit (default from config)
        gpus: Override GPU GRES string (default from config)
        mem: Override memory (default from config)
        cpus: Override CPU count (default from config)

    Returns:
        (job_id, sbatch_script_path) tuple

    Raises:
        RuntimeError: If sbatch submission fails or job ID cannot be parsed
    """
    config = load_engaging_config()
    cluster = _get_cluster_config(config)
    exp_config = cluster.get("experiment_gpu", {})

    # Resolve parameters from config or defaults
    partition = partition or exp_config.get("partition", "pi_tpoggio")
    time_limit = time_limit or exp_config.get("time", "7-00:00:00")
    gpus = gpus or exp_config.get("gres", "gpu:a100:1")
    mem = mem or exp_config.get("mem", "64G")
    cpus = cpus or exp_config.get("cpus", 8)

    # Conda settings — prefer config, fall back to env vars
    conda_init = cluster.get("conda_init_script") or os.environ.get("CONDA_INIT_SCRIPT", "")
    conda_env_prefix = cluster.get("conda_env_prefix") or os.environ.get("CONDA_PREFIX", "")
    conda_module = cluster.get("modules", {}).get("conda", "miniforge/25.11.0-0")
    cuda_module = cluster.get("modules", {}).get("cuda", "cuda/12.4.0")
    cudnn_module = cluster.get("modules", {}).get("cudnn", "cudnn/9.8.0.87-cuda12")

    # Create log directory for SLURM output
    log_dir = os.path.join(run_dir, "slurm_logs")
    os.makedirs(log_dir, exist_ok=True)

    # Build the command string (quote args with spaces)
    cmd_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)

    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --gres={gpus}
#SBATCH --time={time_limit}
#SBATCH --mem={mem}
#SBATCH --output={log_dir}/exp_%j.out
#SBATCH --error={log_dir}/exp_%j.err

echo "========================================"
echo "Consortium Experiment Job"
echo "========================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Started:   $(date)"
echo "Run dir:   {run_dir}"
echo "========================================"

# --- Environment setup ---
module load {conda_module}
module load {cuda_module}
module load {cudnn_module}

source {conda_init}
conda deactivate 2>/dev/null || true
conda activate {conda_env_prefix}

echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi

# --- Run experiment ---
cd {run_dir}
echo "Running: {cmd_str}"
{cmd_str}
EXIT_CODE=$?

echo "========================================"
echo "Experiment completed with exit code $EXIT_CODE at $(date)"
echo "========================================"
exit $EXIT_CODE
"""

    script_path = os.path.join(run_dir, "experiment_job.sh")
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

    # Submit the job
    result = subprocess.run(
        ["sbatch", script_path],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch submission failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # Parse "Submitted batch job 12345"
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout}")

    job_id = int(match.group(1))
    print(f"[slurm] Submitted experiment job {job_id} to {partition}")
    return job_id, script_path


def poll_job_completion(
    job_id: int,
    max_wait: int = _MAX_POLL_TIME,
    poll_interval: int = _POLL_INTERVAL,
) -> Tuple[str, int]:
    """
    Poll SLURM for job completion using sacct.

    Args:
        job_id: SLURM job ID to monitor
        max_wait: Maximum wait time in seconds
        poll_interval: Seconds between polls

    Returns:
        (final_state, elapsed_seconds)
        final_state is one of: COMPLETED, FAILED, TIMEOUT, CANCELLED, etc.
    """
    elapsed = 0
    last_state = "UNKNOWN"

    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval

        try:
            result = subprocess.run(
                [
                    "sacct", "-j", str(job_id),
                    "--format=State", "--noheader", "--parsable2",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

        if result.returncode != 0:
            continue

        # Parse states — sacct may return multiple lines (job + job steps)
        states = [
            line.strip() for line in result.stdout.strip().split("\n")
            if line.strip()
        ]
        if not states:
            continue

        # The first non-empty state is the overall job state
        state = states[0].rstrip("+")  # "CANCELLED+" → "CANCELLED"
        last_state = state

        # Check for terminal state
        if state in _TERMINAL_STATES or states[0] in _TERMINAL_STATES:
            return states[0], elapsed

        # Periodic progress logging
        if elapsed % (poll_interval * 10) < poll_interval:
            print(f"  [slurm] Job {job_id}: {state} ({elapsed}s elapsed)")

    return f"POLL_TIMEOUT({last_state})", elapsed


def is_slurm_job_alive(job_id: int) -> bool:
    """
    Check if a SLURM job is still running or pending.

    Uses squeue for fast lookup (only lists active jobs).
    """
    if not job_id or job_id <= 0:
        return False
    try:
        result = subprocess.run(
            ["squeue", "-j", str(job_id), "--noheader", "--format=%T"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        state = result.stdout.strip()
        return state in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED")
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False
