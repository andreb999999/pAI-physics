"""
Tinker API experiment runner for PoggioAI/MSc.

Submits training/fine-tuning jobs to the Tinker API (thinkingmachines.ai/tinker)
instead of running experiments locally or via SLURM.

Tinker provides four core operations:
  - forward_backward: Execute forward and backward passes, accumulate gradients
  - optim_step:       Update weights based on accumulated gradients
  - sample:           Generate tokens for evaluation or RL tasks
  - save_state:       Persist training checkpoints for resumption

Status: STUB — ready to wire up once Tinker API key/SDK access is available.
See: https://thinkingmachines.ai/tinker/

Authentication:
  Set TINKER_API_KEY in your .env file or environment.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional, Tuple


def is_tinker_enabled() -> bool:
    """Check if Tinker API experiment execution is enabled."""
    mode = os.environ.get("CONSORTIUM_MODE", "local")
    return mode == "tinker" and bool(os.environ.get("TINKER_API_KEY"))


def _get_api_key() -> str:
    """Retrieve and validate the Tinker API key."""
    key = os.environ.get("TINKER_API_KEY", "")
    if not key:
        raise RuntimeError(
            "TINKER_API_KEY is not set. "
            "Sign up at https://auth.thinkingmachines.ai/sign-up "
            "and set TINKER_API_KEY in your .env file."
        )
    return key


def _init_client(api_key: str) -> Any:
    """Initialize the Tinker API client.

    TODO: Replace with actual Tinker SDK client initialization.
    The Tinker API base URL and client library will be documented at
    https://thinkingmachines.ai/tinker/ once SDK access is available.
    """
    # TODO: Install and import tinker SDK
    # from tinker import TinkerClient
    # return TinkerClient(api_key=api_key)
    raise NotImplementedError(
        "Tinker API client not yet implemented. "
        "This is a stub awaiting SDK access. "
        "See https://thinkingmachines.ai/tinker/ for API documentation. "
        "To run experiments now, use --mode local (CPU) or --mode hpc (SLURM)."
    )


def submit_tinker_job(
    idea_data: dict,
    run_dir: str,
    config: dict,
    job_name: str = "poggioaimsc_experiment",
    model_name: Optional[str] = None,
) -> dict:
    """Submit a training experiment to the Tinker API.

    Translates an AI-Scientist-v2 experiment idea into Tinker API calls.
    The Tinker API handles infrastructure, scheduling, and GPU management.

    Args:
        idea_data: Research idea dict with Name, Title, Experiments, etc.
        run_dir: Local directory for storing results and logs.
        config: Experiment config (from bfts_config.yaml).
        job_name: Human-readable job identifier.
        model_name: Base model to fine-tune (e.g., "Qwen/Qwen2.5-1.5B").
                     If None, inferred from idea_data.

    Returns:
        Dict with keys: job_id, status, results_dir, elapsed_seconds.

    Raises:
        NotImplementedError: Until Tinker SDK is integrated.
        RuntimeError: If API key is missing or API call fails.
    """
    api_key = _get_api_key()
    client = _init_client(api_key)  # noqa: F841 — will be used once implemented

    # TODO: Implement the following steps:
    #
    # 1. Parse idea_data to determine:
    #    - Base model (from idea_data["Experiments"] or model_name param)
    #    - Dataset / training data source
    #    - Training hyperparameters (lr, epochs, batch size)
    #    - LoRA rank/alpha if applicable
    #
    # 2. Create training job via Tinker API:
    #    job = client.create_job(
    #        model=model_name,
    #        dataset=dataset_config,
    #        training_config={...},
    #    )
    #
    # 3. Execute training loop using Tinker's core operations:
    #    for step in range(num_steps):
    #        # Forward + backward pass
    #        result = client.forward_backward(
    #            job_id=job.id,
    #            batch=next_batch,
    #        )
    #
    #        # Update weights
    #        client.optim_step(job_id=job.id)
    #
    #        # Periodic evaluation via sampling
    #        if step % eval_interval == 0:
    #            eval_output = client.sample(
    #                job_id=job.id,
    #                prompts=eval_prompts,
    #            )
    #
    #        # Checkpoint
    #        if step % save_interval == 0:
    #            client.save_state(job_id=job.id, tag=f"step_{step}")
    #
    # 4. Download results:
    #    client.download_checkpoint(job_id=job.id, dest=run_dir)
    #    client.download_logs(job_id=job.id, dest=run_dir)
    #
    # 5. Return summary dict

    raise NotImplementedError(
        "Tinker job submission not yet implemented. See TODOs above."
    )


def poll_tinker_job(
    job_id: str,
    max_wait: int = 7200,
    poll_interval: int = 30,
) -> Tuple[str, int]:
    """Poll Tinker API for job completion.

    TODO: Implement once Tinker SDK is available.

    Args:
        job_id: Tinker job identifier.
        max_wait: Maximum wait time in seconds.
        poll_interval: Seconds between status checks.

    Returns:
        (final_status, elapsed_seconds)
    """
    # TODO: Implement polling loop
    # client = _init_client(_get_api_key())
    # elapsed = 0
    # while elapsed < max_wait:
    #     status = client.get_job_status(job_id)
    #     if status.is_terminal:
    #         return status.state, elapsed
    #     time.sleep(poll_interval)
    #     elapsed += poll_interval
    # return "POLL_TIMEOUT", elapsed

    raise NotImplementedError("Tinker job polling not yet implemented.")
