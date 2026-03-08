"""
Track merge node for the parallel theory/experiment workflow.
"""

from __future__ import annotations

from typing import Any, Callable


def _summarize_output(state: dict, agent_name: str) -> str:
    output = str(state.get("agent_outputs", {}).get(agent_name, "")).strip()
    if not output:
        return f"- {agent_name}: no direct agent output recorded"
    compact = output.replace("\n", " ")
    if len(compact) > 300:
        compact = compact[:297] + "..."
    return f"- {agent_name}: {compact}"


def build_node(workspace_dir: str | None = None, **cfg: Any) -> Callable:
    def track_merge_node(state: dict) -> dict:
        track_decomposition = state.get("track_decomposition") or {}
        theory_questions = track_decomposition.get("theory_questions") or []
        empirical_questions = track_decomposition.get("empirical_questions") or []

        theory_status = state.get("theory_track_status")
        experiment_status = state.get("experiment_track_status")

        if theory_questions and not theory_status:
            theory_status = "completed"
        if empirical_questions and not experiment_status:
            experiment_status = "completed"

        notes = [
            "Synthesize the completed theory and empirical tracks before final interpretation.",
            f"Workspace: {workspace_dir or state.get('workspace_dir', '.')}",
            "Inspect math artifacts in `math_workspace/`, experimental artifacts in `experiment_workspace/` and `experiment_runs/`,",
            "and write a focused synthesis literature review that explains contradictions, stronger baselines, and interpretation risks.",
            "Use the prior agent outputs only as signposts; ground conclusions in workspace artifacts.",
            "",
            "Observed outputs:",
            _summarize_output(state, "proof_transcription_agent"),
            _summarize_output(state, "experiment_transcription_agent"),
            _summarize_output(state, "experiment_verification_agent"),
        ]

        return {
            "theory_track_status": theory_status,
            "experiment_track_status": experiment_status,
            "agent_task": "\n".join(notes),
        }

    track_merge_node.__name__ = "track_merge"
    return track_merge_node
