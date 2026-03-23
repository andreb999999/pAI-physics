"""
Track merge node for the parallel theory/experiment workflow.

Reads structured summary files (theory_track_summary.json and
experiment_track_summary.json) produced by the transcription agents,
rather than relying on truncated agent output strings.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable


def _read_json_safe(path: str) -> dict:
    """Read a JSON file, returning {} on any error."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def build_node(workspace_dir: str | None = None, **cfg: Any) -> Callable:
    def track_merge_node(state: dict) -> dict:
        track_decomposition = state.get("track_decomposition") or {}
        theory_questions = track_decomposition.get("theory_questions") or []
        empirical_questions = track_decomposition.get("empirical_questions") or []

        ws = workspace_dir or state.get("workspace_dir", ".")
        paper_ws = os.path.join(ws, "paper_workspace")

        # Read structured summaries instead of truncated agent output strings
        theory_summary = _read_json_safe(
            os.path.join(paper_ws, "theory_track_summary.json")
        )
        experiment_summary = _read_json_safe(
            os.path.join(paper_ws, "experiment_track_summary.json")
        )

        # Artifact-based status check (not assumption-based)
        theory_tex = os.path.join(paper_ws, "theory_sections.tex")
        experiment_tex = os.path.join(paper_ws, "experiment_report.tex")

        theory_status = None
        experiment_status = None

        if theory_questions:
            theory_status = "completed" if os.path.exists(theory_tex) else "failed"
        if empirical_questions:
            experiment_status = "completed" if os.path.exists(experiment_tex) else "failed"

        # Build structured task string from summaries
        notes = [
            "Synthesize the completed theory and empirical tracks before final interpretation.",
            f"Workspace: {ws}",
            "",
        ]

        if theory_summary:
            notes += [
                "THEORY TRACK SUMMARY:",
                f"  verified_numeric_claims: {theory_summary.get('verified_numeric_claims', [])}",
                f"  verified_symbolic_claims: {theory_summary.get('verified_symbolic_claims', [])}",
                f"  conjecture_claims: {theory_summary.get('conjecture_claims', [])}",
                f"  goal_coverage: {json.dumps(theory_summary.get('goal_coverage', {}), indent=4)}",
                f"  theory_sections.tex: {theory_summary.get('output_files', {}).get('theory_sections', 'unknown')}",
                f"  appendix_proofs.tex: {theory_summary.get('output_files', {}).get('appendix_proofs', 'unknown')}",
                "",
            ]
        else:
            notes.append("WARNING: theory_track_summary.json not found — theory track may have failed.")
            notes.append("")

        if experiment_summary:
            notes += [
                "EXPERIMENT TRACK SUMMARY:",
                f"  passed: {experiment_summary.get('passed', [])}",
                f"  partial: {experiment_summary.get('partial', [])}",
                f"  failed: {experiment_summary.get('failed', [])}",
                f"  goal_coverage: {json.dumps(experiment_summary.get('goal_coverage', {}), indent=4)}",
                f"  experiment_report.tex: {experiment_summary.get('output_files', {}).get('experiment_report_tex', 'unknown')}",
                "",
            ]
        else:
            notes.append("WARNING: experiment_track_summary.json not found — experiment track may have failed.")
            notes.append("")

        if theory_status == "failed" or experiment_status == "failed":
            notes.append(
                "WARNING: One or more tracks failed to produce output artifacts. "
                "Proceed with available results and explicitly flag missing track outputs "
                "in the synthesis."
            )

        notes += [
            "Ground all synthesis conclusions in workspace artifacts.",
            "Inspect math_workspace/ and experiment_runs/ directly for evidence.",
            "Write a focused synthesis that explains contradictions, stronger baselines, "
            "and interpretation risks.",
        ]

        return {
            "theory_track_status": theory_status,
            "experiment_track_status": experiment_status,
            "agent_task": "\n".join(notes),
        }

    track_merge_node.__name__ = "track_merge"
    return track_merge_node
