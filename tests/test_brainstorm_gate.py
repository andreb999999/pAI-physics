"""Tests for the brainstorm artifact gate in consortium.graph."""

from __future__ import annotations

import json


def _make_gate(tmp_path):
    from consortium.graph import build_brainstorm_artifact_gate_node

    return build_brainstorm_artifact_gate_node(str(tmp_path), max_retries=2)


def _write_valid_brainstorm_md(tmp_path):
    paper_ws = tmp_path / "paper_workspace"
    paper_ws.mkdir(exist_ok=True)
    (paper_ws / "brainstorm.md").write_text(
        "# Executive Summary\n\n"
        "Summary.\n\n"
        "## Per-Hypothesis Approach Menu\n\n"
        "Approaches.\n\n"
        "## Recommended Priority Ordering\n\n"
        "1. First.\n\n"
        "## Open Questions and Decision Points\n\n"
        "Questions.\n"
    )


def _write_valid_brainstorm_json(tmp_path):
    paper_ws = tmp_path / "paper_workspace"
    paper_ws.mkdir(exist_ok=True)
    (paper_ws / "brainstorm.json").write_text(
        json.dumps(
            {
                "hypotheses_addressed": ["H1"],
                "approaches": [
                    {
                        "id": "approach_001",
                        "title": "Test H1",
                        "type": "experiment",
                        "hypothesis_ids": ["H1"],
                        "priority_rank": 1,
                    }
                ],
            }
        )
    )


def test_summary_only_brainstorm_does_not_advance(tmp_path):
    summary_dir = tmp_path / "stage_summaries"
    summary_dir.mkdir()
    (summary_dir / "brainstorm_agent_summary.tex").write_text("summary only")

    result = _make_gate(tmp_path)({})

    assert result["current_agent"] == "brainstorm_agent"
    assert result["brainstorm_artifact_retries"] == 1
    gate_result = result["validation_results"]["brainstorm_artifact_gate"]
    assert gate_result["is_valid"] is False
    assert "paper_workspace/brainstorm.md" in gate_result["errors"]
    assert "paper_workspace/brainstorm.json" in gate_result["errors"]


def test_missing_brainstorm_md_reroutes_to_brainstorm(tmp_path):
    _write_valid_brainstorm_json(tmp_path)

    result = _make_gate(tmp_path)({})

    assert result["current_agent"] == "brainstorm_agent"
    assert any(
        error == "paper_workspace/brainstorm.md"
        for error in result["validation_results"]["brainstorm_artifact_gate"]["errors"]
    )


def test_malformed_brainstorm_json_reroutes_to_brainstorm(tmp_path):
    _write_valid_brainstorm_md(tmp_path)
    paper_ws = tmp_path / "paper_workspace"
    (paper_ws / "brainstorm.json").write_text("{not valid json")

    result = _make_gate(tmp_path)({})

    assert result["current_agent"] == "brainstorm_agent"
    assert any(
        "not valid JSON" in error
        for error in result["validation_results"]["brainstorm_artifact_gate"]["errors"]
    )


def test_missing_approach_fields_reroutes_to_brainstorm(tmp_path):
    _write_valid_brainstorm_md(tmp_path)
    paper_ws = tmp_path / "paper_workspace"
    (paper_ws / "brainstorm.json").write_text(
        json.dumps(
            {
                "hypotheses_addressed": ["H1"],
                "approaches": [
                    {
                        "id": "approach_001",
                        "type": "experiment",
                        "hypothesis_ids": ["H1"],
                        "priority_rank": 1,
                    }
                ],
            }
        )
    )

    result = _make_gate(tmp_path)({})

    assert result["current_agent"] == "brainstorm_agent"
    assert any(
        "missing non-empty 'title'" in error
        for error in result["validation_results"]["brainstorm_artifact_gate"]["errors"]
    )


def test_valid_brainstorm_artifacts_advance_to_formalize_goals_entry(tmp_path):
    _write_valid_brainstorm_md(tmp_path)
    _write_valid_brainstorm_json(tmp_path)

    result = _make_gate(tmp_path)({"brainstorm_artifact_retries": 1})

    assert result["current_agent"] == "formalize_goals_entry"
    assert result["brainstorm_artifact_retries"] == 0
    assert "brainstorm_artifact_gate" not in result["validation_results"]
    assert result["agent_task"] is None


def test_brainstorm_gate_sets_critical_failure_after_retry_exhaustion(tmp_path):
    result = _make_gate(tmp_path)({"brainstorm_artifact_retries": 2})

    assert result["current_agent"] is None
    assert result["agent_task"] is None
    assert "critical_failure" in result
    assert "BRAINSTORM ARTIFACT GATE FAILURE" in result["critical_failure"]
