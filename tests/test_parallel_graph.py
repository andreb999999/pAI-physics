"""
Tests for the parallel theory/experiment graph helpers.
"""

from __future__ import annotations

import json

from langgraph.graph import END


def _base_state(workspace_dir: str) -> dict:
    return {
        "messages": [],
        "task": "test task",
        "workspace_dir": workspace_dir,
        "pipeline_mode": "default",
        "math_enabled": True,
        "enforce_paper_artifacts": False,
        "enforce_editorial_artifacts": False,
        "require_pdf": False,
        "require_experiment_plan": False,
        "min_review_score": 8,
        "followup_max_iterations": 3,
        "manager_max_steps": 50,
        "pipeline_stages": [],
        "pipeline_stage_index": 0,
        "current_agent": None,
        "agent_task": None,
        "agent_outputs": {},
        "artifacts": {},
        "iteration_count": 0,
        "followup_iteration": 0,
        "research_cycle": 0,
        "max_research_cycles": 3,
        "validation_results": {},
        "interrupt_instruction": None,
        "theory_track_status": None,
        "experiment_track_status": None,
        "track_decomposition": None,
        "finished": False,
    }


def test_track_router_sends_both_tracks(tmp_path):
    from consortium.graph import track_router

    state = _base_state(str(tmp_path))
    state["track_decomposition"] = {
        "theory_questions": ["Can we prove a bound?"],
        "empirical_questions": ["Does the bound predict training behavior?"],
        "recommended_track": "both",
    }

    sends = track_router(state)

    assert len(sends) == 2
    assert {send.node for send in sends} == {"theory_track", "experiment_track"}
    theory_send = next(send for send in sends if send.node == "theory_track")
    exp_send = next(send for send in sends if send.node == "experiment_track")
    assert "prove a bound" in theory_send.arg["agent_task"].lower()
    assert "predict training behavior" in exp_send.arg["agent_task"].lower()


def test_track_router_skips_theory_when_math_disabled(tmp_path):
    from consortium.graph import track_router

    state = _base_state(str(tmp_path))
    state["math_enabled"] = False
    state["track_decomposition"] = {
        "theory_questions": ["Theory question"],
        "empirical_questions": ["Empirical question"],
        "recommended_track": "both",
    }

    sends = track_router(state)

    assert len(sends) == 1
    assert sends[0].node == "experiment_track"


def test_followup_gate_routes_back_to_planner(tmp_path):
    from consortium.graph import build_followup_gate_node, followup_router

    paper_workspace = tmp_path / "paper_workspace"
    paper_workspace.mkdir()
    decision_path = paper_workspace / "followup_decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "decision": "followup_required",
                "confidence": "medium",
                "evidence_summary": ["Need one more ablation"],
            }
        )
    )

    state = _base_state(str(tmp_path))
    gate = build_followup_gate_node(str(tmp_path))
    update = gate(state)

    assert update["current_agent"] == "research_planner_agent"
    assert update["research_cycle"] == 1
    routed = followup_router({**state, **update})
    assert routed == "research_planner_agent"


def test_validation_gate_passes_without_required_artifacts(tmp_path):
    from consortium.graph import build_validation_gate_node, validation_router

    state = _base_state(str(tmp_path))
    gate = build_validation_gate_node()
    update = gate(state)

    assert update["finished"] is True
    assert validation_router({**state, **update}) == END


def test_validation_gate_loops_to_writeup_on_missing_artifacts(tmp_path):
    from consortium.graph import build_validation_gate_node, validation_router

    state = _base_state(str(tmp_path))
    state["pipeline_mode"] = "full_research"
    gate = build_validation_gate_node()
    update = gate(state)

    assert update["finished"] is False
    assert "Validation failures" in update["agent_task"]
    assert validation_router({**state, **update}) == "writeup_agent"
