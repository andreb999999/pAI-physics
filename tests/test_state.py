"""
Tests for consortium/state.py — ResearchState schema and add_messages reducer.
"""

def test_research_state_importable():
    from consortium.state import ResearchState
    assert ResearchState is not None


def test_research_state_fields():
    from consortium.state import ResearchState
    # Check all expected fields are present in the TypedDict annotations
    annotations = ResearchState.__annotations__
    required_fields = [
        "messages", "task", "workspace_dir", "pipeline_mode", "math_enabled",
        "enforce_paper_artifacts", "enforce_editorial_artifacts",
        "require_pdf", "require_experiment_plan",
        "min_review_score", "followup_max_iterations", "manager_max_steps",
        "pipeline_stages", "pipeline_stage_index", "current_agent", "agent_task",
        "agent_outputs", "artifacts", "iteration_count", "followup_iteration",
        "validation_results", "interrupt_instruction",
        "research_cycle", "max_research_cycles",
        "theory_track_status", "experiment_track_status", "track_decomposition",
        "finished",
    ]
    for field in required_fields:
        assert field in annotations, f"Expected field '{field}' missing from ResearchState"


def test_messages_uses_add_messages_reducer():
    """The messages field annotation should use the add_messages reducer."""
    import typing
    from consortium.state import ResearchState

    hints = typing.get_type_hints(ResearchState, include_extras=True)
    ann = hints["messages"]
    # Annotated[list, add_messages] — the metadata should contain add_messages
    args = typing.get_args(ann)
    assert len(args) >= 2, "messages should be Annotated[list, reducer]"
    # The reducer should be callable and named 'add_messages'
    reducer = args[1]
    assert callable(reducer), "add_messages reducer should be callable"


def test_research_state_instantiation_as_dict():
    """ResearchState is a TypedDict — can be constructed as a plain dict."""
    state = {
        "messages": [],
        "task": "test task",
        "workspace_dir": "/tmp/test",
        "pipeline_mode": "full_research",
        "math_enabled": False,
        "enforce_paper_artifacts": False,
        "enforce_editorial_artifacts": False,
        "require_pdf": False,
        "require_experiment_plan": False,
        "min_review_score": 8,
        "followup_max_iterations": 3,
        "manager_max_steps": 50,
        "pipeline_stages": ["ideation_agent", "writeup_agent"],
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
    # Should not raise — TypedDict is just a dict at runtime
    assert state["task"] == "test task"
    assert state["pipeline_stages"] == ["ideation_agent", "writeup_agent"]
