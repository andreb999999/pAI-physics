"""
LangGraph research pipeline — StateGraph builder.

Replaces the smolagents ManagerAgent.run() loop.

Graph topology
--------------
                  ┌──────────────────────────────────┐
    START ──► manager ──► ideation_agent              │
                  ▲       literature_review_agent     │
                  │       research_planner_agent      │
                  │       results_analysis_agent      │
                  └───────experimentation_agent        │
                          resource_preparation_agent  │
                          writeup_agent               │
                          proofreading_agent          │
                          reviewer_agent              │
                          (math agents if enabled)   │
                          FINISH ──────────────────► END

Every specialist node returns straight back to the manager node.
The manager uses an LLM to decide which specialist to call next or to finish.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from .agents import (
    build_experimentation_node,
    build_ideation_node,
    build_literature_review_node,
    build_manager_node,
    build_math_empirical_verifier_node,
    build_math_literature_node,
    build_math_proposer_node,
    build_math_prover_node,
    build_math_rigorous_verifier_node,
    build_proof_transcription_node,
    build_proofreading_node,
    build_research_planner_node,
    build_resource_preparation_node,
    build_results_analysis_node,
    build_reviewer_node,
    build_writeup_node,
)
from .state import ResearchState

# ---------------------------------------------------------------------------
# Specialist roster
# ---------------------------------------------------------------------------

BASE_SPECIALISTS = [
    "ideation_agent",
    "literature_review_agent",
    "research_planner_agent",
    "results_analysis_agent",
    "experimentation_agent",
    "resource_preparation_agent",
    "writeup_agent",
    "proofreading_agent",
    "reviewer_agent",
]

MATH_SPECIALISTS_CORE = [
    "math_proposer_agent",
    "math_prover_agent",
    "math_rigorous_verifier_agent",
    "math_empirical_verifier_agent",
]

MATH_SPECIALISTS_FULL_RESEARCH = [
    "math_literature_agent",
    "proof_transcription_agent",
]


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

def _route_from_manager(state: ResearchState) -> str:
    """Conditional edge: read current_agent from state and route there."""
    if state.get("finished"):
        return END
    agent = state.get("current_agent")
    if not agent or agent == "FINISH":
        return END
    return agent


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_research_graph(
    model: Any,
    workspace_dir: str,
    pipeline_mode: str = "default",
    enable_math_agents: bool = False,
    enforce_paper_artifacts: bool = False,
    enforce_editorial_artifacts: bool = False,
    require_pdf: bool = False,
    require_experiment_plan: bool = False,
    min_review_score: int = 8,
    followup_max_iterations: int = 3,
    manager_max_steps: int = 50,
    authorized_imports: Optional[List[str]] = None,
    checkpointer=None,
):
    """
    Build and compile the full LangGraph research pipeline.

    Args:
        model:                    ChatLiteLLM instance (or any LangChain chat model)
        workspace_dir:            Absolute path to the run workspace
        pipeline_mode:            "default" | "full_research" | "quick"
        enable_math_agents:       Include theorem-oriented math agents
        enforce_paper_artifacts:  Run artifact gate before FINISH
        enforce_editorial_artifacts: Run full editorial gates
        require_pdf:              Require final_paper.pdf at finish
        require_experiment_plan:  Require experiments_to_run_later.md at finish
        min_review_score:         Reviewer score threshold
        followup_max_iterations:  Max follow-up loops in full_research mode
        manager_max_steps:        Max total manager iterations
        authorized_imports:       Authorized Python import list for code tools
        checkpointer:             LangGraph checkpointer (SqliteSaver etc.)

    Returns:
        Compiled LangGraph CompiledGraph.
    """
    # Determine active specialist list
    specialists = list(BASE_SPECIALISTS)
    if enable_math_agents:
        specialists += MATH_SPECIALISTS_CORE
        if str(pipeline_mode).strip().lower() == "full_research":
            specialists += MATH_SPECIALISTS_FULL_RESEARCH

    # Build manager node (needs specialist list for routing prompt)
    manager_node = build_manager_node(
        model=model,
        workspace_dir=workspace_dir,
        pipeline_mode=pipeline_mode,
        available_agents=specialists,
        enable_math_agents=enable_math_agents,
        followup_max_iterations=followup_max_iterations,
        authorized_imports=authorized_imports,
    )

    # Build specialist nodes
    specialist_nodes: dict[str, Any] = {
        "ideation_agent": build_ideation_node(model, workspace_dir, authorized_imports),
        "literature_review_agent": build_literature_review_node(model, workspace_dir, authorized_imports),
        "research_planner_agent": build_research_planner_node(model, workspace_dir, authorized_imports),
        "results_analysis_agent": build_results_analysis_node(model, workspace_dir, authorized_imports),
        "experimentation_agent": build_experimentation_node(model, workspace_dir, authorized_imports),
        "resource_preparation_agent": build_resource_preparation_node(model, workspace_dir, authorized_imports),
        "writeup_agent": build_writeup_node(model, workspace_dir, authorized_imports),
        "proofreading_agent": build_proofreading_node(model, workspace_dir, authorized_imports),
        "reviewer_agent": build_reviewer_node(model, workspace_dir, authorized_imports),
    }
    if enable_math_agents:
        specialist_nodes.update({
            "math_proposer_agent": build_math_proposer_node(model, workspace_dir, authorized_imports),
            "math_prover_agent": build_math_prover_node(model, workspace_dir, authorized_imports),
            "math_rigorous_verifier_agent": build_math_rigorous_verifier_node(model, workspace_dir, authorized_imports),
            "math_empirical_verifier_agent": build_math_empirical_verifier_node(model, workspace_dir, authorized_imports),
        })
        if str(pipeline_mode).strip().lower() == "full_research":
            specialist_nodes.update({
                "math_literature_agent": build_math_literature_node(model, workspace_dir, authorized_imports),
                "proof_transcription_agent": build_proof_transcription_node(model, workspace_dir, authorized_imports),
            })

    # Assemble StateGraph
    graph = StateGraph(ResearchState)

    graph.add_node("manager", manager_node)
    for name, node in specialist_nodes.items():
        graph.add_node(name, node)

    # Entry point
    graph.set_entry_point("manager")

    # Conditional edges: manager -> specialist or END
    routing_map = {name: name for name in specialist_nodes}
    routing_map[END] = END
    routing_map["FINISH"] = END

    graph.add_conditional_edges("manager", _route_from_manager, routing_map)

    # All specialists return to manager
    for name in specialist_nodes:
        graph.add_edge(name, "manager")

    # Compile
    compile_kwargs: dict = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)


def get_default_checkpointer(workspace_dir: str):
    """Return a SqliteSaver checkpointer scoped to the workspace directory."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        db_path = os.path.join(workspace_dir, "checkpoints.db")
        return SqliteSaver.from_conn_string(db_path)
    except (ImportError, Exception) as e:
        print(f"Checkpointer unavailable ({e}); resumability disabled.")
        return None
