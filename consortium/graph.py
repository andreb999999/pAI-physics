"""
LangGraph research pipeline — directly wired multi-phase workflow.

This graph replaces the manager hub-and-spoke loop with:
1. Discovery: ideation -> literature review -> research planner
2. Parallel execution: theory and experiment tracks in parallel
3. Synthesis loop: merge -> synthesis literature review -> results analysis
4. Paper production: resource preparation -> writeup -> proofreading -> reviewer
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Optional

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from .agents import (
    build_experiment_design_node,
    build_experiment_literature_node,
    build_experiment_transcription_node,
    build_experiment_verification_node,
    build_experimentation_node,
    build_ideation_node,
    build_literature_review_node,
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
    build_track_merge_node,
    build_writeup_node,
)
from .pdf_summary import with_pdf_summary
from .state import ResearchState
from .workflow_utils import (
    followup_decision_requires_loop,
    run_validation_gates,
    safe_int,
)

# ---------------------------------------------------------------------------
# Deterministic stage roster
# ---------------------------------------------------------------------------

DISCOVERY_STAGES = [
    "ideation_agent",
    "literature_review_agent",
    "research_planner_agent",
]

MATH_PIPELINE_STAGES = [
    "math_literature_agent",
    "math_proposer_agent",
    "math_prover_agent",
    "math_rigorous_verifier_agent",
    "math_empirical_verifier_agent",
    "proof_transcription_agent",
]

EXPERIMENT_PIPELINE_STAGES = [
    "experiment_literature_agent",
    "experiment_design_agent",
    "experimentation_agent",
    "experiment_verification_agent",
    "experiment_transcription_agent",
]

POST_TRACK_STAGES = [
    "synthesis_literature_review_agent",
    "results_analysis_agent",
    "resource_preparation_agent",
    "writeup_agent",
    "proofreading_agent",
    "reviewer_agent",
]


def build_pipeline_stages(enable_math_agents: bool) -> list[str]:
    stages = list(DISCOVERY_STAGES)
    if enable_math_agents:
        stages.extend(MATH_PIPELINE_STAGES)
    stages.extend(EXPERIMENT_PIPELINE_STAGES)
    stages.extend(POST_TRACK_STAGES)
    return stages


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _format_track_task(state: dict, track_name: str, questions: list[str]) -> str:
    cycle = safe_int(state.get("research_cycle", 0), 0)
    if not questions:
        return f"No {track_name} questions were identified for this cycle."

    question_lines = "\n".join(f"- {question}" for question in questions)
    return (
        f"Research cycle: {cycle}\n"
        f"Execute the {track_name} track for the current research plan.\n"
        f"Questions assigned to this track:\n{question_lines}\n\n"
        "Read the latest planning artifacts from `paper_workspace/`, "
        "produce the mandatory artifacts for your track, and ground all work "
        "in workspace evidence and cited literature."
    )


def track_router(state: ResearchState) -> list[Send]:
    """Fan out to the theory and/or experiment tracks based on track decomposition."""
    track_decomposition = state.get("track_decomposition") or {}
    theory_questions = list(track_decomposition.get("theory_questions") or [])
    empirical_questions = list(track_decomposition.get("empirical_questions") or [])
    recommended_track = str(track_decomposition.get("recommended_track", "")).strip().lower()

    sends: list[Send] = []
    theory_allowed = state.get("math_enabled", False) and recommended_track in {"", "both", "theory"}
    experiment_allowed = recommended_track in {"", "both", "empirical"}

    if theory_allowed and theory_questions:
        sends.append(
            Send(
                "theory_track",
                {
                    **state,
                    "agent_task": _format_track_task(state, "theory", theory_questions),
                    "theory_track_status": "in_progress",
                },
            )
        )

    if experiment_allowed and empirical_questions:
        sends.append(
            Send(
                "experiment_track",
                {
                    **state,
                    "agent_task": _format_track_task(state, "experiment", empirical_questions),
                    "experiment_track_status": "in_progress",
                },
            )
        )

    if not sends:
        sends.append(
            Send(
                "track_merge",
                {
                    **state,
                    "agent_task": (
                        "No theory or empirical execution track was selected. "
                        "Proceed directly to synthesis and results analysis."
                    ),
                },
            )
        )
    return sends


def followup_router(state: ResearchState) -> str:
    # Routing decision was already made by followup_gate_node which sets current_agent
    # to "research_planner_agent" (loop) or "resource_preparation_agent" (continue).
    return state.get("current_agent") or "resource_preparation_agent"


def validation_router(state: ResearchState) -> str:
    if state.get("finished"):
        return END
    return "writeup_agent"


def build_followup_gate_node(workspace_dir: str) -> Any:
    def followup_gate_node(state: dict) -> dict:
        required, reason = followup_decision_requires_loop(workspace_dir)
        research_cycle = safe_int(state.get("research_cycle", 0), 0)
        max_cycles = max(0, safe_int(state.get("max_research_cycles", 3), 3))

        if required and research_cycle < max_cycles:
            return {
                "current_agent": "research_planner_agent",
                "research_cycle": research_cycle + 1,
                "followup_iteration": safe_int(state.get("followup_iteration", 0), 0) + 1,
                "finished": False,
                "agent_task": (
                    "Prepare a focused follow-up research plan based on "
                    f"results analysis. Reason: {reason}"
                ),
            }

        return {
            "current_agent": "resource_preparation_agent",
            "finished": False,
            "agent_task": None,
        }

    followup_gate_node.__name__ = "followup_gate"
    return followup_gate_node


def build_validation_gate_node() -> Any:
    def validation_gate_node(state: dict) -> dict:
        validation = run_validation_gates(state)
        if validation["gate_passed"]:
            return {
                "validation_results": validation["validation_results"],
                "finished": True,
                "agent_task": None,
            }

        error_lines = [
            f"- {gate}: {'; '.join(result['errors'])}"
            for gate, result in validation["validation_results"].items()
            if not result.get("is_valid")
        ]
        return {
            "validation_results": validation["validation_results"],
            "finished": False,
            "agent_task": (
                "Revise the paper to satisfy validation gates before finalization.\n"
                "Validation failures:\n" + "\n".join(error_lines)
            ),
        }

    validation_gate_node.__name__ = "validation_gate"
    return validation_gate_node


def build_novelty_gate_node(workspace_dir: str) -> Any:
    """Gate between ideation and literature review that checks novelty assessment."""

    def novelty_gate_node(state: dict) -> dict:
        assessment_path = os.path.join(
            workspace_dir, "paper_workspace", "novelty_assessment.json"
        )
        max_attempts = 3
        attempts = safe_int(state.get("novelty_check_attempts", 0), 0)

        if not os.path.exists(assessment_path):
            # No assessment file -- pass through (backward compat)
            return {"current_agent": "literature_review_agent", "agent_task": None}

        try:
            with open(assessment_path) as f:
                assessment = json.load(f)
        except Exception:
            return {"current_agent": "literature_review_agent", "agent_task": None}

        if assessment.get("novel", True):
            return {"current_agent": "literature_review_agent", "agent_task": None}

        if attempts >= max_attempts:
            # After max attempts, proceed anyway with a warning
            print(
                f"Warning: novelty gate failed after {max_attempts} attempts, "
                "proceeding to literature review."
            )
            return {"current_agent": "literature_review_agent", "agent_task": None}

        justification = assessment.get("novelty_justification", "N/A")
        closest = assessment.get("closest_existing_work", "N/A")
        return {
            "current_agent": "ideation_agent",
            "novelty_check_attempts": attempts + 1,
            "agent_task": (
                f"NOVELTY GATE REJECTION (attempt {attempts + 1}/{max_attempts}): "
                f"Your previous idea was assessed as NOT NOVEL.\n"
                f"Justification: {justification}\n"
                f"Closest existing work: {closest}\n\n"
                "Generate a substantially different idea that addresses a genuine "
                "gap in the literature. Delete the old novelty_assessment.json and "
                "run CheckIdeaNoveltyTool again on your new idea."
            ),
        }

    novelty_gate_node.__name__ = "novelty_gate"
    return novelty_gate_node


def novelty_router(state: ResearchState) -> str:
    """Route based on novelty gate decision."""
    return state.get("current_agent") or "literature_review_agent"


def build_theory_track_subgraph(
    model: Any,
    workspace_dir: str,
    authorized_imports: Optional[List[str]] = None,
    counsel_models: Optional[List[Any]] = None,
    summary_model_id: Optional[str] = "claude-sonnet-4-6",
):
    graph = StateGraph(ResearchState)
    counsel_kwargs = {"counsel_models": counsel_models} if counsel_models else {}

    def _wrap(node, name):
        return with_pdf_summary(node, name, workspace_dir, summary_model_id)

    graph.add_node(
        "math_literature_agent",
        _wrap(build_math_literature_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "math_literature_agent"),
    )
    graph.add_node(
        "math_proposer_agent",
        _wrap(build_math_proposer_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "math_proposer_agent"),
    )
    graph.add_node(
        "math_prover_agent",
        _wrap(build_math_prover_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "math_prover_agent"),
    )
    graph.add_node(
        "math_rigorous_verifier_agent",
        _wrap(build_math_rigorous_verifier_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "math_rigorous_verifier_agent"),
    )
    graph.add_node(
        "math_empirical_verifier_agent",
        _wrap(build_math_empirical_verifier_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "math_empirical_verifier_agent"),
    )
    graph.add_node(
        "proof_transcription_agent",
        _wrap(build_proof_transcription_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "proof_transcription_agent"),
    )
    graph.set_entry_point("math_literature_agent")
    graph.add_edge("math_literature_agent", "math_proposer_agent")
    graph.add_edge("math_proposer_agent", "math_prover_agent")
    graph.add_edge("math_prover_agent", "math_rigorous_verifier_agent")
    graph.add_edge("math_rigorous_verifier_agent", "math_empirical_verifier_agent")
    graph.add_edge("math_empirical_verifier_agent", "proof_transcription_agent")
    graph.add_edge("proof_transcription_agent", END)
    return graph.compile()


def build_experiment_track_subgraph(
    model: Any,
    workspace_dir: str,
    authorized_imports: Optional[List[str]] = None,
    counsel_models: Optional[List[Any]] = None,
    summary_model_id: Optional[str] = "claude-sonnet-4-6",
):
    graph = StateGraph(ResearchState)
    counsel_kwargs = {"counsel_models": counsel_models} if counsel_models else {}

    def _wrap(node, name):
        return with_pdf_summary(node, name, workspace_dir, summary_model_id)

    graph.add_node(
        "experiment_literature_agent",
        _wrap(build_experiment_literature_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "experiment_literature_agent"),
    )
    graph.add_node(
        "experiment_design_agent",
        _wrap(build_experiment_design_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "experiment_design_agent"),
    )
    graph.add_node(
        "experimentation_agent",
        _wrap(build_experimentation_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "experimentation_agent"),
    )
    graph.add_node(
        "experiment_verification_agent",
        _wrap(build_experiment_verification_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "experiment_verification_agent"),
    )
    graph.add_node(
        "experiment_transcription_agent",
        _wrap(build_experiment_transcription_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "experiment_transcription_agent"),
    )
    graph.set_entry_point("experiment_literature_agent")
    graph.add_edge("experiment_literature_agent", "experiment_design_agent")
    graph.add_edge("experiment_design_agent", "experimentation_agent")
    graph.add_edge("experimentation_agent", "experiment_verification_agent")
    graph.add_edge("experiment_verification_agent", "experiment_transcription_agent")
    graph.add_edge("experiment_transcription_agent", END)
    return graph.compile()


def build_track_subgraph_node(
    subgraph: Any,
    status_field: str,
    status_value: str = "completed",
) -> Any:
    def node(state: dict) -> dict:
        final_state = subgraph.invoke(state)
        return {
            "agent_outputs": final_state.get("agent_outputs", {}),
            status_field: status_value,
            "agent_task": None,
        }

    node.__name__ = status_field.removesuffix("_status")
    return node


def build_noop_track_node(status_field: str) -> Any:
    def node(state: dict) -> dict:
        return {
            status_field: state.get(status_field),
            "agent_task": None,
        }

    node.__name__ = status_field.removesuffix("_status")
    return node


def build_synthesis_literature_node(
    model: Any,
    workspace_dir: str,
    authorized_imports: Optional[List[str]] = None,
    counsel_models: Optional[List[Any]] = None,
) -> Any:
    base_node = build_literature_review_node(
        model,
        workspace_dir,
        authorized_imports,
        **({"counsel_models": counsel_models} if counsel_models else {}),
    )

    def synthesis_node(state: dict) -> dict:
        previous_output = state.get("agent_outputs", {}).get("literature_review_agent")
        result = base_node(state)
        outputs = dict(result.get("agent_outputs", {}))
        new_output = outputs.pop("literature_review_agent", previous_output)
        if previous_output is not None:
            outputs["literature_review_agent"] = previous_output
        outputs["synthesis_literature_review_agent"] = new_output
        return {
            **result,
            "agent_outputs": outputs,
        }

    synthesis_node.__name__ = "synthesis_literature_review_agent"
    return synthesis_node


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
    counsel_models: Optional[List[Any]] = None,
    summary_model_id: Optional[str] = "claude-sonnet-4-6",
    tree_search_config: Optional[Any] = None,
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
        counsel_models:           List of ChatLiteLLM instances for counsel mode.
                                  When provided, specialist nodes run multi-model debate
                                  instead of a single-model ReAct agent.
        summary_model_id:         Model used to format stage summary PDFs.
                                  Set to None to disable automatic PDF summaries.
        tree_search_config:       TreeSearchConfig instance (or None).
                                  When provided and enabled, the theory track uses
                                  DAG-layered best-first search to explore multiple
                                  proof strategies in parallel.

    Returns:
        Compiled LangGraph CompiledGraph.
    """
    counsel_kwargs = {"counsel_models": counsel_models} if counsel_models else {}

    def _wrap(node, name):
        """Wrap an agent node with automatic PDF summary generation."""
        return with_pdf_summary(node, name, workspace_dir, summary_model_id)

    theory_track_node = build_noop_track_node("theory_track_status")
    if enable_math_agents:
        # Use tree-search-enabled theory track when configured
        if tree_search_config and getattr(tree_search_config, "enabled", False):
            from consortium.tree_search.graph_integration import (
                build_tree_search_theory_track,
            )
            theory_subgraph = build_tree_search_theory_track(
                model=model,
                workspace_dir=workspace_dir,
                authorized_imports=authorized_imports,
                counsel_models=counsel_models,
                summary_model_id=summary_model_id,
                tree_config=tree_search_config,
            )
        else:
            theory_subgraph = build_theory_track_subgraph(
                model=model,
                workspace_dir=workspace_dir,
                authorized_imports=authorized_imports,
                counsel_models=counsel_models,
                summary_model_id=summary_model_id,
            )
        theory_track_node = build_track_subgraph_node(theory_subgraph, "theory_track_status")
    experiment_subgraph = build_experiment_track_subgraph(
        model=model,
        workspace_dir=workspace_dir,
        authorized_imports=authorized_imports,
        counsel_models=counsel_models,
        summary_model_id=summary_model_id,
    )

    nodes: dict[str, Any] = {
        "ideation_agent": _wrap(build_ideation_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "ideation_agent"),
        "literature_review_agent": _wrap(build_literature_review_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "literature_review_agent"),
        "research_planner_agent": _wrap(build_research_planner_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "research_planner_agent"),
        "theory_track": theory_track_node,
        "experiment_track": build_track_subgraph_node(experiment_subgraph, "experiment_track_status"),
        "track_merge": build_track_merge_node(workspace_dir=workspace_dir),
        "synthesis_literature_review_agent": _wrap(build_synthesis_literature_node(
            model, workspace_dir, authorized_imports, counsel_models
        ), "synthesis_literature_review_agent"),
        "results_analysis_agent": _wrap(build_results_analysis_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "results_analysis_agent"),
        "followup_gate": build_followup_gate_node(workspace_dir),
        "resource_preparation_agent": _wrap(build_resource_preparation_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "resource_preparation_agent"),
        "writeup_agent": _wrap(build_writeup_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "writeup_agent"),
        "proofreading_agent": _wrap(build_proofreading_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "proofreading_agent"),
        "reviewer_agent": _wrap(build_reviewer_node(model, workspace_dir, authorized_imports, **counsel_kwargs), "reviewer_agent"),
        "validation_gate": build_validation_gate_node(),
        "novelty_gate": build_novelty_gate_node(workspace_dir),
    }

    graph = StateGraph(ResearchState)
    for name, node in nodes.items():
        graph.add_node(name, node)

    graph.set_entry_point("ideation_agent")
    graph.add_edge("ideation_agent", "novelty_gate")
    graph.add_conditional_edges(
        "novelty_gate",
        novelty_router,
        {
            "ideation_agent": "ideation_agent",
            "literature_review_agent": "literature_review_agent",
        },
    )
    graph.add_edge("literature_review_agent", "research_planner_agent")
    graph.add_conditional_edges("research_planner_agent", track_router)
    graph.add_edge("theory_track", "track_merge")
    graph.add_edge("experiment_track", "track_merge")
    graph.add_edge("track_merge", "synthesis_literature_review_agent")
    graph.add_edge("synthesis_literature_review_agent", "results_analysis_agent")
    graph.add_edge("results_analysis_agent", "followup_gate")
    graph.add_conditional_edges(
        "followup_gate",
        followup_router,
        {
            "research_planner_agent": "research_planner_agent",
            "resource_preparation_agent": "resource_preparation_agent",
        },
    )
    graph.add_edge("resource_preparation_agent", "writeup_agent")
    graph.add_edge("writeup_agent", "proofreading_agent")
    graph.add_edge("proofreading_agent", "reviewer_agent")
    graph.add_edge("reviewer_agent", "validation_gate")
    graph.add_conditional_edges(
        "validation_gate",
        validation_router,
        {
            END: END,
            "writeup_agent": "writeup_agent",
        },
    )

    compile_kwargs: dict = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)


def get_default_checkpointer(workspace_dir: str):
    """Return a SqliteSaver checkpointer scoped to the workspace directory."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3
        db_path = os.path.join(workspace_dir, "checkpoints.db")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        return SqliteSaver(conn)
    except (ImportError, Exception) as e:
        print(f"Checkpointer unavailable ({e}); resumability disabled.")
        return None
