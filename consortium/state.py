"""
Shared graph state schema for the LangGraph research pipeline.

ResearchState is the single source of truth passed between all nodes.
The `messages` field uses add_messages reducer so each node can append
without overwriting prior conversation history.
"""

from __future__ import annotations

from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def _merge_dicts(left: dict, right: dict) -> dict:
    """Merge dict state produced by parallel branches."""
    return {**(left or {}), **(right or {})}


class ResearchState(TypedDict):
    # -----------------------------------------------------------------
    # Core conversation -- uses add_messages reducer (append-only)
    # -----------------------------------------------------------------
    messages: Annotated[list, add_messages]

    # -----------------------------------------------------------------
    # Run configuration (set once at graph entry, read-only after that)
    # -----------------------------------------------------------------
    task: str
    workspace_dir: str
    pipeline_mode: str          # default | full_research | quick
    math_enabled: bool
    enforce_paper_artifacts: bool
    enforce_editorial_artifacts: bool
    require_pdf: bool
    require_experiment_plan: bool
    min_review_score: int
    followup_max_iterations: int
    manager_max_steps: int
    pipeline_stages: list[str]   # deterministic stage order for this run

    # -----------------------------------------------------------------
    # Dynamic routing state
    # -----------------------------------------------------------------
    pipeline_stage_index: int             # next stage index in pipeline_stages
    current_agent: Optional[str]          # name of next specialist to invoke
    agent_task: Optional[str]             # task string sent to that specialist

    # -----------------------------------------------------------------
    # Agent outputs -- keyed by agent name
    # -----------------------------------------------------------------
    agent_outputs: Annotated[dict, _merge_dicts]  # agent_name -> last output string

    # -----------------------------------------------------------------
    # Artifact tracking
    # -----------------------------------------------------------------
    artifacts: dict                       # artifact_name -> absolute path

    # -----------------------------------------------------------------
    # Iteration counters
    # -----------------------------------------------------------------
    iteration_count: int                  # total manager steps taken
    followup_iteration: int               # Step 6/6.2 loop counter
    research_cycle: int                   # current plan-execute-analyze cycle
    max_research_cycles: int              # hard cap on follow-up replanning cycles
    novelty_check_attempts: int           # novelty gate retry counter (max 3)
    rebuttal_iteration: int               # rebuttal loop counter (Phase 2)
    max_rebuttal_iterations: int          # hard cap on rebuttal loops (default 2)

    # -----------------------------------------------------------------
    # Validation results
    # -----------------------------------------------------------------
    validation_results: dict              # gate_name -> {is_valid, errors}

    # -----------------------------------------------------------------
    # Live-steering interrupt
    # -----------------------------------------------------------------
    interrupt_instruction: Optional[str]  # set by socket listener thread

    # -----------------------------------------------------------------
    # Theory / experiment track state (populated after Step 4.5)
    # -----------------------------------------------------------------
    theory_track_status: Optional[str]       # pending | in_progress | completed
    experiment_track_status: Optional[str]   # pending | in_progress | completed
    track_decomposition: Optional[dict]      # {"empirical_questions": [...], "theory_questions": [...]}

    # -----------------------------------------------------------------
    # Agentic tree search state
    # -----------------------------------------------------------------
    tree_search_enabled: bool                # whether tree search is active for this run
    tree_state_path: Optional[str]           # path to tree_search_state.json on disk
    active_branch_id: Optional[str]          # currently executing branch node ID

    # -----------------------------------------------------------------
    # Terminal flag
    # -----------------------------------------------------------------
    finished: bool
