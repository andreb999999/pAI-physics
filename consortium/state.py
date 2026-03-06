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
    agent_outputs: dict                   # agent_name -> last output string

    # -----------------------------------------------------------------
    # Artifact tracking
    # -----------------------------------------------------------------
    artifacts: dict                       # artifact_name -> absolute path

    # -----------------------------------------------------------------
    # Iteration counters
    # -----------------------------------------------------------------
    iteration_count: int                  # total manager steps taken
    followup_iteration: int               # Step 6/6.2 loop counter

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
    # Terminal flag
    # -----------------------------------------------------------------
    finished: bool
