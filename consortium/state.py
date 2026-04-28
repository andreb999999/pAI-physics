"""
Shared graph state schema for the LangGraph research pipeline.

ResearchState is the single source of truth passed between all nodes.
The `messages` field uses add_messages reducer so each node can append
without overwriting prior conversation history.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# Required fields that MUST be present when the graph starts
_REQUIRED_FIELDS = frozenset({
    "task", "workspace_dir", "pipeline_mode", "pipeline_stages",
})

# Fields expected to be initialized with defaults at graph entry
_EXPECTED_FIELDS = frozenset({
    "math_enabled", "enforce_paper_artifacts", "enforce_editorial_artifacts",
    "require_pdf", "require_experiment_plan", "min_review_score",
    "followup_max_iterations", "manager_max_steps",
    "pipeline_stage_index", "iteration_count", "followup_iteration",
    "research_cycle", "max_research_cycles", "novelty_check_attempts",
    "rebuttal_iteration", "max_rebuttal_iterations",
    "validation_retry_count", "max_validation_retries",
    "tree_search_enabled", "enable_milestone_gates", "milestone_timeout",
    "iterate_mode", "autonomous_mode", "finished",
    "lit_review_attempts", "brainstorm_cycle", "brainstorm_artifact_retries",
    "verify_rework_attempts", "duality_rework_attempts",
    "theory_repair_count",
    "iterate_start_stage_override", "executed_stages",
})


def validate_initial_state(state: dict) -> list[str]:
    """Validate that a state dict has all required fields for graph entry.

    Raises ValueError if required fields are missing.
    Returns a list of warnings for missing expected (but optional) fields.
    """
    missing_required = _REQUIRED_FIELDS - set(state.keys())
    if missing_required:
        raise ValueError(
            f"ResearchState missing required fields: {sorted(missing_required)}"
        )

    warnings: list[str] = []
    missing_expected = _EXPECTED_FIELDS - set(state.keys())
    if missing_expected:
        warnings.append(
            f"ResearchState missing expected fields (will use defaults): "
            f"{sorted(missing_expected)}"
        )
        for field in missing_expected:
            logger.debug("State field '%s' not initialized at graph entry", field)

    return warnings


def _merge_dicts(left: dict, right: dict) -> dict:
    """Merge dict state produced by parallel branches."""
    return {**(left or {}), **(right or {})}


# ---------------------------------------------------------------------------
# Message history pruning — keeps state from growing unbounded.
# ---------------------------------------------------------------------------

_MAX_MESSAGES = 50  # Keep the last N messages; summarize earlier ones.


def prune_messages(state: dict, max_messages: int = _MAX_MESSAGES) -> dict:
    """Return a state update that trims the message history.

    Keeps the most recent *max_messages* messages.  Earlier messages are
    replaced by a single summary message so downstream agents still have
    context about what happened.

    Call this from checkpoint hooks or between graph stages.
    """
    messages = state.get("messages", [])
    if len(messages) <= max_messages:
        return {}  # nothing to prune

    # Split into old (to summarize) and recent (to keep)
    old_msgs = messages[:-max_messages]
    recent_msgs = messages[-max_messages:]

    # Build a lightweight summary of the pruned messages
    agent_names = set()
    tool_calls = 0
    for msg in old_msgs:
        name = getattr(msg, "name", None) or type(msg).__name__
        agent_names.add(name)
        if type(msg).__name__ == "ToolMessage":
            tool_calls += 1

    from langchain_core.messages import SystemMessage
    summary = SystemMessage(
        content=(
            f"[History pruned: {len(old_msgs)} earlier messages from agents "
            f"{', '.join(sorted(agent_names)[:10])} with {tool_calls} tool calls "
            f"were removed to save context. Recent {max_messages} messages retained.]"
        )
    )
    return {"messages": [summary] + recent_msgs}


class ResearchState(TypedDict):
    """Single source of truth for the LangGraph research pipeline.

    Lifecycle:
      1. **Graph entry**: Runner constructs the initial state dict with all
         required fields (task, workspace_dir, pipeline_mode, pipeline_stages)
         and expected fields initialized to defaults (counters to 0, bools to
         False, Optional fields to None). Use ``validate_initial_state()``
         to verify completeness before invoking the graph.
      2. **During execution**: Each node returns a partial dict that LangGraph
         merges into the state via the field's reducer (``add_messages`` for
         append-only message history, ``_merge_dicts`` for agent_outputs,
         ``operator.add`` for list accumulators).
      3. **Terminal**: ``finished=True`` signals the graph to exit.

    Invariants:
      - ``pipeline_stage_index`` is always in ``[0, len(pipeline_stages)]``.
      - ``iteration_count`` increments monotonically.
      - ``agent_outputs`` keys are agent names; values are output strings
        (or ``[AGENT_ERROR: ...]`` on failure).
    """

    # -----------------------------------------------------------------
    # Core conversation -- uses add_messages reducer (append-only)
    # -----------------------------------------------------------------
    messages: Annotated[list, add_messages]

    # -----------------------------------------------------------------
    # Run configuration (set once at graph entry, read-only after that)
    # Required: must be present for graph to start.
    # -----------------------------------------------------------------
    task: str                    # The research task description
    workspace_dir: str           # Absolute path to run workspace
    pipeline_mode: str           # default | full_research | quick
    math_enabled: bool           # Whether math/theory agents are enabled
    enforce_paper_artifacts: bool  # Require paper_workspace artifacts at gates
    enforce_editorial_artifacts: bool  # Require editorial quality at gates
    require_pdf: bool            # Require PDF generation before completion
    require_experiment_plan: bool  # Require experiment_plan.json
    min_review_score: int        # Minimum review score to pass validation gate (1-10)
    followup_max_iterations: int  # Max follow-up research cycles
    manager_max_steps: int       # Max total manager steps before forced exit
    pipeline_stages: list[str]   # Ordered list of stage names for this run

    # -----------------------------------------------------------------
    # Dynamic routing state (managed by router/dispatcher nodes)
    # Initialized: pipeline_stage_index=0, others=None
    # -----------------------------------------------------------------
    pipeline_stage_index: int             # Index into pipeline_stages; incremented after each stage completes
    current_agent: Optional[str]          # Name of next specialist to invoke (set by router)
    agent_task: Optional[str]             # Task prompt for the specialist (cleared to None after use)
    iterate_start_stage_override: Optional[str]  # explicit iterate-mode entry stage override

    # -----------------------------------------------------------------
    # Agent outputs — keyed by agent name, shallow-merged across parallel branches.
    # Values are output strings or "[AGENT_ERROR: ...]" on failure.
    # -----------------------------------------------------------------
    agent_outputs: Annotated[dict, _merge_dicts]

    # -----------------------------------------------------------------
    # Artifact tracking
    # -----------------------------------------------------------------
    artifacts: dict                       # artifact_name -> absolute path
    executed_stages: Annotated[list[str], operator.add]  # ordered canonical stage visits

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
    validation_retry_count: int           # validation loop retry counter
    max_validation_retries: int           # hard cap on validation retries (default 3)

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
    theory_track_summary: Optional[dict]     # structured summary from proof_transcription_agent
    theory_repair_count: int                 # intra-track retry counter (max 2)

    # -----------------------------------------------------------------
    # Agentic tree search state
    # -----------------------------------------------------------------
    tree_search_enabled: bool                # whether tree search is active for this run
    tree_state_path: Optional[str]           # path to tree_search_state.json on disk
    active_branch_id: Optional[str]          # currently executing branch node ID

    # -----------------------------------------------------------------
    # Milestone reports & human-in-the-loop gates
    # -----------------------------------------------------------------
    milestone_reports: Annotated[list[str], operator.add]  # paths to generated milestone report PDFs
    human_feedback_history: Annotated[list[dict], operator.add]  # [{phase, action, feedback, timestamp}]
    enable_milestone_gates: bool             # pause at milestones for human input
    milestone_timeout: int                   # seconds to wait for human (default 3600)

    # -----------------------------------------------------------------
    # Iterate mode (revision from prior paper + feedback)
    # -----------------------------------------------------------------
    iterate_mode: bool                           # True when running in revision mode
    iterate_prior_paper_path: Optional[str]      # path to prior paper in workspace
    iterate_feedback_path: Optional[str]         # path to consolidated feedback markdown
    iterate_feedback_summary: Optional[str]      # short summary of key changes needed
    iterate_binding_constraints: Optional[str]   # PI's non-negotiable research directives (from human_directive.md)
    iterate_route: Optional[str]                 # routing decision: writing_only, needs_research, needs_full_rethink

    # -----------------------------------------------------------------
    # Intermediate validation checkpoints
    # -----------------------------------------------------------------
    intermediate_validation_log: Annotated[list[dict], operator.add]  # [{checkpoint, timestamp, results}]

    # -----------------------------------------------------------------
    # V2 pipeline — persona council output
    # -----------------------------------------------------------------
    research_proposal: Optional[str]           # 1-2 page synthesized proposal from persona council
    autonomous_mode: bool                      # True = no human-in-the-loop gates

    # -----------------------------------------------------------------
    # V2 pipeline — brainstorm & goals
    # -----------------------------------------------------------------
    brainstorm_output: Optional[str]           # structured practical approaches
    brainstorm_history: list[str]              # accumulated brainstorm outputs across cycles
    research_goals: Optional[dict]             # {goals: [{id, description, success_criteria, track}], total_goals: int}
    formalized_results: Optional[str]          # structured findings from execution

    # -----------------------------------------------------------------
    # V2 pipeline — gate results
    # -----------------------------------------------------------------
    lit_review_feasibility: Optional[dict]     # {feasible: bool, reason: str}
    verify_completion_result: Optional[dict]   # {goals_met, goals_total, ratio, verdict, goal_verdicts}
    verify_completion_history: list[dict]      # previous verify_completion_results for progress vetting
    duality_check_result: Optional[dict]       # {both_passed, check_a: {passed, reasoning, score, suggestions}, check_b: {...}}

    # -----------------------------------------------------------------
    # V2 pipeline — loop counters
    # -----------------------------------------------------------------
    lit_review_attempts: int                   # lit_review→council loop (max 2)
    brainstorm_cycle: int                      # brainstorm re-entry from verify or duality (max 3)
    brainstorm_artifact_retries: int           # brainstorm artifact repair attempts (max 2)
    verify_rework_attempts: int                # verify→formalize_goals loop (max 3)
    duality_rework_attempts: int               # duality→followup_lit→brainstorm loop (max 2)

    # -----------------------------------------------------------------
    # Critical failure — halts pipeline on non-retryable errors (e.g. 4xx API)
    # -----------------------------------------------------------------
    critical_failure: Optional[str]              # None = OK; set to error message to halt pipeline

    # -----------------------------------------------------------------
    # Terminal flag
    # -----------------------------------------------------------------
    finished: bool
