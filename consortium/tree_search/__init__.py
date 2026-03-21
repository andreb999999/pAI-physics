"""Agentic tree search for PoggioAI/MSc.

Implements DAG-layered best-first search for both theory (proof stage) and
experiment tracks. Inspired by AI Scientist-v2 progressive agentic tree search,
adapted for mathematical research with claim-graph-aware branching.
"""

from consortium.tree_search.tree_state import (
    CounselMode,
    NodeStatus,
    NodeType,
    TreeNode,
    TreeSearchConfig,
    TreeSearchState,
)
from consortium.tree_search.tree_persistence import (
    create_tree_state,
    ensure_tree_state,
    load_tree_state,
    save_tree_state,
)
from consortium.tree_search.tree_manager import (
    get_downstream_impact,
    get_frontier_claims,
    process_branch_result,
    run_tree_search_step,
    select_branches_for_execution,
)
from consortium.tree_search.workspace_fork import (
    fork_workspace,
    merge_branch,
    populate_branch,
    promote_branch,
    retarget_tools,
)
from consortium.tree_search.budget_allocator import TreeBudgetAllocator
from consortium.tree_search.failure_memory import FailureMemory, FailureRecord
from consortium.tree_search.node_evaluator import ScoreCalibrator
from consortium.tree_search.tree_visualization import (
    ascii_tree,
    mermaid_diagram,
    summary_table,
)
from consortium.tree_search.experiment_tree_integration import (
    ExperimentStrategy,
    build_experiment_tree_controller,
    build_tree_search_experiment_track,
    generate_experiment_strategies,
)

__all__ = [
    # State
    "CounselMode",
    "NodeStatus",
    "NodeType",
    "TreeNode",
    "TreeSearchConfig",
    "TreeSearchState",
    # Persistence
    "create_tree_state",
    "ensure_tree_state",
    "load_tree_state",
    "save_tree_state",
    # Manager
    "get_downstream_impact",
    "get_frontier_claims",
    "process_branch_result",
    "run_tree_search_step",
    "select_branches_for_execution",
    # Workspace
    "fork_workspace",
    "merge_branch",
    "populate_branch",
    "promote_branch",
    "retarget_tools",
    # Budget
    "TreeBudgetAllocator",
    # Failure memory & calibration
    "FailureMemory",
    "FailureRecord",
    "ScoreCalibrator",
    # Visualization
    "ascii_tree",
    "mermaid_diagram",
    "summary_table",
    # Experiment tree search
    "ExperimentStrategy",
    "build_experiment_tree_controller",
    "build_tree_search_experiment_track",
    "generate_experiment_strategies",
]
