"""Agentic tree search for phdlabor-1.

Implements DAG-layered best-first search for the theory track (proof stage).
Inspired by AI Scientist-v2 progressive agentic tree search, adapted for
mathematical research with claim-graph-aware branching.
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
from consortium.tree_search.tree_visualization import (
    ascii_tree,
    mermaid_diagram,
    summary_table,
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
    # Visualization
    "ascii_tree",
    "mermaid_diagram",
    "summary_table",
]
