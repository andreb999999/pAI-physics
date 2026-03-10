"""JSON-based persistence for tree search state.

The tree state file lives at ``<workspace_dir>/tree_search_state.json``.
This module provides load/save/create operations, integrating with
LangGraph's SqliteSaver for graph-level resumability while keeping
tree-level state independently serialised.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from consortium.tree_search.tree_state import (
    NodeType,
    TreeNode,
    TreeSearchConfig,
    TreeSearchState,
)

TREE_STATE_FILENAME = "tree_search_state.json"


def tree_state_path(workspace_dir: str) -> str:
    """Return the canonical path for the tree state file."""
    return os.path.join(workspace_dir, TREE_STATE_FILENAME)


def save_tree_state(state: TreeSearchState, workspace_dir: str) -> str:
    """Serialise *state* to ``<workspace_dir>/tree_search_state.json``.

    Returns the absolute path of the written file.
    """
    path = tree_state_path(workspace_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state.to_dict(), f, indent=2)
    return path


def load_tree_state(workspace_dir: str) -> Optional[TreeSearchState]:
    """Load tree state from disk.  Returns ``None`` if the file does not exist."""
    path = tree_state_path(workspace_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return TreeSearchState.from_dict(data)


def create_tree_state(
    workspace_dir: str,
    config: TreeSearchConfig,
) -> TreeSearchState:
    """Initialise a fresh tree state with a ROOT node and persist it.

    Returns the new TreeSearchState.
    """
    state = TreeSearchState(config=config)
    root = TreeNode(
        id="root",
        node_type=NodeType.ROOT,
        workspace_path=workspace_dir,
    )
    root.mark_running()
    state.add_node(root)

    save_tree_state(state, workspace_dir)
    return state


def ensure_tree_state(
    workspace_dir: str,
    config: TreeSearchConfig,
) -> TreeSearchState:
    """Load existing tree state or create a new one."""
    existing = load_tree_state(workspace_dir)
    if existing is not None:
        return existing
    return create_tree_state(workspace_dir, config)
