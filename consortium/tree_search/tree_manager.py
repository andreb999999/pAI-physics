"""DAG-layered best-first search controller for the theory track.

Orchestrates the tree search loop: identify frontier claims in topological
order, generate proof strategies, score them, select the top-K, and prepare
``Send`` descriptors for parallel execution by LangGraph.

This module is stateless — all persistent data lives in :class:`TreeSearchState`
(on disk) and :class:`ResearchState` (in the LangGraph graph).
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from consortium.tree_search.node_evaluator import rescore_all, score_node
from consortium.tree_search.strategy_generator import (
    generate_experiment_variants,
    generate_idea_variants,
    generate_proof_strategies,
    load_prior_proof,
    load_verification_gaps,
)
from consortium.tree_search.tree_persistence import load_tree_state, save_tree_state
from consortium.tree_search.tree_state import (
    NodeStatus,
    NodeType,
    TreeNode,
    TreeSearchConfig,
    TreeSearchState,
)
from consortium.tree_search.workspace_fork import fork_workspace


# ---------------------------------------------------------------------------
# Claim-graph helpers
# ---------------------------------------------------------------------------

def _load_claim_graph(workspace_dir: str) -> dict[str, Any]:
    """Load ``math_workspace/claim_graph.json`` from *workspace_dir*."""
    path = os.path.join(workspace_dir, "math_workspace", "claim_graph.json")
    if not os.path.exists(path):
        return {"claims": []}
    with open(path) as f:
        return json.load(f)


def get_frontier_claims(
    claim_graph: dict[str, Any],
    completed_claims: list[str],
) -> list[dict[str, Any]]:
    """Return claims whose dependencies are all resolved.

    A claim is in the *frontier* when:
    1. It is **not** in ``completed_claims`` (i.e. not yet resolved).
    2. Every claim listed in its ``depends_on`` **is** in ``completed_claims``.

    The frontier is the set of claims that can be worked on right now.
    """
    claims = claim_graph.get("claims", [])
    completed = set(completed_claims)
    frontier = []
    for c in claims:
        cid = c["id"]
        if cid in completed:
            continue
        # Status-based skip: already accepted or rejected
        status = c.get("status", "proposed")
        if status in ("accepted", "rejected"):
            continue
        deps = c.get("depends_on", [])
        if all(d in completed for d in deps):
            frontier.append(c)
    return frontier


def get_downstream_impact(
    claim_id: str,
    claim_graph: dict[str, Any],
) -> int:
    """Count transitive dependents of *claim_id* in the claim graph."""
    claims = claim_graph.get("claims", [])
    dependents: dict[str, set[str]] = {}
    for c in claims:
        for dep in c.get("depends_on", []):
            dependents.setdefault(dep, set()).add(c["id"])
    visited: set[str] = set()
    stack = [claim_id]
    while stack:
        cid = stack.pop()
        if cid in visited:
            continue
        visited.add(cid)
        stack.extend(dependents.get(cid, set()))
    return len(visited) - 1  # exclude the claim itself


def _topological_sort_frontier(
    frontier: list[dict[str, Any]],
    claim_graph: dict[str, Any],
) -> list[dict[str, Any]]:
    """Sort frontier claims by downstream impact (highest first).

    Claims that unblock the most downstream work are prioritised.
    """
    return sorted(
        frontier,
        key=lambda c: get_downstream_impact(c["id"], claim_graph),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Branch creation
# ---------------------------------------------------------------------------

def create_proof_branches(
    claim: dict[str, Any],
    tree_state: TreeSearchState,
    claim_graph: dict[str, Any],
    workspace_dir: str,
    *,
    model: str = "claude-sonnet-4-6",
) -> list[TreeNode]:
    """Generate proof strategy branches for *claim* and add them to the tree.

    Returns the list of newly created TreeNode objects.
    """
    config = tree_state.config
    claim_id = claim["id"]

    # Load context for the strategy generator
    prior_proof = load_prior_proof(workspace_dir, claim_id)
    gaps = load_verification_gaps(workspace_dir, claim_id)

    # Build a brief claim graph context string
    cg_summary_parts = []
    for c in claim_graph.get("claims", []):
        cg_summary_parts.append(
            f"  {c['id']} [{c.get('status', '?')}]: {c.get('statement', '')[:120]}"
        )
    cg_context = "Current claim graph:\n" + "\n".join(cg_summary_parts)

    strategies = generate_proof_strategies(
        claim,
        n=config.max_breadth,
        prior_proof=prior_proof,
        verification_gaps=gaps,
        claim_graph_context=cg_context,
        model=model,
    )

    nodes: list[TreeNode] = []
    root = tree_state.get_root()
    parent_id = root.id if root else None

    # Check if branches for this claim already exist
    existing_claim_branches = [
        n for n in tree_state.nodes.values()
        if n.claim_id == claim_id and n.node_type == NodeType.PROOF_STRATEGY
        and n.status not in (NodeStatus.FAILED, NodeStatus.PRUNED)
    ]
    if existing_claim_branches:
        # Already have active branches for this claim — skip
        return []

    per_branch_budget = (
        tree_state.config.budget_fraction
        / max(config.max_breadth, 1)
    )

    for strategy in strategies:
        node_id = TreeSearchState.make_node_id(f"{claim_id}_{strategy.name}")
        branch_dir = fork_workspace(workspace_dir, node_id)

        node = TreeNode(
            id=node_id,
            node_type=NodeType.PROOF_STRATEGY,
            parent_id=parent_id,
            claim_id=claim_id,
            strategy_description=strategy.description,
            workspace_path=branch_dir,
            depth=(tree_state.nodes[parent_id].depth + 1) if parent_id and parent_id in tree_state.nodes else 1,
            budget_cap_usd=per_branch_budget,
            metadata={
                "strategy_name": strategy.name,
                "prompt_directive": strategy.prompt_directive,
                "estimated_difficulty": strategy.estimated_difficulty,
                "rationale": strategy.rationale,
                "prior_gaps": gaps,
            },
        )
        node.score = score_node(node, tree_state, claim_graph, model=model)
        tree_state.add_node(node)
        tree_state.budget_allocations[node_id] = per_branch_budget
        nodes.append(node)

    save_tree_state(tree_state, workspace_dir)
    return nodes


def create_debugging_branches(
    failed_node: TreeNode,
    tree_state: TreeSearchState,
    claim_graph: dict[str, Any],
    workspace_dir: str,
    *,
    model: str = "claude-sonnet-4-6",
) -> list[TreeNode]:
    """Create child branches for a failed proof attempt.

    Generates alternative strategies that explicitly address the gaps found
    in the failed attempt.
    """
    config = tree_state.config
    if failed_node.depth >= config.max_depth:
        return []

    claim_id = failed_node.claim_id
    if not claim_id:
        return []

    # Load the claim from the graph
    claims = claim_graph.get("claims", [])
    claim = next((c for c in claims if c["id"] == claim_id), None)
    if claim is None:
        return []

    # Use the failed node's workspace for context
    branch_ws = failed_node.workspace_path or workspace_dir
    prior_proof = load_prior_proof(branch_ws, claim_id)
    gaps = load_verification_gaps(branch_ws, claim_id)

    # Add context about what failed
    gap_context = (
        f"The previous attempt ('{failed_node.strategy_description}') FAILED. "
        f"Gaps found: {json.dumps(gaps, indent=2) if gaps else 'unknown'}. "
        f"Generate strategies that specifically address or avoid these issues."
    )

    strategies = generate_proof_strategies(
        claim,
        n=min(config.max_breadth, 2),  # fewer debugging branches
        prior_proof=prior_proof,
        verification_gaps=gaps,
        claim_graph_context=gap_context,
        model=model,
    )

    nodes: list[TreeNode] = []
    per_branch_budget = failed_node.budget_cap_usd * 0.5  # half of parent's remaining budget

    for strategy in strategies:
        node_id = TreeSearchState.make_node_id(f"{claim_id}_debug_{strategy.name}")
        branch_dir = fork_workspace(workspace_dir, node_id)

        node = TreeNode(
            id=node_id,
            node_type=NodeType.DEBUGGING,
            parent_id=failed_node.id,
            claim_id=claim_id,
            strategy_description=f"[DEBUG] {strategy.description}",
            workspace_path=branch_dir,
            depth=failed_node.depth + 1,
            budget_cap_usd=per_branch_budget,
            metadata={
                "strategy_name": strategy.name,
                "prompt_directive": strategy.prompt_directive,
                "parent_failure_reason": failed_node.metadata.get("failure_reason", ""),
                "prior_gaps": gaps,
            },
        )
        node.score = score_node(node, tree_state, claim_graph, model=model)
        tree_state.add_node(node)
        nodes.append(node)

    save_tree_state(tree_state, workspace_dir)
    return nodes


# ---------------------------------------------------------------------------
# Branch selection
# ---------------------------------------------------------------------------

def select_branches_for_execution(
    tree_state: TreeSearchState,
    claim_graph: dict[str, Any],
    *,
    model: str = "claude-sonnet-4-6",
) -> list[TreeNode]:
    """Select the top-K pending nodes for parallel execution.

    Rescores all pending nodes, prunes below threshold, and returns
    the best candidates up to ``config.max_parallel``.
    """
    config = tree_state.config

    # Rescore (cheap — skips LLM by default)
    rescore_all(tree_state, claim_graph, model=model, skip_llm=True)

    # Prune low scorers
    tree_state.prune_below_threshold(config.pruning_threshold)

    # Select top-K
    return tree_state.get_top_k(config.max_parallel)


# ---------------------------------------------------------------------------
# Result processing
# ---------------------------------------------------------------------------

def process_branch_result(
    node: TreeNode,
    success: bool,
    tree_state: TreeSearchState,
    claim_graph: dict[str, Any],
    workspace_dir: str,
    *,
    failure_reason: str = "",
    model: str = "claude-sonnet-4-6",
) -> None:
    """Process the result of a completed branch execution.

    If successful, marks the claim as resolved and promotes the branch
    workspace. If failed, optionally creates debugging children.
    """
    if success:
        node.mark_succeeded()
        if node.claim_id:
            tree_state.mark_claim_resolved(node.claim_id)
        # Promote winning workspace
        from consortium.tree_search.workspace_fork import promote_branch
        promote_branch(node.workspace_path, workspace_dir)
        # Prune sibling branches for the same claim
        _prune_siblings(node, tree_state)
    else:
        node.mark_failed()
        node.metadata["failure_reason"] = failure_reason
        # Create debugging branches if within depth limit
        create_debugging_branches(
            node, tree_state, claim_graph, workspace_dir, model=model
        )

    save_tree_state(tree_state, workspace_dir)


def _prune_siblings(node: TreeNode, tree_state: TreeSearchState) -> None:
    """Prune all siblings of *node* that target the same claim.

    Once a proof strategy succeeds, there's no need to continue exploring
    alternative strategies for the same claim.
    """
    if not node.claim_id:
        return
    for n in list(tree_state.nodes.values()):
        if (
            n.id != node.id
            and n.claim_id == node.claim_id
            and not n.is_terminal
        ):
            tree_state.prune_subtree(n.id)


# ---------------------------------------------------------------------------
# Full DAG-layered search step
# ---------------------------------------------------------------------------

def run_tree_search_step(
    tree_state: TreeSearchState,
    workspace_dir: str,
    *,
    model: str = "claude-sonnet-4-6",
) -> list[TreeNode]:
    """Execute one step of DAG-layered best-first search.

    1. Load claim graph.
    2. Identify frontier claims (dependencies resolved).
    3. For each frontier claim without active branches, create proof branches.
    4. Select top-K branches for execution.
    5. Return the selected nodes (caller is responsible for executing them).
    """
    claim_graph = _load_claim_graph(workspace_dir)

    # Step 1: Identify frontier claims
    frontier = get_frontier_claims(claim_graph, tree_state.completed_claims)
    frontier = _topological_sort_frontier(frontier, claim_graph)

    if not frontier:
        return []

    # Step 2: Create branches for frontier claims that lack them
    for claim in frontier:
        create_proof_branches(
            claim, tree_state, claim_graph, workspace_dir, model=model
        )

    # Step 3: Select best branches for execution
    return select_branches_for_execution(tree_state, claim_graph, model=model)
