"""LangGraph integration for agentic tree search.

Provides:
- ``build_tree_search_theory_track``: a replacement theory track subgraph
  that inserts a tree search controller between the proposer and prover.
- ``build_tree_search_controller``: the LangGraph node that orchestrates
  branching, scoring, and merging for the proof stage.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional

from langgraph.graph import END, StateGraph
from consortium.state import ResearchState
from consortium.tree_search.tree_manager import (
    _load_claim_graph,
    create_proof_branches,
    get_frontier_claims,
    process_branch_result,
    run_tree_search_step,
    select_branches_for_execution,
)
from consortium.tree_search.tree_persistence import ensure_tree_state, save_tree_state
from consortium.tree_search.tree_state import (
    NodeStatus,
    TreeNode,
    TreeSearchConfig,
    TreeSearchState,
)
from consortium.tree_search.workspace_fork import promote_branch


# ---------------------------------------------------------------------------
# Tree search controller node
# ---------------------------------------------------------------------------

def build_tree_search_controller(
    prover_node: Callable,
    rigorous_verifier_node: Callable,
    empirical_verifier_node: Callable,
    workspace_dir: str,
    tree_config: TreeSearchConfig,
    *,
    model_id: str = "claude-sonnet-4-6",
) -> Callable:
    """Build a LangGraph node that runs the tree search loop for the theory track.

    Instead of a single pass through prover → verifier → verifier, this node:
    1. Loads the claim graph and tree state
    2. Identifies frontier claims (dependencies resolved)
    3. Generates proof strategy branches for each frontier claim
    4. Executes branches in parallel (each runs prover + verifiers)
    5. Evaluates results, promotes winners, creates debugging branches for failures
    6. Repeats until all claims are resolved or budget is exhausted
    """

    def tree_search_node(state: dict) -> dict:
        ws = state.get("workspace_dir", workspace_dir)
        tree_state = ensure_tree_state(ws, tree_config)

        # Populate completed_claims from existing claim graph
        claim_graph = _load_claim_graph(ws)
        for c in claim_graph.get("claims", []):
            if c.get("status") in ("accepted", "verified_numeric", "verified_symbolic"):
                tree_state.mark_claim_resolved(c["id"])

        max_iterations = tree_config.max_depth * tree_config.max_breadth
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Get branches to execute
            branches = run_tree_search_step(tree_state, ws, model=model_id)
            if not branches:
                break  # No more work to do

            # Execute branches in parallel
            results = _execute_branches_parallel(
                branches,
                prover_node=prover_node,
                rigorous_verifier_node=rigorous_verifier_node,
                empirical_verifier_node=empirical_verifier_node,
                state=state,
                tree_state=tree_state,
                claim_graph=claim_graph,
                workspace_dir=ws,
                max_parallel=tree_config.max_parallel,
            )

            # Process results
            any_succeeded = False
            for node, success, failure_reason in results:
                process_branch_result(
                    node,
                    success,
                    tree_state,
                    claim_graph,
                    ws,
                    failure_reason=failure_reason,
                    model=model_id,
                )
                if success:
                    any_succeeded = True

            # Reload claim graph after promotions
            claim_graph = _load_claim_graph(ws)

            # Check if all must_accept claims are resolved
            if _all_must_accept_resolved(claim_graph, tree_state):
                break

        save_tree_state(tree_state, ws)

        # Add tree search summary to agent outputs
        summary = tree_state.summary()
        return {
            "agent_outputs": {
                "tree_search_controller": json.dumps(summary, indent=2),
            },
            "tree_state_path": os.path.join(ws, "tree_search_state.json"),
            "agent_task": None,
        }

    tree_search_node.__name__ = "tree_search_controller"
    return tree_search_node


def _execute_branches_parallel(
    branches: list[TreeNode],
    *,
    prover_node: Callable,
    rigorous_verifier_node: Callable,
    empirical_verifier_node: Callable,
    state: dict,
    tree_state: TreeSearchState,
    claim_graph: dict,
    workspace_dir: str,
    max_parallel: int = 6,
) -> list[tuple[TreeNode, bool, str]]:
    """Execute proof branches in parallel.

    Each branch runs the prover → rigorous_verifier → empirical_verifier
    pipeline in its own forked workspace.

    Returns a list of (node, success, failure_reason) tuples.
    """
    results: list[tuple[TreeNode, bool, str]] = []

    def _run_branch(node: TreeNode) -> tuple[TreeNode, bool, str]:
        node.mark_running()
        save_tree_state(tree_state, workspace_dir)

        # Build branch state with strategy-specific task
        strategy_directive = node.metadata.get("prompt_directive", "")
        claim_id = node.claim_id or "unknown"
        branch_task = (
            f"[TREE SEARCH BRANCH — Strategy: {node.metadata.get('strategy_name', 'default')}]\n\n"
            f"Focus on proving claim '{claim_id}' using the following strategy:\n\n"
            f"{strategy_directive}\n\n"
            f"Work in the current workspace. Use the math_claim_graph_tool and "
            f"math_proof_workspace_tool to draft and record your proof. "
            f"Set the claim status to 'proved_draft' when complete."
        )

        branch_state = {
            **state,
            "workspace_dir": node.workspace_path,
            "agent_task": branch_task,
            "active_branch_id": node.id,
        }

        try:
            # Run prover
            prover_result = prover_node(branch_state)
            branch_state = {**branch_state, **prover_result}

            # Run rigorous verifier
            branch_state["agent_task"] = (
                f"Verify the proof of claim '{claim_id}' that was just drafted. "
                f"Check for logical gaps, missing steps, and mathematical correctness."
            )
            verifier_result = rigorous_verifier_node(branch_state)
            branch_state = {**branch_state, **verifier_result}

            # Run empirical verifier
            branch_state["agent_task"] = (
                f"Numerically verify claim '{claim_id}'. "
                f"Check that the theoretical predictions match numerical experiments."
            )
            empirical_result = empirical_verifier_node(branch_state)

            # Check if the claim was successfully verified
            branch_claim_graph = _load_claim_graph(node.workspace_path)
            claim_status = _get_claim_status(branch_claim_graph, claim_id)

            if claim_status in ("verified_symbolic", "verified_numeric", "accepted"):
                return (node, True, "")
            else:
                # Check verification output for gap information
                verifier_output = branch_state.get("agent_outputs", {}).get(
                    "math_rigorous_verifier_agent", ""
                )
                return (node, False, f"Claim status: {claim_status}. {verifier_output[:500]}")

        except Exception as exc:
            return (node, False, f"Branch execution error: {exc}")

    with ThreadPoolExecutor(max_workers=min(len(branches), max_parallel)) as executor:
        futures = {executor.submit(_run_branch, node): node for node in branches}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                node = futures[future]
                results.append((node, False, f"Future error: {exc}"))

    return results


def _get_claim_status(claim_graph: dict, claim_id: str) -> str:
    """Get the status of a specific claim in the graph."""
    for c in claim_graph.get("claims", []):
        if c["id"] == claim_id:
            return c.get("status", "proposed")
    return "unknown"


def _all_must_accept_resolved(
    claim_graph: dict,
    tree_state: TreeSearchState,
) -> bool:
    """Check if all must_accept claims are in the completed set."""
    for c in claim_graph.get("claims", []):
        if c.get("must_accept", False):
            if c["id"] not in tree_state.completed_claims:
                return False
    return True


# ---------------------------------------------------------------------------
# Enhanced theory track subgraph with tree search
# ---------------------------------------------------------------------------

def build_tree_search_theory_track(
    model: Any,
    workspace_dir: str,
    authorized_imports: Optional[List[str]] = None,
    counsel_models: Optional[List[Any]] = None,
    summary_model_id: Optional[str] = "claude-sonnet-4-6",
    tree_config: Optional[TreeSearchConfig] = None,
    model_id: str = "claude-sonnet-4-6",
) -> Any:
    """Build the theory track subgraph with tree search integration.

    When tree search is enabled, the pipeline becomes:
        math_literature → math_proposer → [TREE SEARCH CONTROLLER] → proof_transcription

    The tree search controller internally manages parallel execution of
    math_prover + verifier chains across multiple proof strategies.

    When tree search is disabled, falls back to the standard linear pipeline.
    """
    from consortium.agents import (
        build_math_empirical_verifier_node,
        build_math_literature_node,
        build_math_proposer_node,
        build_math_prover_node,
        build_math_rigorous_verifier_node,
        build_proof_transcription_node,
    )
    from consortium.pdf_summary import with_pdf_summary

    graph = StateGraph(ResearchState)
    counsel_kwargs = {"counsel_models": counsel_models} if counsel_models else {}

    def _wrap(node, name):
        return with_pdf_summary(node, name, workspace_dir, summary_model_id)

    # Literature and proposer stages are unchanged
    graph.add_node(
        "math_literature_agent",
        _wrap(
            build_math_literature_node(model, workspace_dir, authorized_imports, **counsel_kwargs),
            "math_literature_agent",
        ),
    )
    graph.add_node(
        "math_proposer_agent",
        _wrap(
            build_math_proposer_node(model, workspace_dir, authorized_imports, **counsel_kwargs),
            "math_proposer_agent",
        ),
    )

    if tree_config and tree_config.enabled:
        # Build the raw agent nodes (unwrapped) for use inside the tree controller
        prover = build_math_prover_node(model, workspace_dir, authorized_imports, **counsel_kwargs)
        rigorous_verifier = build_math_rigorous_verifier_node(
            model, workspace_dir, authorized_imports, **counsel_kwargs
        )
        empirical_verifier = build_math_empirical_verifier_node(
            model, workspace_dir, authorized_imports, **counsel_kwargs
        )

        tree_controller = build_tree_search_controller(
            prover_node=prover,
            rigorous_verifier_node=rigorous_verifier,
            empirical_verifier_node=empirical_verifier,
            workspace_dir=workspace_dir,
            tree_config=tree_config,
            model_id=model_id,
        )

        graph.add_node("tree_search_controller", tree_controller)
    else:
        # Standard linear pipeline nodes
        graph.add_node(
            "math_prover_agent",
            _wrap(
                build_math_prover_node(model, workspace_dir, authorized_imports, **counsel_kwargs),
                "math_prover_agent",
            ),
        )
        graph.add_node(
            "math_rigorous_verifier_agent",
            _wrap(
                build_math_rigorous_verifier_node(model, workspace_dir, authorized_imports, **counsel_kwargs),
                "math_rigorous_verifier_agent",
            ),
        )
        graph.add_node(
            "math_empirical_verifier_agent",
            _wrap(
                build_math_empirical_verifier_node(model, workspace_dir, authorized_imports, **counsel_kwargs),
                "math_empirical_verifier_agent",
            ),
        )

    graph.add_node(
        "proof_transcription_agent",
        _wrap(
            build_proof_transcription_node(model, workspace_dir, authorized_imports, **counsel_kwargs),
            "proof_transcription_agent",
        ),
    )

    # Wire edges
    graph.set_entry_point("math_literature_agent")
    graph.add_edge("math_literature_agent", "math_proposer_agent")

    if tree_config and tree_config.enabled:
        graph.add_edge("math_proposer_agent", "tree_search_controller")
        graph.add_edge("tree_search_controller", "proof_transcription_agent")
    else:
        graph.add_edge("math_proposer_agent", "math_prover_agent")
        graph.add_edge("math_prover_agent", "math_rigorous_verifier_agent")
        graph.add_edge("math_rigorous_verifier_agent", "math_empirical_verifier_agent")
        graph.add_edge("math_empirical_verifier_agent", "proof_transcription_agent")

    graph.add_edge("proof_transcription_agent", END)
    return graph.compile()
