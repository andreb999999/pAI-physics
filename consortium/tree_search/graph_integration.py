"""LangGraph integration for agentic tree search.

Provides:
- ``build_tree_search_theory_track``: a replacement theory track subgraph
  that inserts a tree search controller between the proposer and prover.
- ``tree_search_controller_node``: the LangGraph node that orchestrates
  branching, scoring, and merging for the proof stage.
- ``tree_search_track_router``: enhanced track router that initialises tree
  state when tree search is enabled.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional

from langgraph.graph import END, StateGraph
from langgraph.types import Send

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
    CounselMode,
    NodeStatus,
    NodeType,
    TreeNode,
    TreeSearchConfig,
    TreeSearchState,
)
from consortium.tree_search.workspace_fork import fork_workspace, promote_branch


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


# ---------------------------------------------------------------------------
# Phase C: Ideation tree search
# ---------------------------------------------------------------------------

def build_ideation_tree_router(
    tree_config: TreeSearchConfig,
    workspace_dir: str,
    *,
    model_id: str = "claude-sonnet-4-6",
) -> Callable:
    """Build a node that branches ideation into parallel idea variants.

    When tree search is enabled, generates N idea variants in parallel,
    develops each to the literature-review stage, then selects the best.
    Returns a router function that produces ``Send`` messages.
    """
    from consortium.tree_search.strategy_generator import generate_idea_variants

    def ideation_tree_node(state: dict) -> list[Send]:
        if not tree_config.enabled:
            return [Send("literature_review_agent", state)]

        task = state.get("task", "")
        ws = state.get("workspace_dir", workspace_dir)

        # Initialize tree state for ideation
        tree_state = ensure_tree_state(ws, tree_config)

        try:
            variants = generate_idea_variants(
                task,
                n=tree_config.max_breadth,
                model=model_id,
            )
        except Exception:
            # Fallback to single idea path
            return [Send("literature_review_agent", state)]

        if not variants:
            return [Send("literature_review_agent", state)]

        sends: list[Send] = []
        for variant in variants:
            branch_id = TreeSearchState.make_node_id(f"idea_{variant.name}")
            branch_dir = fork_workspace(ws, branch_id)

            node = TreeNode(
                id=branch_id,
                node_type=NodeType.IDEA_VARIANT,
                parent_id="root",
                strategy_description=variant.description,
                workspace_path=branch_dir,
                depth=1,
                metadata={
                    "prompt_directive": variant.prompt_directive,
                    "novelty_rationale": variant.novelty_rationale,
                },
            )
            tree_state.add_node(node)

            sends.append(
                Send(
                    "ideation_branch",
                    {
                        **state,
                        "workspace_dir": branch_dir,
                        "agent_task": (
                            f"[TREE SEARCH BRANCH — Idea: {variant.name}]\n\n"
                            f"{variant.prompt_directive}\n\n"
                            f"Develop this idea fully: generate a research idea JSON, "
                            f"check its novelty, and refine it."
                        ),
                        "active_branch_id": branch_id,
                    },
                )
            )

        save_tree_state(tree_state, ws)
        return sends

    ideation_tree_node.__name__ = "ideation_tree_router"
    return ideation_tree_node


def build_idea_selector(
    workspace_dir: str,
    tree_config: TreeSearchConfig,
    *,
    model_id: str = "claude-sonnet-4-6",
) -> Callable:
    """Build a merge node that selects the best idea from parallel branches.

    Evaluates each branch's ideation output (novelty, tractability, impact)
    and promotes the winning branch's workspace.
    """
    from consortium.tree_search.tree_persistence import load_tree_state, save_tree_state

    def idea_selector_node(state: dict) -> dict:
        ws = state.get("workspace_dir", workspace_dir)
        tree_state = load_tree_state(ws)
        if tree_state is None:
            return {"agent_task": None}

        # Find all idea variant nodes
        idea_nodes = [
            n for n in tree_state.nodes.values()
            if n.node_type == NodeType.IDEA_VARIANT
            and n.status in (NodeStatus.PENDING, NodeStatus.RUNNING, NodeStatus.SUCCEEDED)
        ]

        if not idea_nodes:
            return {"agent_task": None}

        # Score and select the best idea
        best_node = max(idea_nodes, key=lambda n: n.score)
        best_node.mark_succeeded()

        # Promote the winning idea's workspace
        promote_branch(best_node.workspace_path, ws)

        # Prune losing ideas
        for n in idea_nodes:
            if n.id != best_node.id:
                n.mark_pruned()

        save_tree_state(tree_state, ws)

        return {
            "agent_outputs": {
                "idea_selector": f"Selected idea: {best_node.strategy_description}",
            },
            "agent_task": None,
        }

    idea_selector_node.__name__ = "idea_selector"
    return idea_selector_node


# ---------------------------------------------------------------------------
# Phase D: Experiment track tree search
# ---------------------------------------------------------------------------

def build_experiment_tree_router(
    tree_config: TreeSearchConfig,
    workspace_dir: str,
    *,
    model_id: str = "claude-sonnet-4-6",
) -> Callable:
    """Build a router that branches experiment designs in parallel.

    Generates N experiment design variants and sends each to a parallel
    experiment track execution.
    """
    from consortium.tree_search.strategy_generator import generate_experiment_variants

    def experiment_tree_node(state: dict) -> list[Send]:
        if not tree_config.enabled:
            return [Send("experiment_track", state)]

        ws = state.get("workspace_dir", workspace_dir)
        track_decomposition = state.get("track_decomposition") or {}
        empirical_questions = list(track_decomposition.get("empirical_questions") or [])

        if not empirical_questions:
            return [Send("experiment_track", state)]

        tree_state = ensure_tree_state(ws, tree_config)

        try:
            variants = generate_experiment_variants(
                empirical_questions,
                n=min(tree_config.max_breadth, 3),
                model=model_id,
            )
        except Exception:
            return [Send("experiment_track", state)]

        sends: list[Send] = []
        for variant in variants:
            branch_id = TreeSearchState.make_node_id(f"exp_{variant.name}")
            branch_dir = fork_workspace(ws, branch_id)

            node = TreeNode(
                id=branch_id,
                node_type=NodeType.EXPERIMENT_DESIGN,
                parent_id="root",
                strategy_description=variant.description,
                workspace_path=branch_dir,
                depth=1,
                metadata={
                    "prompt_directive": variant.prompt_directive,
                    "rationale": variant.rationale,
                },
            )
            tree_state.add_node(node)

            sends.append(
                Send(
                    "experiment_branch",
                    {
                        **state,
                        "workspace_dir": branch_dir,
                        "agent_task": (
                            f"[TREE SEARCH BRANCH — Experiment: {variant.name}]\n\n"
                            f"{variant.prompt_directive}\n\n"
                            f"Design and execute this experimental approach. "
                            f"Focus on producing clear, reproducible results."
                        ),
                        "experiment_track_status": "in_progress",
                        "active_branch_id": branch_id,
                    },
                )
            )

        save_tree_state(tree_state, ws)
        return sends

    experiment_tree_node.__name__ = "experiment_tree_router"
    return experiment_tree_node


# ---------------------------------------------------------------------------
# Phase E: Follow-up loop branching
# ---------------------------------------------------------------------------

def build_followup_tree_router(
    tree_config: TreeSearchConfig,
    workspace_dir: str,
    *,
    model_id: str = "claude-sonnet-4-6",
) -> Callable:
    """Build a router that branches follow-up cycles into parallel gap-fixing directions.

    Instead of linear cycling (plan → execute → analyze → repeat), this
    branches into parallel directions: one fixes proof gaps, another extends
    results, another designs new experiments.
    """

    def followup_tree_node(state: dict) -> list[Send] | str:
        # If tree search is disabled or the gate says to continue, use normal routing
        current_agent = state.get("current_agent")
        if not tree_config.enabled or current_agent != "research_planner_agent":
            return current_agent or "resource_preparation_agent"

        ws = state.get("workspace_dir", workspace_dir)
        tree_state = ensure_tree_state(ws, tree_config)

        # Analyze what gaps exist
        claim_graph = _load_claim_graph(ws)
        unresolved = [
            c for c in claim_graph.get("claims", [])
            if c.get("must_accept") and c.get("status") not in ("accepted", "verified_numeric")
        ]

        if not unresolved:
            return "resource_preparation_agent"

        # Create parallel follow-up branches
        sends: list[Send] = []
        directions = _identify_followup_directions(unresolved, state)

        for i, direction in enumerate(directions[:tree_config.max_breadth]):
            branch_id = TreeSearchState.make_node_id(f"followup_{direction['type']}")
            branch_dir = fork_workspace(ws, branch_id)

            node = TreeNode(
                id=branch_id,
                node_type=(
                    NodeType.FOLLOWUP_GAP_FIX
                    if direction["type"] == "gap_fix"
                    else NodeType.FOLLOWUP_EXTENSION
                ),
                parent_id="root",
                strategy_description=direction["description"],
                workspace_path=branch_dir,
                depth=1,
                metadata=direction,
            )
            tree_state.add_node(node)

            sends.append(
                Send(
                    "research_planner_agent",
                    {
                        **state,
                        "workspace_dir": branch_dir,
                        "agent_task": direction["task"],
                        "active_branch_id": branch_id,
                    },
                )
            )

        save_tree_state(tree_state, ws)

        if sends:
            return sends
        return "resource_preparation_agent"

    followup_tree_node.__name__ = "followup_tree_router"
    return followup_tree_node


def _identify_followup_directions(
    unresolved_claims: list[dict],
    state: dict,
) -> list[dict]:
    """Categorise gaps into follow-up directions."""
    directions = []

    # Direction 1: Fix proof gaps for blocking claims
    blocking = [c for c in unresolved_claims if c.get("status") in ("proposed", "proved_draft")]
    if blocking:
        claim_ids = ", ".join(c["id"] for c in blocking[:3])
        directions.append({
            "type": "gap_fix",
            "description": f"Fix proof gaps for claims: {claim_ids}",
            "task": (
                f"Focus this research cycle on fixing the proof gaps for "
                f"the following unresolved claims: {claim_ids}. "
                f"Read the verification reports and address the identified issues."
            ),
        })

    # Direction 2: Try alternative approaches
    if len(blocking) > 0:
        directions.append({
            "type": "alternative",
            "description": "Explore alternative proof approaches for stuck claims",
            "task": (
                f"The current proof approaches for some claims are blocked. "
                f"Explore fundamentally different approaches: alternative lemma "
                f"decompositions, different key inequalities, or bypassing "
                f"problematic dependencies entirely."
            ),
        })

    # Direction 3: Strengthen empirical evidence
    directions.append({
        "type": "empirical",
        "description": "Design additional experiments to strengthen evidence",
        "task": (
            "Design new experiments that provide stronger empirical support "
            "for the theoretical claims. Focus on edge cases, scaling behaviour, "
            "and comparison with alternative methods."
        ),
    })

    return directions
