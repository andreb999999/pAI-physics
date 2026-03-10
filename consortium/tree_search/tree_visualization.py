"""ASCII and Mermaid visualization of the search tree.

Produces human-readable tree summaries for logs and optional Mermaid
diagram export for documentation/reports.
"""

from __future__ import annotations

from typing import Optional

from consortium.tree_search.tree_state import NodeStatus, TreeNode, TreeSearchState


# ---------------------------------------------------------------------------
# ASCII tree
# ---------------------------------------------------------------------------

_STATUS_ICONS = {
    NodeStatus.PENDING: "[ ]",
    NodeStatus.RUNNING: "[~]",
    NodeStatus.SUCCEEDED: "[+]",
    NodeStatus.FAILED: "[X]",
    NodeStatus.PRUNED: "[-]",
}


def ascii_tree(tree_state: TreeSearchState) -> str:
    """Render the search tree as an indented ASCII string.

    Example output::

        [~] root (ROOT)
          [+] L4_strategy_A (proof_strategy) score=0.82 claim=L4
          [X] L4_strategy_B (proof_strategy) score=0.45 claim=L4
            [-] L4_debug_alt (debugging) score=0.30 claim=L4
          [ ] T1_strategy_A (proof_strategy) score=0.70 claim=T1
    """
    root = tree_state.get_root()
    if root is None:
        return "(empty tree)"
    lines: list[str] = []
    _render_node(root, tree_state, lines, indent=0)
    return "\n".join(lines)


def _render_node(
    node: TreeNode,
    tree_state: TreeSearchState,
    lines: list[str],
    indent: int,
) -> None:
    icon = _STATUS_ICONS.get(node.status, "[?]")
    parts = [f"{icon} {node.id} ({node.node_type.value})"]
    if node.score > 0:
        parts.append(f"score={node.score:.2f}")
    if node.claim_id:
        parts.append(f"claim={node.claim_id}")
    if node.cost_usd > 0:
        parts.append(f"${node.cost_usd:.2f}")

    prefix = "  " * indent
    lines.append(f"{prefix}{' '.join(parts)}")

    for child in tree_state.get_children(node.id):
        _render_node(child, tree_state, lines, indent + 1)


# ---------------------------------------------------------------------------
# Mermaid diagram
# ---------------------------------------------------------------------------

_MERMAID_STYLE = {
    NodeStatus.PENDING: ":::pending",
    NodeStatus.RUNNING: ":::running",
    NodeStatus.SUCCEEDED: ":::succeeded",
    NodeStatus.FAILED: ":::failed",
    NodeStatus.PRUNED: ":::pruned",
}


def mermaid_diagram(tree_state: TreeSearchState) -> str:
    """Render the search tree as a Mermaid flowchart.

    Can be embedded in Markdown or rendered via mermaid-cli.

    Example output::

        ```mermaid
        graph TD
          root["root<br/>ROOT"]:::running
          L4_A["L4_strategy_A<br/>proof_strategy<br/>score=0.82"]:::succeeded
          root --> L4_A
        ```
    """
    lines = [
        "```mermaid",
        "graph TD",
        "  classDef pending fill:#f9f9f9,stroke:#999",
        "  classDef running fill:#fff3cd,stroke:#ffc107",
        "  classDef succeeded fill:#d4edda,stroke:#28a745",
        "  classDef failed fill:#f8d7da,stroke:#dc3545",
        "  classDef pruned fill:#e2e3e5,stroke:#6c757d",
    ]

    for node in tree_state.nodes.values():
        label_parts = [node.id, node.node_type.value]
        if node.score > 0:
            label_parts.append(f"score={node.score:.2f}")
        if node.claim_id:
            label_parts.append(f"claim={node.claim_id}")
        label = "<br/>".join(label_parts)
        style = _MERMAID_STYLE.get(node.status, "")
        # Sanitize node ID for Mermaid (replace dashes/dots with underscores)
        safe_id = node.id.replace("-", "_").replace(".", "_")
        lines.append(f'  {safe_id}["{label}"]{style}')

    for node in tree_state.nodes.values():
        if node.parent_id and node.parent_id in tree_state.nodes:
            safe_parent = node.parent_id.replace("-", "_").replace(".", "_")
            safe_child = node.id.replace("-", "_").replace(".", "_")
            lines.append(f"  {safe_parent} --> {safe_child}")

    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def summary_table(tree_state: TreeSearchState) -> str:
    """Render a compact markdown summary table of all nodes."""
    lines = [
        "| Node ID | Type | Status | Score | Claim | Cost |",
        "|---------|------|--------|-------|-------|------|",
    ]
    for node in tree_state.nodes.values():
        lines.append(
            f"| {node.id} | {node.node_type.value} | {node.status.value} "
            f"| {node.score:.2f} | {node.claim_id or '-'} | ${node.cost_usd:.2f} |"
        )

    stats = tree_state.summary()
    lines.append("")
    lines.append(
        f"**Total:** {stats['total_nodes']} nodes, "
        f"{stats['completed_claims']} claims resolved, "
        f"${stats['total_cost_usd']:.2f} spent"
    )
    return "\n".join(lines)
