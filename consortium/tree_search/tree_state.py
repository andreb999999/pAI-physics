"""Core data structures for agentic tree search.

Defines the tree node representation, search configuration, and overall tree
state.  Designed to be serialisable to JSON for persistence and resumability.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    """Kind of exploration each tree node represents."""

    ROOT = "root"

    # Theory / proof exploration
    PROOF_STRATEGY = "proof_strategy"
    LEMMA_DECOMPOSITION = "lemma_decomposition"
    ASSUMPTION_VARIANT = "assumption_variant"
    DEBUGGING = "debugging"

    # Verification
    INDEPENDENT_VERIFICATION = "independent_verification"
    COMPONENT_ABLATION = "component_ablation"

    # Experiment exploration
    EXPERIMENT_DESIGN = "experiment_design"
    HYPERPARAMETER_VARIANT = "hyperparameter_variant"
    ABLATION_STUDY = "ablation_study"


class NodeStatus(str, Enum):
    """Lifecycle status of a tree node."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PRUNED = "pruned"


class CounselMode(str, Enum):
    """When to enable multi-model debate within tree branches."""

    ALL_NODES = "all_nodes"
    FINAL_ONLY = "final_only"
    BY_DEPTH = "by_depth"
    BY_NODE_TYPE = "by_node_type"


# ---------------------------------------------------------------------------
# TreeNode
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """A single node in the search tree.

    Each node represents one exploration branch — e.g. a proof strategy, an
    alternative idea, or an experiment design variant.
    """

    id: str
    node_type: NodeType
    status: NodeStatus = NodeStatus.PENDING
    parent_id: Optional[str] = None
    claim_id: Optional[str] = None
    strategy_description: str = ""
    score: float = 0.0
    cost_usd: float = 0.0
    budget_cap_usd: float = 0.0
    workspace_path: str = ""
    depth: int = 0
    children: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: Optional[str] = None

    # -- helpers -------------------------------------------------------------

    def mark_running(self) -> None:
        self.status = NodeStatus.RUNNING

    def mark_succeeded(self) -> None:
        self.status = NodeStatus.SUCCEEDED
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def mark_failed(self) -> None:
        self.status = NodeStatus.FAILED
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def mark_pruned(self) -> None:
        self.status = NodeStatus.PRUNED
        self.completed_at = datetime.now(timezone.utc).isoformat()

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            NodeStatus.SUCCEEDED,
            NodeStatus.FAILED,
            NodeStatus.PRUNED,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "status": self.status.value,
            "parent_id": self.parent_id,
            "claim_id": self.claim_id,
            "strategy_description": self.strategy_description,
            "score": self.score,
            "cost_usd": self.cost_usd,
            "budget_cap_usd": self.budget_cap_usd,
            "workspace_path": self.workspace_path,
            "depth": self.depth,
            "children": self.children,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TreeNode":
        return cls(
            id=data["id"],
            node_type=NodeType(data["node_type"]),
            status=NodeStatus(data["status"]),
            parent_id=data.get("parent_id"),
            claim_id=data.get("claim_id"),
            strategy_description=data.get("strategy_description", ""),
            score=data.get("score", 0.0),
            cost_usd=data.get("cost_usd", 0.0),
            budget_cap_usd=data.get("budget_cap_usd", 0.0),
            workspace_path=data.get("workspace_path", ""),
            depth=data.get("depth", 0),
            children=data.get("children", []),
            metadata=data.get("metadata", {}),
            created_at=data.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
            completed_at=data.get("completed_at"),
        )


# ---------------------------------------------------------------------------
# TreeSearchConfig
# ---------------------------------------------------------------------------

@dataclass
class TreeSearchConfig:
    """User-facing configuration for tree search behaviour."""

    enabled: bool = False
    max_breadth: int = 3
    max_depth: int = 4
    max_parallel: int = 6
    pruning_threshold: float = 0.2
    debug_probability: float = 0.7
    budget_fraction: float = 0.6

    # Counsel integration
    counsel_mode: CounselMode = CounselMode.ALL_NODES
    counsel_at_depth: list[int] = field(default_factory=lambda: [0, 1])
    counsel_node_types: list[str] = field(
        default_factory=lambda: [
            NodeType.PROOF_STRATEGY.value,
            NodeType.LEMMA_DECOMPOSITION.value,
            NodeType.EXPERIMENT_DESIGN.value,
        ]
    )

    def should_counsel(self, node: TreeNode) -> bool:
        """Decide whether to run multi-model counsel for *node*."""
        if self.counsel_mode == CounselMode.ALL_NODES:
            return True
        if self.counsel_mode == CounselMode.FINAL_ONLY:
            return False  # caller enables counsel only on winning branch
        if self.counsel_mode == CounselMode.BY_DEPTH:
            return node.depth in self.counsel_at_depth
        if self.counsel_mode == CounselMode.BY_NODE_TYPE:
            return node.node_type.value in self.counsel_node_types
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_breadth": self.max_breadth,
            "max_depth": self.max_depth,
            "max_parallel": self.max_parallel,
            "pruning_threshold": self.pruning_threshold,
            "debug_probability": self.debug_probability,
            "budget_fraction": self.budget_fraction,
            "counsel_mode": self.counsel_mode.value,
            "counsel_at_depth": self.counsel_at_depth,
            "counsel_node_types": self.counsel_node_types,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TreeSearchConfig":
        return cls(
            enabled=data.get("enabled", False),
            max_breadth=data.get("max_breadth", 3),
            max_depth=data.get("max_depth", 4),
            max_parallel=data.get("max_parallel", 6),
            pruning_threshold=data.get("pruning_threshold", 0.2),
            debug_probability=data.get("debug_probability", 0.7),
            budget_fraction=data.get("budget_fraction", 0.6),
            counsel_mode=CounselMode(
                data.get("counsel_mode", CounselMode.ALL_NODES.value)
            ),
            counsel_at_depth=data.get("counsel_at_depth", [0, 1]),
            counsel_node_types=data.get(
                "counsel_node_types",
                [
                    NodeType.PROOF_STRATEGY.value,
                    NodeType.LEMMA_DECOMPOSITION.value,
                ],
            ),
        )


# ---------------------------------------------------------------------------
# TreeSearchState — full tree
# ---------------------------------------------------------------------------

@dataclass
class TreeSearchState:
    """Persistent state of an entire search tree.

    Stored on disk as ``tree_search_state.json`` inside the run workspace.
    """

    version: int = 1
    config: TreeSearchConfig = field(default_factory=TreeSearchConfig)
    nodes: dict[str, TreeNode] = field(default_factory=dict)
    completed_claims: list[str] = field(default_factory=list)
    total_cost_usd: float = 0.0
    budget_allocations: dict[str, float] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # -- Tree operations -----------------------------------------------------

    def add_node(self, node: TreeNode) -> None:
        """Insert *node* and register it as a child of its parent."""
        self.nodes[node.id] = node
        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.id not in parent.children:
                parent.children.append(node.id)
        self._touch()

    def get_node(self, node_id: str) -> Optional[TreeNode]:
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> list[TreeNode]:
        node = self.nodes.get(node_id)
        if node is None:
            return []
        return [self.nodes[cid] for cid in node.children if cid in self.nodes]

    def get_root(self) -> Optional[TreeNode]:
        for node in self.nodes.values():
            if node.parent_id is None:
                return node
        return None

    # -- Querying ------------------------------------------------------------

    def get_pending_nodes(self) -> list[TreeNode]:
        """All nodes that have not yet been executed or pruned."""
        return [n for n in self.nodes.values() if n.status == NodeStatus.PENDING]

    def get_frontier(self) -> list[TreeNode]:
        """Pending nodes sorted by score (descending)."""
        return sorted(self.get_pending_nodes(), key=lambda n: n.score, reverse=True)

    def get_top_k(self, k: int) -> list[TreeNode]:
        """Return the *k* highest-scoring pending nodes."""
        return self.get_frontier()[:k]

    def get_failed_nodes(self) -> list[TreeNode]:
        return [n for n in self.nodes.values() if n.status == NodeStatus.FAILED]

    def get_succeeded_nodes(self) -> list[TreeNode]:
        return [n for n in self.nodes.values() if n.status == NodeStatus.SUCCEEDED]

    # -- Pruning -------------------------------------------------------------

    def prune_below_threshold(self, threshold: float) -> list[str]:
        """Prune all pending nodes scoring below *threshold*.

        Returns list of pruned node IDs.
        """
        pruned: list[str] = []
        for node in list(self.nodes.values()):
            if node.status == NodeStatus.PENDING and node.score < threshold:
                node.mark_pruned()
                pruned.append(node.id)
        if pruned:
            self._touch()
        return pruned

    def prune_subtree(self, node_id: str) -> list[str]:
        """Recursively prune *node_id* and all its descendants."""
        pruned: list[str] = []
        stack = [node_id]
        while stack:
            nid = stack.pop()
            node = self.nodes.get(nid)
            if node is None:
                continue
            if not node.is_terminal:
                node.mark_pruned()
                pruned.append(nid)
            stack.extend(node.children)
        if pruned:
            self._touch()
        return pruned

    # -- Cost tracking -------------------------------------------------------

    def record_cost(self, node_id: str, cost_usd: float) -> None:
        node = self.nodes.get(node_id)
        if node:
            node.cost_usd += cost_usd
        self.total_cost_usd += cost_usd
        self._touch()

    # -- Claim management ----------------------------------------------------

    def mark_claim_resolved(self, claim_id: str) -> None:
        if claim_id not in self.completed_claims:
            self.completed_claims.append(claim_id)
            self._touch()

    def is_claim_resolved(self, claim_id: str) -> bool:
        return claim_id in self.completed_claims

    # -- Serialisation -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "config": self.config.to_dict(),
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "completed_claims": self.completed_claims,
            "total_cost_usd": self.total_cost_usd,
            "budget_allocations": self.budget_allocations,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TreeSearchState":
        nodes = {
            nid: TreeNode.from_dict(nd)
            for nid, nd in data.get("nodes", {}).items()
        }
        return cls(
            version=data.get("version", 1),
            config=TreeSearchConfig.from_dict(data.get("config", {})),
            nodes=nodes,
            completed_claims=data.get("completed_claims", []),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            budget_allocations=data.get("budget_allocations", {}),
            created_at=data.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
            updated_at=data.get(
                "updated_at", datetime.now(timezone.utc).isoformat()
            ),
        )

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def make_node_id(prefix: str = "node") -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    # -- Summary / display ---------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Compact statistics dict for logging / run_summary.json."""
        by_status: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for n in self.nodes.values():
            by_status[n.status.value] = by_status.get(n.status.value, 0) + 1
            by_type[n.node_type.value] = by_type.get(n.node_type.value, 0) + 1
        return {
            "total_nodes": len(self.nodes),
            "by_status": by_status,
            "by_type": by_type,
            "completed_claims": len(self.completed_claims),
            "total_cost_usd": round(self.total_cost_usd, 4),
        }
