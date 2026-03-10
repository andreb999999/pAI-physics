"""Budget-aware allocation for tree search branches.

Wraps the existing :class:`BudgetManager` with per-branch budget tracking,
distributing the global budget across tree branches and pruning branches that
exceed their allocation.
"""

from __future__ import annotations

from typing import Any, Optional

from consortium.tree_search.tree_state import TreeNode, TreeSearchConfig, TreeSearchState


class TreeBudgetAllocator:
    """Distribute and track budget across tree search branches.

    Parameters
    ----------
    total_budget_usd : float
        Total budget available for the run (from BudgetManager.usd_limit).
    tree_config : TreeSearchConfig
        Tree search configuration (budget_fraction, max_breadth, etc.).
    """

    def __init__(
        self,
        total_budget_usd: float,
        tree_config: TreeSearchConfig,
    ) -> None:
        self.total_budget_usd = total_budget_usd
        self.tree_fraction = tree_config.budget_fraction
        self.max_breadth = tree_config.max_breadth
        self.tree_budget_usd = total_budget_usd * self.tree_fraction
        self.branch_budgets: dict[str, float] = {}
        self.branch_spent: dict[str, float] = {}

    @property
    def remaining_tree_budget(self) -> float:
        """Budget not yet allocated to any branch."""
        allocated = sum(self.branch_budgets.values())
        return max(0.0, self.tree_budget_usd - allocated)

    def allocate_branch(self, branch_id: str, num_claims: int = 1) -> float:
        """Allocate budget for a new branch.

        Splits the remaining tree budget evenly across expected branches
        (``num_claims × max_breadth``).

        Returns the allocated amount in USD.
        """
        expected_branches = max(num_claims * self.max_breadth, 1)
        allocation = self.remaining_tree_budget / expected_branches
        self.branch_budgets[branch_id] = allocation
        self.branch_spent[branch_id] = 0.0
        return allocation

    def record_spend(self, branch_id: str, cost_usd: float) -> None:
        """Record spend against a branch."""
        self.branch_spent[branch_id] = self.branch_spent.get(branch_id, 0.0) + cost_usd

    def should_prune(self, branch_id: str) -> bool:
        """Check if a branch has exhausted its budget."""
        budget = self.branch_budgets.get(branch_id, 0.0)
        spent = self.branch_spent.get(branch_id, 0.0)
        return spent >= budget and budget > 0

    def reallocate_from_pruned(self, pruned_branch_id: str) -> float:
        """Return unspent budget from a pruned branch to the pool.

        Returns the amount reclaimed.
        """
        budget = self.branch_budgets.pop(pruned_branch_id, 0.0)
        spent = self.branch_spent.pop(pruned_branch_id, 0.0)
        reclaimed = max(0.0, budget - spent)
        # The reclaimed budget goes back into remaining_tree_budget
        # (since it's no longer allocated)
        return reclaimed

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for logging."""
        return {
            "total_budget_usd": round(self.total_budget_usd, 4),
            "tree_budget_usd": round(self.tree_budget_usd, 4),
            "remaining_tree_budget": round(self.remaining_tree_budget, 4),
            "num_branches": len(self.branch_budgets),
            "total_branch_spend": round(sum(self.branch_spent.values()), 4),
            "branches": {
                bid: {
                    "budget": round(self.branch_budgets.get(bid, 0.0), 4),
                    "spent": round(self.branch_spent.get(bid, 0.0), 4),
                }
                for bid in self.branch_budgets
            },
        }
