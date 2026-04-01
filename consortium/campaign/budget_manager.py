"""Campaign-level budget tracking and graceful degradation.

Maintains a running sum of costs across all stages in a campaign with a hard
cap.  When approaching the limit, automatically reduces spend rate by
downgrading rigor parameters (fewer counsel models, shorter debates, narrower
tree breadth, lower thinking budgets).

Usage::

    mgr = CampaignBudgetManager(campaign_dir, usd_limit=2000)
    rigor = mgr.recommended_rigor_level()
    profile = DEGRADATION_PROFILES[rigor]
    # Apply profile to stage config...
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Degradation profiles
# ---------------------------------------------------------------------------

# Maps price tiers to their starting rigor level.
# Lower tiers start at lower rigor to avoid immediate degradation.
TIER_TO_INITIAL_RIGOR: dict[str, str] = {
    "budget": "minimal",
    "light": "reduced",
    "medium": "standard",
    "pro": "high",
    "max": "maximum",
}

DEGRADATION_PROFILES: dict[str, dict] = {
    "maximum": {
        "counsel_models": 4,         # all 4 frontier models
        "debate_rounds": 3,          # full 3-round debate
        "tree_breadth": 3,           # 3 branches per claim
        "compute_tier": "max",       # max thinking tokens
        "adversarial": True,         # adversarial verification on
        "milestone_pdfs": True,      # generate milestone reports
    },
    "high": {
        "counsel_models": 4,         # keep all models
        "debate_rounds": 2,          # reduce debate rounds
        "tree_breadth": 2,           # 2 branches per claim
        "compute_tier": "high",      # reduce thinking tokens
        "adversarial": True,         # keep adversarial
        "milestone_pdfs": True,
    },
    "standard": {
        "counsel_models": 2,         # only Opus + Sonnet (drop GPT, Gemini)
        "debate_rounds": 2,
        "tree_breadth": 2,
        "compute_tier": "standard",
        "adversarial": True,
        "milestone_pdfs": True,
    },
    "reduced": {
        "counsel_models": 1,         # Opus only (no debate, just single model)
        "debate_rounds": 0,
        "tree_breadth": 1,           # linear (no tree search)
        "compute_tier": "standard",
        "adversarial": False,        # skip adversarial to save cost
        "milestone_pdfs": True,      # keep for interpretability
    },
    "minimal": {
        "counsel_models": 0,         # no counsel
        "debate_rounds": 0,
        "tree_breadth": 1,
        "compute_tier": "low",
        "adversarial": False,
        "milestone_pdfs": False,     # skip PDF generation
    },
}


# ---------------------------------------------------------------------------
# Campaign budget manager
# ---------------------------------------------------------------------------

class CampaignBudgetManager:
    """Track and enforce budget across all stages of a campaign.

    Reads per-stage ``budget_state.json`` files to compute total spend.
    Recommends a rigor level based on remaining budget.  Provides a
    ``GET /budget``-compatible summary dict.
    """

    def __init__(self, campaign_dir: str, usd_limit: float = 2000.0):
        self.campaign_dir = campaign_dir
        self.usd_limit = usd_limit
        self.state_path = os.path.join(campaign_dir, "campaign_budget.json")
        self._degrade_threshold = 0.50  # start degrading at 50% spent
        self._warning_threshold = 0.70  # warn at 70% spent

    # ------------------------------------------------------------------
    # Cost tracking
    # ------------------------------------------------------------------

    def _stage_dirs(self) -> list[str]:
        """Find all stage workspace directories under the campaign dir."""
        dirs = []
        if not os.path.isdir(self.campaign_dir):
            return dirs
        for entry in os.listdir(self.campaign_dir):
            full = os.path.join(self.campaign_dir, entry)
            if os.path.isdir(full):
                dirs.append(full)
        return sorted(dirs)

    @property
    def total_spent(self) -> float:
        """Sum budget_state.json across all stage workspaces."""
        total = 0.0
        for stage_dir in self._stage_dirs():
            bs = os.path.join(stage_dir, "budget_state.json")
            if os.path.exists(bs):
                try:
                    with open(bs) as f:
                        data = json.load(f)
                    total += float(data.get("total_usd", data.get("total_cost_usd", 0.0)))
                except Exception:
                    pass
        return total

    @property
    def remaining(self) -> float:
        return max(0.0, self.usd_limit - self.total_spent)

    @property
    def spend_fraction(self) -> float:
        """0.0 = nothing spent, 1.0 = budget exhausted."""
        if self.usd_limit <= 0:
            return 1.0
        return min(1.0, self.total_spent / self.usd_limit)

    # ------------------------------------------------------------------
    # Rigor recommendation
    # ------------------------------------------------------------------

    def recommended_rigor_level(self) -> str:
        """Return rigor level string based on remaining budget."""
        frac = self.spend_fraction
        if frac < self._degrade_threshold:
            return "maximum"
        if frac < 0.70:
            return "high"
        if frac < 0.85:
            return "standard"
        if frac < 0.95:
            return "reduced"
        return "minimal"

    def budget_status_line(self) -> str:
        """One-line budget summary for milestone report headers."""
        total = self.total_spent
        frac = self.spend_fraction
        rigor = self.recommended_rigor_level()
        return f"Budget: ${total:.2f} / ${self.usd_limit:.2f} ({frac * 100:.0f}%) — Rigor level: {rigor.upper()}"

    # ------------------------------------------------------------------
    # Hard-cap enforcement and threshold alerts
    # ------------------------------------------------------------------

    def is_budget_exceeded(self) -> bool:
        """Return True if total spend has reached or exceeded the budget cap."""
        if self.usd_limit <= 0:
            return False  # unlimited
        return self.total_spent >= self.usd_limit

    def check_thresholds(self) -> list[str]:
        """Return list of budget threshold alerts that should be fired.

        Possible alerts: '85pct_warning', '95pct_critical', 'exceeded'.
        Callers should use sentinel files to avoid re-firing the same alert.
        """
        if self.usd_limit <= 0:
            return []  # unlimited
        alerts = []
        frac = self.spend_fraction
        if frac >= 0.85:
            alerts.append("85pct_warning")
        if frac >= 0.95:
            alerts.append("95pct_critical")
        if frac >= 1.0:
            alerts.append("exceeded")
        return alerts

    # ------------------------------------------------------------------
    # Stage budget allocation
    # ------------------------------------------------------------------

    def allocate_stage(self, stage_name: str, total_stages_remaining: int = 1) -> float:
        """Return USD budget for next stage based on remaining budget."""
        if total_stages_remaining <= 0:
            return self.remaining
        return self.remaining / total_stages_remaining

    # ------------------------------------------------------------------
    # Per-stage cost breakdown
    # ------------------------------------------------------------------

    def per_stage_costs(self) -> list[dict]:
        """Return per-stage cost and status summary."""
        entries = []
        for stage_dir in self._stage_dirs():
            name = os.path.basename(stage_dir)
            cost = 0.0
            status = "unknown"

            bs = os.path.join(stage_dir, "budget_state.json")
            if os.path.exists(bs):
                try:
                    with open(bs) as f:
                        data = json.load(f)
                    cost = float(data.get("total_usd", data.get("total_cost_usd", 0.0)))
                except Exception:
                    pass

            st = os.path.join(stage_dir, "STATUS.txt")
            if os.path.exists(st):
                try:
                    with open(st) as f:
                        status = f.read().strip()[:20]
                except Exception:
                    pass

            entries.append({"stage": name, "cost_usd": round(cost, 2), "status": status})
        return entries

    # ------------------------------------------------------------------
    # HTTP endpoint summary
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a dict suitable for ``GET /budget`` JSON response."""
        return {
            "campaign_limit_usd": self.usd_limit,
            "total_spent_usd": round(self.total_spent, 2),
            "remaining_usd": round(self.remaining, 2),
            "spend_fraction": round(self.spend_fraction, 4),
            "current_rigor_level": self.recommended_rigor_level(),
            "per_stage_costs": self.per_stage_costs(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Write campaign_budget.json."""
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, campaign_dir: str, usd_limit: float = 2000.0) -> "CampaignBudgetManager":
        """Load or create a CampaignBudgetManager."""
        return cls(campaign_dir=campaign_dir, usd_limit=usd_limit)


# ---------------------------------------------------------------------------
# Profile application helper
# ---------------------------------------------------------------------------

def apply_degradation_profile(
    profile: dict,
    *,
    counsel_models: Optional[list] = None,
    tree_search_config: Optional[object] = None,
    max_debate_rounds_env: str = "CONSORTIUM_COUNSEL_MAX_DEBATE_ROUNDS",
) -> dict:
    """Apply a degradation profile to the current run parameters.

    Returns a dict of overrides that were applied (for logging).
    """
    overrides = {}

    if counsel_models is not None:
        target_count = profile.get("counsel_models", len(counsel_models))
        if target_count < len(counsel_models):
            del counsel_models[target_count:]
            overrides["counsel_models"] = target_count

    debate_rounds = profile.get("debate_rounds")
    if debate_rounds is not None:
        os.environ[max_debate_rounds_env] = str(debate_rounds)
        overrides["debate_rounds"] = debate_rounds

    if tree_search_config is not None:
        breadth = profile.get("tree_breadth")
        if breadth is not None and hasattr(tree_search_config, "max_breadth"):
            tree_search_config.max_breadth = min(tree_search_config.max_breadth, breadth)
            overrides["tree_breadth"] = tree_search_config.max_breadth

    return overrides
