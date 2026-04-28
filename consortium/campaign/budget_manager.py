"""Campaign-level budget tracking and graceful degradation."""

from __future__ import annotations

import json
import os
from typing import Optional


# ---------------------------------------------------------------------------
# Degradation profiles
# ---------------------------------------------------------------------------

TIER_TO_INITIAL_RIGOR: dict[str, str] = {
    "budget": "minimal",
    "light": "reduced",
    "medium": "standard",
    "pro": "high",
    "max": "maximum",
}

DEGRADATION_PROFILES: dict[str, dict] = {
    "maximum": {
        "counsel_models": 4,
        "debate_rounds": 3,
        "tree_breadth": 3,
        "compute_tier": "max",
        "adversarial": True,
        "milestone_pdfs": True,
    },
    "high": {
        "counsel_models": 4,
        "debate_rounds": 2,
        "tree_breadth": 2,
        "compute_tier": "high",
        "adversarial": True,
        "milestone_pdfs": True,
    },
    "standard": {
        "counsel_models": 2,
        "debate_rounds": 2,
        "tree_breadth": 2,
        "compute_tier": "standard",
        "adversarial": True,
        "milestone_pdfs": True,
    },
    "reduced": {
        "counsel_models": 1,
        "debate_rounds": 0,
        "tree_breadth": 1,
        "compute_tier": "standard",
        "adversarial": False,
        "milestone_pdfs": True,
    },
    "minimal": {
        "counsel_models": 0,
        "debate_rounds": 0,
        "tree_breadth": 1,
        "compute_tier": "low",
        "adversarial": False,
        "milestone_pdfs": False,
    },
}


class CampaignBudgetManager:
    """Track and enforce budget across all stages of a campaign."""

    def __init__(self, campaign_dir: str, usd_limit: float = 2000.0):
        self.campaign_dir = campaign_dir
        self.usd_limit = usd_limit
        self.state_path = os.path.join(campaign_dir, "campaign_budget.json")
        self._degrade_threshold = 0.50
        self._warning_threshold = 0.70

    # ------------------------------------------------------------------
    # Cost tracking helpers
    # ------------------------------------------------------------------

    def _read_status_workspaces(self) -> list[str]:
        status_path = os.path.join(self.campaign_dir, "campaign_status.json")
        if not os.path.exists(status_path):
            return []
        try:
            with open(status_path) as fh:
                payload = json.load(fh)
        except Exception:
            return []

        seen: set[str] = set()
        workspaces: list[str] = []
        for stage in (payload.get("stages") or {}).values():
            workspace = stage.get("workspace")
            if not workspace:
                continue
            full = workspace if os.path.isabs(workspace) else os.path.join(self.campaign_dir, workspace)
            full = os.path.abspath(full)
            if full in seen or not os.path.isdir(full):
                continue
            seen.add(full)
            workspaces.append(full)
        return sorted(workspaces)

    def _archived_stage_dirs(self) -> list[str]:
        roots = [
            os.path.join(self.campaign_dir, "archive"),
            os.path.join(self.campaign_dir, "archived_attempts"),
        ]
        dirs: list[str] = []
        seen: set[str] = set()
        for root in roots:
            if not os.path.isdir(root):
                continue
            for dirpath, _, files in os.walk(root):
                if "budget_state.json" not in files:
                    continue
                abs_dir = os.path.abspath(dirpath)
                if abs_dir in seen:
                    continue
                seen.add(abs_dir)
                dirs.append(abs_dir)
        return sorted(dirs)

    @staticmethod
    def _dir_cost(stage_dir: str) -> float:
        bs = os.path.join(stage_dir, "budget_state.json")
        if not os.path.exists(bs):
            return 0.0
        try:
            with open(bs) as fh:
                data = json.load(fh)
            return float(data.get("total_usd", data.get("total_cost_usd", 0.0)))
        except Exception:
            return 0.0

    @property
    def current_attempt_spent(self) -> float:
        return sum(self._dir_cost(stage_dir) for stage_dir in self._read_status_workspaces())

    @property
    def archived_spent(self) -> float:
        return sum(self._dir_cost(stage_dir) for stage_dir in self._archived_stage_dirs())

    @property
    def campaign_lifetime_spent(self) -> float:
        return self.current_attempt_spent + self.archived_spent

    @property
    def total_spent(self) -> float:
        return self.current_attempt_spent

    @property
    def remaining(self) -> float:
        return max(0.0, self.usd_limit - self.current_attempt_spent)

    @property
    def spend_fraction(self) -> float:
        if self.usd_limit <= 0:
            return 1.0
        return min(1.0, self.current_attempt_spent / self.usd_limit)

    # ------------------------------------------------------------------
    # Rigor recommendation
    # ------------------------------------------------------------------

    def recommended_rigor_level(self) -> str:
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
        total = self.current_attempt_spent
        frac = self.spend_fraction
        rigor = self.recommended_rigor_level()
        return (
            f"Budget: ${total:.2f} / ${self.usd_limit:.2f} ({frac * 100:.0f}%)"
            f" — Rigor level: {rigor.upper()}"
        )

    # ------------------------------------------------------------------
    # Hard-cap enforcement and threshold alerts
    # ------------------------------------------------------------------

    def is_budget_exceeded(self) -> bool:
        if self.usd_limit <= 0:
            return False
        return self.current_attempt_spent >= self.usd_limit

    def check_thresholds(self) -> list[str]:
        if self.usd_limit <= 0:
            return []
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
        if total_stages_remaining <= 0:
            return self.remaining
        return self.remaining / total_stages_remaining

    # ------------------------------------------------------------------
    # Per-stage cost breakdown
    # ------------------------------------------------------------------

    def per_stage_costs(self) -> list[dict]:
        entries = []
        for stage_dir in self._read_status_workspaces():
            name = os.path.basename(stage_dir)
            cost = self._dir_cost(stage_dir)
            status = "unknown"
            st = os.path.join(stage_dir, "STATUS.txt")
            if os.path.exists(st):
                try:
                    with open(st) as fh:
                        status = fh.read().strip()[:20]
                except Exception:
                    pass
            entries.append({"stage": name, "cost_usd": round(cost, 2), "status": status})
        return entries

    def archived_stage_costs(self) -> list[dict]:
        entries = []
        for stage_dir in self._archived_stage_dirs():
            entries.append({
                "stage": os.path.basename(stage_dir),
                "cost_usd": round(self._dir_cost(stage_dir), 2),
                "status": "archived",
            })
        return entries

    # ------------------------------------------------------------------
    # HTTP endpoint summary
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "campaign_limit_usd": self.usd_limit,
            "current_attempt_usd": round(self.current_attempt_spent, 2),
            "archived_attempts_usd": round(self.archived_spent, 2),
            "campaign_lifetime_usd": round(self.campaign_lifetime_spent, 2),
            "remaining_usd": round(self.remaining, 2),
            "spend_fraction": round(self.spend_fraction, 4),
            "current_rigor_level": self.recommended_rigor_level(),
            "per_stage_costs": self.per_stage_costs(),
            "archived_stage_costs": self.archived_stage_costs(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        with open(self.state_path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, campaign_dir: str, usd_limit: float = 2000.0) -> "CampaignBudgetManager":
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
    """Apply a degradation profile to the current run parameters."""
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
