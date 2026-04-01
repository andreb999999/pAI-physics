"""Research run presets — map user-friendly names to consortium flags."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    output_format: str
    model: str
    budget_usd: int
    enable_counsel: bool
    no_counsel: bool
    enable_math_agents: bool
    enable_tree_search: bool
    adversarial_verification: bool
    enable_planning: bool
    enforce_paper_artifacts: bool
    enforce_editorial_artifacts: bool
    autonomous_mode: bool
    time_estimate: str
    cost_estimate: str


PRESETS: dict[str, Preset] = {
    "quick": Preset(
        name="quick",
        description="Fast markdown output, single model, minimal budget",
        output_format="markdown",
        model="claude-sonnet-4-6",
        budget_usd=5,
        enable_counsel=False,
        no_counsel=True,
        enable_math_agents=False,
        enable_tree_search=False,
        adversarial_verification=False,
        enable_planning=False,
        enforce_paper_artifacts=False,
        enforce_editorial_artifacts=False,
        autonomous_mode=True,
        time_estimate="~30 min",
        cost_estimate="$2-5",
    ),
    "standard": Preset(
        name="standard",
        description="Balanced quality and cost, markdown output",
        output_format="markdown",
        model="claude-sonnet-4-6",
        budget_usd=25,
        enable_counsel=False,
        no_counsel=True,
        enable_math_agents=False,
        enable_tree_search=False,
        adversarial_verification=False,
        enable_planning=True,
        enforce_paper_artifacts=False,
        enforce_editorial_artifacts=False,
        autonomous_mode=True,
        time_estimate="~2 hours",
        cost_estimate="$10-25",
    ),
    "thorough": Preset(
        name="thorough",
        description="High quality with counsel debate, LaTeX output, math agents",
        output_format="latex",
        model="claude-opus-4-6",
        budget_usd=100,
        enable_counsel=True,
        no_counsel=False,
        enable_math_agents=True,
        enable_tree_search=False,
        adversarial_verification=False,
        enable_planning=True,
        enforce_paper_artifacts=True,
        enforce_editorial_artifacts=False,
        autonomous_mode=True,
        time_estimate="~6 hours",
        cost_estimate="$40-100",
    ),
    "maximum": Preset(
        name="maximum",
        description="Maximum quality: counsel + tree search + adversarial verification",
        output_format="latex",
        model="claude-opus-4-6",
        budget_usd=200,
        enable_counsel=True,
        no_counsel=False,
        enable_math_agents=True,
        enable_tree_search=True,
        adversarial_verification=True,
        enable_planning=True,
        enforce_paper_artifacts=True,
        enforce_editorial_artifacts=True,
        autonomous_mode=True,
        time_estimate="~12+ hours",
        cost_estimate="$80-200",
    ),
}


def list_presets() -> list[Preset]:
    """Return all presets in order of increasing quality."""
    return [PRESETS[k] for k in ("quick", "standard", "thorough", "maximum")]
