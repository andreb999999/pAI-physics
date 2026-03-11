"""Cross-branch failure learning for tree search.

When a proof branch fails, the failure details are recorded so that future
strategy generation can avoid repeating the same mistakes.  FailureMemory
is consulted during strategy expansion to bias the search away from
approaches that have already been tried and found wanting.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import ClassVar


@dataclass
class FailureRecord:
    """A single recorded proof-branch failure."""

    claim_id: str
    strategy_name: str
    failure_reason: str
    adversarial_report: str  # From Phase 4; empty string if unavailable.
    verification_gaps: list[dict]
    depth: int
    timestamp: str


class FailureMemory:
    """Accumulates failure records and exposes them for strategy generation.

    Provides keyword-based clustering of failure reasons and formatted
    summaries that can be appended directly to LLM prompts.
    """

    # Common keywords used to cluster failure reasons.
    PATTERN_KEYWORDS: ClassVar[list[str]] = [
        "convergence",
        "bound",
        "assumption",
        "limit exchange",
        "regularity",
    ]

    def __init__(self) -> None:
        self.records: list[FailureRecord] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_failure(self, record: FailureRecord) -> None:
        """Append a failure record."""
        self.records.append(record)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_relevant_failures(self, claim_id: str) -> list[FailureRecord]:
        """Return all failure records for *claim_id*."""
        return [r for r in self.records if r.claim_id == claim_id]

    def get_failure_patterns(self) -> dict[str, int]:
        """Cluster failure reasons by common keywords and return counts.

        Each failure reason is scanned (case-insensitive) for every keyword
        in ``PATTERN_KEYWORDS``.  A keyword's count reflects how many
        distinct failure records mention it.
        """
        counts: dict[str, int] = {}
        for keyword in self.PATTERN_KEYWORDS:
            total = sum(
                1
                for r in self.records
                if keyword.lower() in r.failure_reason.lower()
            )
            if total > 0:
                counts[keyword] = total
        return counts

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_for_strategy_prompt(self, claim_id: str) -> str:
        """Format failures into a string for a strategy-generation prompt.

        Lists every failed strategy for *claim_id* together with its reason,
        then appends aggregate failure-pattern counts across all claims so
        the model can recognise systemic issues.
        """
        relevant = self.get_relevant_failures(claim_id)
        if not relevant and not self.records:
            return ""

        lines: list[str] = []

        if relevant:
            lines.append("## Previously failed strategies for this claim")
            for rec in relevant:
                lines.append(
                    f"- **{rec.strategy_name}** (depth {rec.depth}): "
                    f"{rec.failure_reason}"
                )
                if rec.adversarial_report:
                    lines.append(
                        f"  Adversarial report: {rec.adversarial_report}"
                    )
            lines.append("")

        patterns = self.get_failure_patterns()
        if patterns:
            lines.append("## Common failure patterns (all claims)")
            for kw, count in sorted(patterns.items(), key=lambda x: -x[1]):
                lines.append(f"- {kw}: {count} occurrence(s)")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the failure memory to a JSON file at *path*."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = [asdict(r) for r in self.records]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> FailureMemory:
        """Deserialize a ``FailureMemory`` from a JSON file at *path*."""
        mem = cls()
        if not os.path.exists(path):
            return mem
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for entry in data:
            mem.records.append(FailureRecord(**entry))
        return mem
