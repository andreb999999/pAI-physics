"""Canonical paper artifact contract helpers.

This module defines the deterministic paper shape expected by strict iterate and
editorial runs. The same contract is consumed by the graph gates, final quality
validation, and campaign artifact validators.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping, Optional

PAPER_WORKSPACE = "paper_workspace"
PAPER_CONTRACT_PATH = "paper_workspace/paper_contract.json"
FINAL_PAPER_TEX = "paper_workspace/final_paper.tex"
FINAL_PAPER_PDF = "paper_workspace/final_paper.pdf"
COPYEDIT_REPORT_TEX = "paper_workspace/copyedit_report.tex"
COPYEDIT_REPORT_PDF = "paper_workspace/copyedit_report.pdf"
REVIEW_REPORT_TEX = "paper_workspace/review_report.tex"
REVIEW_REPORT_PDF = "paper_workspace/review_report.pdf"
REVIEW_VERDICT_JSON = "paper_workspace/review_verdict.json"
PERSONA_VERDICTS_JSON = "paper_workspace/persona_verdicts.json"

CANONICAL_SECTION_FILES = [
    "abstract.tex",
    "introduction.tex",
    "methods.tex",
    "results.tex",
    "discussion.tex",
    "conclusion.tex",
]

DEFAULT_REQUIRED_OUTPUTS = [
    FINAL_PAPER_TEX,
    FINAL_PAPER_PDF,
    REVIEW_VERDICT_JSON,
    COPYEDIT_REPORT_TEX,
    COPYEDIT_REPORT_PDF,
    PERSONA_VERDICTS_JSON,
]

_MUON_REQUIRED_TERMS = [
    "Muon",
    "H1",
    "H2",
    "H3",
    "ATSR",
    "1.83",
    "B_crit",
    "S(mu)|S(μ)",
    "language model",
]

_MUON_STORY_BEATS = [
    "Restore the H1 -> H2 -> H3 storyline and tie it to ATSR.",
    "Explain the role of the 1.83 critical exponent in the Muon narrative.",
    "Discuss B_crit together with the entropy/objective term S(mu) or S(μ).",
    "State the language-model payoff and why the Muon framing matters downstream.",
]


def paper_workspace_path(workspace_dir: str, *parts: str) -> str:
    return os.path.join(workspace_dir, PAPER_WORKSPACE, *parts)


def canonical_section_paths() -> list[str]:
    return [f"{PAPER_WORKSPACE}/{name}" for name in CANONICAL_SECTION_FILES]


def required_writeup_outputs(require_pdf: bool = False) -> list[str]:
    outputs = [PAPER_CONTRACT_PATH, *canonical_section_paths(), FINAL_PAPER_TEX]
    if require_pdf:
        outputs.append(FINAL_PAPER_PDF)
    return outputs


def required_editorial_outputs(require_pdf: bool = True) -> list[str]:
    outputs = list(DEFAULT_REQUIRED_OUTPUTS)
    if not require_pdf and FINAL_PAPER_PDF in outputs:
        outputs.remove(FINAL_PAPER_PDF)
    return outputs


def _safe_jsonish(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        return str(value)


def _is_muon_contract_context(state: Mapping[str, Any]) -> bool:
    haystack = "\n".join(
        [
            _safe_jsonish(state.get("task")),
            _safe_jsonish(state.get("iterate_feedback_summary")),
            _safe_jsonish(state.get("iterate_binding_constraints")),
            _safe_jsonish(state.get("research_proposal")),
            _safe_jsonish(state.get("research_goals")),
        ]
    ).lower()
    trigger_terms = ("muon", "atsr", "b_crit", "h1", "h2", "h3")
    return any(term in haystack for term in trigger_terms)


def build_paper_contract_payload(state: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "required_sections": list(CANONICAL_SECTION_FILES),
        "required_outputs": list(DEFAULT_REQUIRED_OUTPUTS),
        "required_terms": [],
        "required_story_beats": [],
        "task_summary": str(state.get("task") or "")[:2000],
        "iterate_mode": bool(state.get("iterate_mode")),
    }

    if _is_muon_contract_context(state):
        payload["required_terms"] = list(_MUON_REQUIRED_TERMS)
        payload["required_story_beats"] = list(_MUON_STORY_BEATS)

    return payload


def write_paper_contract(workspace_dir: str, state: Mapping[str, Any]) -> str:
    os.makedirs(paper_workspace_path(workspace_dir), exist_ok=True)
    payload = build_paper_contract_payload(state)
    path = os.path.join(workspace_dir, PAPER_CONTRACT_PATH)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return path


def load_paper_contract(workspace_dir: str) -> Optional[dict[str, Any]]:
    path = os.path.join(workspace_dir, PAPER_CONTRACT_PATH)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def missing_writeup_artifacts(workspace_dir: str, require_pdf: bool = False) -> list[str]:
    missing: list[str] = []
    for rel_path in required_writeup_outputs(require_pdf=require_pdf):
        if not os.path.exists(os.path.join(workspace_dir, rel_path)):
            missing.append(rel_path)
    return missing


def missing_editorial_artifacts(workspace_dir: str, require_pdf: bool = True) -> list[str]:
    missing: list[str] = []
    for rel_path in required_editorial_outputs(require_pdf=require_pdf):
        if not os.path.exists(os.path.join(workspace_dir, rel_path)):
            missing.append(rel_path)
    return missing


def build_term_patterns(required_terms: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for raw_term in required_terms:
        if not raw_term:
            continue
        pieces = [piece.strip() for piece in raw_term.split("|") if piece.strip()]
        escaped = [re.escape(piece) for piece in pieces]
        pattern = re.compile(r"(?:" + "|".join(escaped) + r")", re.IGNORECASE)
        patterns.append((raw_term, pattern))
    return patterns


def validate_required_terms(content: str, contract: Optional[Mapping[str, Any]]) -> list[str]:
    if not contract:
        return []
    required_terms = contract.get("required_terms") or []
    if not required_terms:
        return []

    missing: list[str] = []
    for raw_term, pattern in build_term_patterns(list(required_terms)):
        if not pattern.search(content):
            missing.append(raw_term)
    return missing
