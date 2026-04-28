"""Heuristic content-quality validation for strict editorial runs.

This module is intentionally lightweight and deterministic: it blocks obvious
false-pass artifacts such as unresolved placeholders, stub skeletons, missing
section inputs, and contract-term omissions in the assembled paper sources.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Set

from ..paper_contract import FINAL_PAPER_PDF, load_paper_contract, validate_required_terms


_PLACEHOLDER_PATTERNS = [
    re.compile(r"\[DATA REQUIRED:[^\]]*\]", re.IGNORECASE),
    re.compile(r"\bTODO\b", re.IGNORECASE),
    re.compile(r"\bTBD\b", re.IGNORECASE),
    re.compile(r"\[fill[^\]]*\]", re.IGNORECASE),
    re.compile(r"\[cite:[^\]]+\]", re.IGNORECASE),
    re.compile(r"\\title\{Research Paper Title\}", re.IGNORECASE),
    re.compile(r"\\author\{Author Names\}", re.IGNORECASE),
]

_UNRESOLVED_REFERENCE_PATTERNS = [
    re.compile(r"(?:Table|Fig(?:ure)?|Section|Eq(?:uation)?\.?)\s*\?\?", re.IGNORECASE),
    re.compile(r"\\ref\{\?\}"),
    re.compile(r"\\cite\{\?\}"),
]

_SKELETON_ONLY_PATTERNS = [
    re.compile(r"\\input\{abstract\}"),
    re.compile(r"\\input\{introduction\}"),
    re.compile(r"\\input\{methods\}"),
    re.compile(r"\\input\{results\}"),
    re.compile(r"\\input\{discussion\}"),
    re.compile(r"\\input\{conclusion\}"),
]


def _candidate_tex_paths(workspace_dir: str) -> List[str]:
    return [
        os.path.join(workspace_dir, "final_paper.tex"),
        os.path.join(workspace_dir, "paper_workspace", "final_paper.tex"),
    ]


def _resolve_final_tex(workspace_dir: str) -> Optional[str]:
    for path in _candidate_tex_paths(workspace_dir):
        if os.path.exists(path):
            return path
    return None


def _load_with_inputs(
    path: str,
    visited: Optional[Set[str]] = None,
    depth: int = 0,
    missing_inputs: Optional[List[str]] = None,
) -> str:
    if visited is None:
        visited = set()
    if missing_inputs is None:
        missing_inputs = []
    if depth > 8:
        return ""

    abs_path = os.path.abspath(path)
    if abs_path in visited:
        return ""
    if not os.path.exists(abs_path):
        missing_inputs.append(abs_path)
        return ""
    visited.add(abs_path)

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return ""

    expanded = [content]
    base_dir = os.path.dirname(abs_path)
    for token in re.findall(r"\\(?:input|include)\{([^}]+)\}", content):
        rel_path = token if token.endswith(".tex") else f"{token}.tex"
        child = os.path.join(base_dir, rel_path)
        if not os.path.exists(child):
            missing_inputs.append(os.path.relpath(child, base_dir))
            continue
        expanded.append(_load_with_inputs(child, visited, depth + 1, missing_inputs))
    return "\n".join(expanded)


def _match_examples(content: str, pattern: re.Pattern, limit: int = 5) -> List[str]:
    hits = []
    for m in pattern.finditer(content):
        snippet = m.group(0).strip().replace("\n", " ")
        if snippet:
            hits.append(snippet[:200])
        if len(hits) >= limit:
            break
    return hits


def _looks_like_generic_skeleton(main_tex: str) -> bool:
    return all(pattern.search(main_tex) for pattern in _SKELETON_ONLY_PATTERNS)


def validate_paper_quality(
    workspace_dir: str,
    require_pdf: bool = False,
    enforce_contract: bool = False,
) -> Dict[str, object]:
    """Validate strict editorial hygiene in paper sources."""
    tex_path = _resolve_final_tex(workspace_dir)
    if not tex_path:
        return {
            "paper_found": False,
            "is_valid": False,
            "errors": ["missing final_paper.tex (root or paper_workspace/)"] ,
            "warnings": [],
            "source_tex_path": None,
        }

    try:
        with open(tex_path, "r", encoding="utf-8") as fh:
            main_tex = fh.read()
    except Exception:
        main_tex = ""

    missing_inputs: List[str] = []
    combined_content = _load_with_inputs(tex_path, missing_inputs=missing_inputs)
    if not combined_content.strip():
        return {
            "paper_found": True,
            "is_valid": False,
            "errors": ["final paper source is empty or unreadable"],
            "warnings": [],
            "source_tex_path": tex_path,
        }

    errors: List[str] = []
    warnings: List[str] = []

    if missing_inputs:
        errors.append(
            "missing \\input/\\include sources: " + ", ".join(sorted(set(missing_inputs)))
        )

    for pattern in _PLACEHOLDER_PATTERNS:
        examples = _match_examples(combined_content, pattern)
        if examples:
            errors.append(
                f"unresolved placeholder pattern '{pattern.pattern}' found (examples: {examples})"
            )

    for pattern in _UNRESOLVED_REFERENCE_PATTERNS:
        examples = _match_examples(combined_content, pattern)
        if examples:
            errors.append(
                f"unresolved reference/citation pattern '{pattern.pattern}' found (examples: {examples})"
            )

    if _looks_like_generic_skeleton(main_tex):
        errors.append(
            "final_paper.tex is still a generic \\input skeleton; canonical section content is missing or incomplete"
        )

    contract = load_paper_contract(workspace_dir)
    if enforce_contract and contract:
        missing_terms = validate_required_terms(combined_content, contract)
        if missing_terms:
            errors.append(
                "paper contract required terms missing from assembled paper: "
                + ", ".join(missing_terms)
            )

    if require_pdf:
        pdf_path = os.path.join(workspace_dir, FINAL_PAPER_PDF)
        if not os.path.exists(pdf_path):
            errors.append("missing paper_workspace/final_paper.pdf")

    has_table = "\\begin{table" in combined_content
    has_figure = "\\begin{figure" in combined_content or "\\includegraphics" in combined_content
    if not (has_table or has_figure):
        warnings.append("no figure/table environments found in paper source")

    return {
        "paper_found": True,
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "source_tex_path": tex_path,
    }
