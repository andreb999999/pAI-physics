"""
Heuristic content-quality validation for strict editorial runs.

This module is intentionally lightweight and deterministic: it blocks obvious
false-pass artifacts such as unresolved placeholders, TODO markers, and
unresolved citation/reference tokens in the assembled paper sources.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Set


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


def _load_with_inputs(path: str, visited: Optional[Set[str]] = None, depth: int = 0) -> str:
    if visited is None:
        visited = set()
    if depth > 8:
        return ""

    abs_path = os.path.abspath(path)
    if abs_path in visited or not os.path.exists(abs_path):
        return ""
    visited.add(abs_path)

    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return ""

    expanded = [content]
    base_dir = os.path.dirname(abs_path)
    # Follow both \input{} and \include{} directives
    for token in re.findall(r"\\(?:input|include)\{([^}]+)\}", content):
        rel_path = token if token.endswith(".tex") else f"{token}.tex"
        expanded.append(_load_with_inputs(os.path.join(base_dir, rel_path), visited, depth + 1))
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


def validate_paper_quality(workspace_dir: str) -> Dict[str, object]:
    """
    Validate strict editorial hygiene in paper sources.

    Returns:
      {
        "paper_found": bool,
        "is_valid": bool,
        "errors": [ ... ],
        "warnings": [ ... ],
        "source_tex_path": "...",
      }
    """
    tex_path = _resolve_final_tex(workspace_dir)
    if not tex_path:
        return {
            "paper_found": False,
            "is_valid": False,
            "errors": ["missing final_paper.tex (root or paper_workspace/)"],
            "warnings": [],
            "source_tex_path": None,
        }

    combined_content = _load_with_inputs(tex_path)
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

    # Soft warning if no visual evidence commands found in source.
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

