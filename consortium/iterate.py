"""
Iterate mode — feedback ingestion and state seeding for paper revision.

Provides utilities to:
1. Validate an iterate directory (prior paper + feedback files)
2. Parse and structure feedback from .tex/.md files
3. Seed pipeline state for a revision run
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Names that signal "this is the paper, not feedback"
_PAPER_BASENAMES = {"final_paper", "main", "paper", "manuscript", "draft"}

# No artificial truncation — frontier models have 200K-2M token contexts.
# A typical 40-page paper is ~80K chars (~20K tokens), well within limits.
# Set to None to read the entire file; set to a positive int to cap.
_MAX_FEEDBACK_CHARS = None     # None = unlimited
_MAX_PAPER_CHARS = None        # None = unlimited
_MAX_TOTAL_FEEDBACK_CHARS = None  # None = unlimited


def validate_iterate_dir(path: str) -> dict:
    """Scan *path* for a prior paper and feedback files.

    Returns::

        {
            "paper_tex": str | None,   # path to .tex paper (preferred)
            "paper_pdf": str | None,   # path to .pdf paper
            "feedback_files": [str],   # .tex / .md feedback files
        }

    Raises ``ValueError`` when the directory is missing, contains no
    paper, or contains no feedback.
    """
    dirpath = Path(path).resolve()
    if not dirpath.is_dir():
        raise ValueError(
            f"Iterate directory does not exist: {dirpath}"
        )

    tex_files: list[Path] = []
    pdf_files: list[Path] = []
    md_files: list[Path] = []

    for f in dirpath.iterdir():
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext == ".tex":
            tex_files.append(f)
        elif ext == ".pdf":
            pdf_files.append(f)
        elif ext == ".md":
            md_files.append(f)

    # --- identify the paper among .tex files ---
    paper_tex: Optional[Path] = None
    if tex_files:
        # Prefer known paper basenames
        for t in tex_files:
            if t.stem.lower() in _PAPER_BASENAMES:
                paper_tex = t
                break
        # Fallback: largest .tex file
        if paper_tex is None:
            paper_tex = max(tex_files, key=lambda p: p.stat().st_size)

    paper_pdf: Optional[Path] = None
    if pdf_files:
        for p in pdf_files:
            if p.stem.lower() in _PAPER_BASENAMES:
                paper_pdf = p
                break
        if paper_pdf is None:
            paper_pdf = max(pdf_files, key=lambda p: p.stat().st_size)

    if paper_tex is None and paper_pdf is None:
        raise ValueError(
            f"No paper found in iterate directory {dirpath}. "
            "Expected a .tex or .pdf file (e.g. final_paper.tex)."
        )

    # --- feedback files = everything else ---
    feedback: list[Path] = []
    for f in md_files:
        feedback.append(f)
    for f in tex_files:
        if f != paper_tex:
            feedback.append(f)

    if not feedback:
        raise ValueError(
            f"No feedback files found in iterate directory {dirpath}. "
            "Expected .tex or .md files containing reviewer feedback."
        )

    return {
        "paper_tex": str(paper_tex) if paper_tex else None,
        "paper_pdf": str(paper_pdf) if paper_pdf else None,
        "feedback_files": [str(f) for f in sorted(feedback)],
    }


def parse_feedback_files(files: list[str]) -> list[dict]:
    """Read each feedback file and return structured items.

    Returns a list of::

        {"source": str, "content": str, "format": "tex" | "md"}
    """
    items: list[dict] = []
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read(_MAX_FEEDBACK_CHARS)
        except OSError:
            continue

        if not content.strip():
            continue

        fmt = "tex" if fpath.endswith(".tex") else "md"
        items.append({
            "source": os.path.basename(fpath),
            "content": content,
            "format": fmt,
        })
    return items


def extract_paper_content(
    paper_tex: Optional[str],
    paper_pdf: Optional[str],
    workspace_dir: str,
) -> str:
    """Read the prior paper content and copy originals into workspace.

    Returns the paper text (LaTeX source preferred, else PDF-extracted markdown).
    Copies originals to ``{workspace_dir}/paper_workspace/prior_paper/``.
    """
    prior_dir = os.path.join(workspace_dir, "paper_workspace", "prior_paper")
    os.makedirs(prior_dir, exist_ok=True)

    paper_content = ""

    if paper_tex:
        shutil.copy2(paper_tex, prior_dir)
        with open(paper_tex, "r", encoding="utf-8", errors="replace") as fh:
            paper_content = fh.read(_MAX_PAPER_CHARS)

    if paper_pdf:
        shutil.copy2(paper_pdf, prior_dir)
        if not paper_content:
            # No .tex available — extract text from PDF
            paper_content = _extract_pdf_text(paper_pdf)

    return paper_content


def _extract_pdf_text(pdf_path: str) -> str:
    """Best-effort PDF text extraction."""
    try:
        import pdfminer.high_level
        return pdfminer.high_level.extract_text(pdf_path)[:_MAX_PAPER_CHARS]
    except Exception:
        pass
    try:
        from consortium.toolkits.search.text_web_browser.mdconvert import (
            MarkdownConverter,
        )
        converter = MarkdownConverter()
        result = converter.convert_local(pdf_path, file_extension=".pdf")
        if result and result.text_content:
            return result.text_content[:_MAX_PAPER_CHARS]
    except Exception:
        pass
    return "[PDF text extraction failed — agents should use file tools to read the PDF directly]"


def structure_feedback(items: list[dict]) -> str:
    """Concatenate parsed feedback items into a single markdown document."""
    if not items:
        return ""

    parts: list[str] = [
        "# Iteration Feedback\n",
        "The following feedback was provided for the prior version of the paper. "
        "Each section below corresponds to one feedback source file.\n",
    ]

    total_chars = 0
    for item in items:
        if _MAX_TOTAL_FEEDBACK_CHARS is not None and total_chars >= _MAX_TOTAL_FEEDBACK_CHARS:
            parts.append(
                "\n> **Note:** Remaining feedback truncated due to length.\n"
            )
            break
        header = f"\n---\n## Feedback from `{item['source']}` ({item['format']})\n\n"
        content = item["content"]
        if _MAX_TOTAL_FEEDBACK_CHARS is not None:
            remaining = _MAX_TOTAL_FEEDBACK_CHARS - total_chars
            if len(content) > remaining:
                content = content[:remaining] + "\n\n[...truncated...]"
        parts.append(header)
        parts.append(content)
        total_chars += len(content)

    return "\n".join(parts)


def build_iterate_state_seed(iterate_dir: str, workspace_dir: str) -> dict:
    """Orchestrate feedback ingestion and return state overrides.

    1. Validates the iterate directory
    2. Copies the prior paper into the workspace
    3. Parses and consolidates feedback
    4. Writes workspace artifacts
    5. Returns a dict of state field overrides to merge into initial_state
    """
    info = validate_iterate_dir(iterate_dir)

    # Extract paper and copy to workspace
    paper_content = extract_paper_content(
        info["paper_tex"], info["paper_pdf"], workspace_dir
    )

    # Parse feedback
    items = parse_feedback_files(info["feedback_files"])
    consolidated = structure_feedback(items)

    # Write consolidated feedback
    paper_ws = os.path.join(workspace_dir, "paper_workspace")
    os.makedirs(paper_ws, exist_ok=True)

    feedback_path = os.path.join(paper_ws, "iteration_feedback.md")
    with open(feedback_path, "w", encoding="utf-8") as fh:
        fh.write(consolidated)

    # Write manifest
    manifest = {
        "iterate_dir": str(Path(iterate_dir).resolve()),
        "paper_tex": info["paper_tex"],
        "paper_pdf": info["paper_pdf"],
        "feedback_files": info["feedback_files"],
        "feedback_count": len(items),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = os.path.join(paper_ws, "iteration_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # Determine prior paper path inside workspace
    prior_dir = os.path.join(paper_ws, "prior_paper")
    if info["paper_tex"]:
        prior_paper_ws = os.path.join(prior_dir, os.path.basename(info["paper_tex"]))
    elif info["paper_pdf"]:
        prior_paper_ws = os.path.join(prior_dir, os.path.basename(info["paper_pdf"]))
    else:
        prior_paper_ws = ""

    # Build short feedback summary for state
    summary_lines = [f"- {item['source']}" for item in items[:10]]
    feedback_summary = (
        f"Revision based on {len(items)} feedback source(s):\n"
        + "\n".join(summary_lines)
    )

    # Extract binding constraints from human_directive.md if present.
    # These are non-negotiable research decisions set by the PI that persona
    # council members must respect (they may flag concerns but not REJECT
    # based on these constraints).
    binding_constraints = ""
    directive_path = os.path.join(iterate_dir, "human_directive.md")
    if os.path.isfile(directive_path):
        with open(directive_path, "r", encoding="utf-8", errors="replace") as fh:
            binding_constraints = fh.read()

    return {
        "iterate_mode": True,
        "iterate_prior_paper_path": prior_paper_ws,
        "iterate_feedback_path": feedback_path,
        "iterate_feedback_summary": feedback_summary,
        "iterate_binding_constraints": binding_constraints,
        "formalized_results": (
            f"[Prior paper — revision mode]\n\n"
            f"{paper_content}"
        ),
    }
