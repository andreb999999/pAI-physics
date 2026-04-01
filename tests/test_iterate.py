"""
Tests for consortium/iterate.py — iterate mode feedback ingestion.
"""

import json
import os
import tempfile

import pytest

from consortium.iterate import (
    validate_iterate_dir,
    parse_feedback_files,
    extract_paper_content,
    structure_feedback,
    build_iterate_state_seed,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def iterate_dir(tmp_path):
    """Create a minimal iterate directory with a paper and feedback."""
    paper = tmp_path / "final_paper.tex"
    paper.write_text(r"\documentclass{article}" + "\n" + r"\begin{document}" + "\n"
                     r"Hello world." + "\n" + r"\end{document}" + "\n")
    fb1 = tmp_path / "review_feedback.md"
    fb1.write_text("# Reviewer 1\n\nThe introduction is too vague.\n")
    fb2 = tmp_path / "comments.tex"
    fb2.write_text("% Reviewer 2\n% Add more experiments.\n")
    return tmp_path


@pytest.fixture
def pdf_only_dir(tmp_path):
    """Iterate dir with only a PDF (no .tex paper)."""
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake content")
    fb = tmp_path / "feedback.md"
    fb.write_text("Please fix the abstract.\n")
    return tmp_path


@pytest.fixture
def workspace(tmp_path):
    """Empty workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


# ---------------------------------------------------------------------------
# validate_iterate_dir
# ---------------------------------------------------------------------------

class TestValidateIterateDir:
    def test_valid_dir(self, iterate_dir):
        result = validate_iterate_dir(str(iterate_dir))
        assert result["paper_tex"] is not None
        assert "final_paper.tex" in result["paper_tex"]
        assert len(result["feedback_files"]) == 2

    def test_nonexistent_dir(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            validate_iterate_dir(str(tmp_path / "nonexistent"))

    def test_no_paper(self, tmp_path):
        (tmp_path / "feedback.md").write_text("some feedback")
        with pytest.raises(ValueError, match="No paper found"):
            validate_iterate_dir(str(tmp_path))

    def test_no_feedback(self, tmp_path):
        (tmp_path / "final_paper.tex").write_text(r"\documentclass{article}")
        with pytest.raises(ValueError, match="No feedback files"):
            validate_iterate_dir(str(tmp_path))

    def test_prefers_known_paper_names(self, tmp_path):
        """When multiple .tex files exist, prefer known names like final_paper."""
        (tmp_path / "final_paper.tex").write_text("short")
        (tmp_path / "huge_other.tex").write_text("x" * 10000)  # larger but not a known name
        (tmp_path / "feedback.md").write_text("fix it")
        result = validate_iterate_dir(str(tmp_path))
        assert "final_paper.tex" in result["paper_tex"]

    def test_fallback_largest_tex(self, tmp_path):
        """When no known names, pick the largest .tex as the paper."""
        (tmp_path / "a.tex").write_text("small")
        (tmp_path / "b.tex").write_text("x" * 500)
        (tmp_path / "feedback.md").write_text("fix it")
        result = validate_iterate_dir(str(tmp_path))
        assert "b.tex" in result["paper_tex"]
        # a.tex should be in feedback_files
        assert any("a.tex" in f for f in result["feedback_files"])

    def test_pdf_only(self, pdf_only_dir):
        result = validate_iterate_dir(str(pdf_only_dir))
        assert result["paper_tex"] is None
        assert result["paper_pdf"] is not None
        assert len(result["feedback_files"]) == 1


# ---------------------------------------------------------------------------
# parse_feedback_files
# ---------------------------------------------------------------------------

class TestParseFeedbackFiles:
    def test_basic_parse(self, iterate_dir):
        info = validate_iterate_dir(str(iterate_dir))
        items = parse_feedback_files(info["feedback_files"])
        assert len(items) == 2
        assert all("content" in item for item in items)
        assert all(item["format"] in ("tex", "md") for item in items)

    def test_empty_files_skipped(self, tmp_path):
        empty = tmp_path / "empty.md"
        empty.write_text("")
        nonempty = tmp_path / "real.md"
        nonempty.write_text("feedback here")
        items = parse_feedback_files([str(empty), str(nonempty)])
        assert len(items) == 1
        assert items[0]["source"] == "real.md"


# ---------------------------------------------------------------------------
# extract_paper_content
# ---------------------------------------------------------------------------

class TestExtractPaperContent:
    def test_tex_extraction(self, iterate_dir, workspace):
        info = validate_iterate_dir(str(iterate_dir))
        content = extract_paper_content(info["paper_tex"], info["paper_pdf"], str(workspace))
        assert "Hello world" in content
        # Prior paper should be copied
        prior_dir = workspace / "paper_workspace" / "prior_paper"
        assert prior_dir.exists()
        assert (prior_dir / "final_paper.tex").exists()

    def test_no_paper(self, workspace):
        content = extract_paper_content(None, None, str(workspace))
        assert content == ""


# ---------------------------------------------------------------------------
# structure_feedback
# ---------------------------------------------------------------------------

class TestStructureFeedback:
    def test_basic_structure(self):
        items = [
            {"source": "review.md", "content": "Fix intro", "format": "md"},
            {"source": "notes.tex", "content": "Add more refs", "format": "tex"},
        ]
        result = structure_feedback(items)
        assert "# Iteration Feedback" in result
        assert "review.md" in result
        assert "notes.tex" in result
        assert "Fix intro" in result

    def test_empty_items(self):
        assert structure_feedback([]) == ""


# ---------------------------------------------------------------------------
# build_iterate_state_seed
# ---------------------------------------------------------------------------

class TestBuildIterateStateSeed:
    def test_full_seed(self, iterate_dir, workspace):
        seed = build_iterate_state_seed(str(iterate_dir), str(workspace))
        assert seed["iterate_mode"] is True
        assert seed["iterate_prior_paper_path"] is not None
        assert seed["iterate_feedback_path"] is not None
        assert "iterate_feedback_summary" in seed
        assert "formalized_results" in seed

        # Check workspace artifacts were created
        paper_ws = workspace / "paper_workspace"
        assert (paper_ws / "iteration_feedback.md").exists()
        assert (paper_ws / "iteration_manifest.json").exists()

        # Check manifest contents
        manifest = json.loads((paper_ws / "iteration_manifest.json").read_text())
        assert manifest["feedback_count"] == 2
        assert "iterate_dir" in manifest
