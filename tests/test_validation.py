"""
Tests for consortium/supervision/result_validation.py and related validators.
"""

import json
import os
import tempfile
import pytest

from consortium.supervision.result_validation import (
    parse_result_payload,
    artifact_exists,
    validate_result_artifacts,
    sanitize_result_payload,
)
from consortium.supervision.review_verdict_validation import validate_review_verdict
from consortium.supervision.paper_quality_validation import validate_paper_quality
from consortium.supervision.math_acceptance_validation import validate_math_acceptance


# ---------------------------------------------------------------------------
# parse_result_payload
# ---------------------------------------------------------------------------

class TestParseResultPayload:
    def test_dict_passthrough(self):
        d = {"status": "ok", "artifacts": ["a.tex"]}
        assert parse_result_payload(d) == d

    def test_json_string(self):
        d = {"status": "ok"}
        assert parse_result_payload(json.dumps(d)) == d

    def test_python_literal_string(self):
        s = "{'status': 'ok', 'artifacts': ['a.tex']}"
        result = parse_result_payload(s)
        assert result == {"status": "ok", "artifacts": ["a.tex"]}

    def test_non_dict_json_returns_none(self):
        assert parse_result_payload(json.dumps([1, 2, 3])) is None

    def test_plain_string_returns_none(self):
        assert parse_result_payload("some plain text") is None

    def test_none_returns_none(self):
        assert parse_result_payload(None) is None


# ---------------------------------------------------------------------------
# artifact_exists
# ---------------------------------------------------------------------------

class TestArtifactExists:
    def test_existing_file(self, tmp_path):
        f = tmp_path / "final_paper.tex"
        f.write_text("content")
        assert artifact_exists("final_paper.tex", str(tmp_path))

    def test_missing_file(self, tmp_path):
        assert not artifact_exists("missing.tex", str(tmp_path))

    def test_absolute_path_existing(self, tmp_path):
        f = tmp_path / "out.tex"
        f.write_text("x")
        assert artifact_exists(str(f), str(tmp_path))

    def test_fallback_paper_workspace(self, tmp_path):
        pw = tmp_path / "paper_workspace"
        pw.mkdir()
        (pw / "final_paper.tex").write_text("x")
        assert artifact_exists("final_paper.tex", str(tmp_path))


# ---------------------------------------------------------------------------
# validate_result_artifacts / sanitize_result_payload
# ---------------------------------------------------------------------------

class TestSanitizeResultPayload:
    def test_removes_missing_artifacts(self, tmp_path):
        (tmp_path / "real.tex").write_text("x")
        payload = {"artifacts": ["real.tex", "ghost.tex"]}
        result = sanitize_result_payload(payload, str(tmp_path))
        assert "real.tex" in result["artifacts"]
        assert "ghost.tex" not in result["artifacts"]
        assert "ghost.tex" in result["missing_artifacts"]

    def test_marks_incomplete_when_required_missing(self, tmp_path):
        payload = {"artifacts": []}
        result = sanitize_result_payload(
            payload, str(tmp_path), required_artifacts=["final_paper.tex"]
        )
        assert result["status"] == "incomplete"
        assert "final_paper.tex" in result["missing_required_artifacts"]

    def test_marks_complete_when_required_present(self, tmp_path):
        (tmp_path / "final_paper.tex").write_text("\\documentclass{article}")
        payload = {"artifacts": ["final_paper.tex"]}
        result = sanitize_result_payload(
            payload, str(tmp_path), required_artifacts=["final_paper.tex"]
        )
        assert result["status"] == "complete"


# ---------------------------------------------------------------------------
# validate_review_verdict
# ---------------------------------------------------------------------------

class TestValidateReviewVerdict:
    def _make_verdict_file(self, tmp_path, verdict_dict):
        pw = tmp_path / "paper_workspace"
        pw.mkdir(exist_ok=True)
        (pw / "review_verdict.json").write_text(json.dumps(verdict_dict))

    def test_missing_file(self, tmp_path):
        result = validate_review_verdict(str(tmp_path))
        assert not result["present"]
        assert not result["is_valid"]

    def test_passing_verdict(self, tmp_path):
        self._make_verdict_file(tmp_path, {
            "overall_score": 9,
            "ai_voice_risk": "low",
            "hard_blockers": [],
            "intro_compliance": {
                "has_questions": True,
                "has_takeaways": True,
                "questions_answered": True,
                "takeaways_supported": True,
            },
        })
        result = validate_review_verdict(str(tmp_path), min_review_score=8)
        assert result["is_valid"]
        assert result["errors"] == []

    def test_score_below_threshold(self, tmp_path):
        self._make_verdict_file(tmp_path, {
            "overall_score": 5,
            "ai_voice_risk": "low",
            "hard_blockers": [],
            "intro_compliance": {
                "has_questions": True, "has_takeaways": True,
                "questions_answered": True, "takeaways_supported": True,
            },
        })
        result = validate_review_verdict(str(tmp_path), min_review_score=8)
        assert not result["is_valid"]
        assert any("below threshold" in e for e in result["errors"])

    def test_ai_voice_risk_high(self, tmp_path):
        self._make_verdict_file(tmp_path, {
            "overall_score": 9,
            "ai_voice_risk": "high",
            "hard_blockers": [],
            "intro_compliance": {
                "has_questions": True, "has_takeaways": True,
                "questions_answered": True, "takeaways_supported": True,
            },
        })
        result = validate_review_verdict(str(tmp_path))
        assert not result["is_valid"]
        assert any("ai_voice_risk" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# validate_paper_quality
# ---------------------------------------------------------------------------

class TestValidatePaperQuality:
    def test_missing_tex(self, tmp_path):
        result = validate_paper_quality(str(tmp_path))
        assert not result["paper_found"]
        assert not result["is_valid"]

    def test_clean_paper(self, tmp_path):
        tex = (
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "\\begin{figure}\\includegraphics{fig.png}\\end{figure}\n"
            "Hello world.\n"
            "\\end{document}\n"
        )
        (tmp_path / "final_paper.tex").write_text(tex)
        result = validate_paper_quality(str(tmp_path))
        assert result["paper_found"]
        assert result["is_valid"]
        assert result["errors"] == []

    def test_placeholder_detected(self, tmp_path):
        tex = "\\documentclass{article}\n\\begin{document}\nTODO fix this\n\\end{document}\n"
        (tmp_path / "final_paper.tex").write_text(tex)
        result = validate_paper_quality(str(tmp_path))
        assert not result["is_valid"]
        assert result["errors"]


# ---------------------------------------------------------------------------
# validate_math_acceptance — no claim graph → valid (math not used)
# ---------------------------------------------------------------------------

class TestValidateMathAcceptance:
    def test_no_claim_graph_is_valid(self, tmp_path):
        result = validate_math_acceptance(str(tmp_path))
        assert result["is_valid"]
        assert not result["graph_present"]

    def test_accepted_claim_missing_proof(self, tmp_path):
        math_dir = tmp_path / "math_workspace"
        math_dir.mkdir()
        graph = {"claims": [{"id": "T_main", "status": "accepted", "depends_on": []}]}
        (math_dir / "claim_graph.json").write_text(json.dumps(graph))
        result = validate_math_acceptance(str(tmp_path))
        assert not result["is_valid"]
        assert any("proof file" in e for e in result["errors"])
