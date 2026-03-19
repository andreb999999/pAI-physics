"""Tests for build_lit_review_gate_node in graph.py — novelty gating."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


def _make_gate(workspace_dir):
    """Build a lit_review_gate_node with the given workspace."""
    from consortium.graph import build_lit_review_gate_node
    return build_lit_review_gate_node(workspace_dir)


def _write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


class TestNoveltyGating:
    """Test that the gate correctly blocks on novelty_flags.json."""

    def test_blocking_claim_routes_to_persona_council(self, tmp_path):
        """A blocking claim should route back to persona_council."""
        paper_ws = tmp_path / "paper_workspace"
        paper_ws.mkdir()

        # Write minimal lit review so we pass the "no lit review" check
        (paper_ws / "literature_review.tex").write_text("\\section{Review}...")

        # Write novelty_flags with a blocking claim
        flags = {
            "claims": [
                {
                    "claim_id": "C1",
                    "claim_text": "Convergence of Adam with cyclic LR",
                    "status": "KNOWN",
                    "blocking": True,
                    "evidence": [{"source": "arxiv:1904.00962", "relationship": "proves_same", "detail": "Proved in 2019"}],
                    "confidence": "high",
                }
            ],
            "overall_novelty_assessment": "Core claim already proven.",
            "has_blocking_issues": True,
        }
        (paper_ws / "novelty_flags.json").write_text(json.dumps(flags))

        gate = _make_gate(str(tmp_path))
        state = {"lit_review_attempts": 0}
        result = gate(state)

        assert result["current_agent"] == "persona_council"
        assert "NOVELTY GATE REJECTION" in result["agent_task"]
        assert result["lit_review_attempts"] == 1
        assert result["lit_review_feasibility"]["feasible"] is False

    def test_blocking_claim_exhausted_passes_through(self, tmp_path):
        """Blocking claim with max attempts exhausted should pass to brainstorm."""
        paper_ws = tmp_path / "paper_workspace"
        paper_ws.mkdir()
        (paper_ws / "literature_review.tex").write_text("\\section{Review}...")

        flags = {
            "claims": [{"claim_id": "C1", "claim_text": "X", "status": "KNOWN", "blocking": True, "evidence": [], "confidence": "high"}],
            "overall_novelty_assessment": "Blocked.",
            "has_blocking_issues": True,
        }
        (paper_ws / "novelty_flags.json").write_text(json.dumps(flags))

        gate = _make_gate(str(tmp_path))
        state = {"lit_review_attempts": 2}  # max_attempts = 2
        result = gate(state)

        assert result["current_agent"] == "brainstorm_agent"
        assert "NOVELTY WARNING" in result.get("agent_task", "")

    def test_no_blocking_claims_falls_through_to_llm(self, tmp_path):
        """Non-blocking claims should not trigger novelty gate."""
        paper_ws = tmp_path / "paper_workspace"
        paper_ws.mkdir()
        (paper_ws / "literature_review.tex").write_text("\\section{Review}...")

        flags = {
            "claims": [{"claim_id": "C1", "claim_text": "X", "status": "OPEN", "blocking": False, "evidence": [], "confidence": "high"}],
            "overall_novelty_assessment": "All claims novel.",
            "has_blocking_issues": False,
        }
        (paper_ws / "novelty_flags.json").write_text(json.dumps(flags))

        # Mock the LLM call that the feasibility check makes
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"feasible": true, "reason": "Looks good"}'

        gate = _make_gate(str(tmp_path))
        with patch("litellm.completion", return_value=mock_resp):
            state = {"lit_review_attempts": 0}
            result = gate(state)

        assert result["current_agent"] == "brainstorm_agent"
        assert result["lit_review_feasibility"]["feasible"] is True


class TestBackwardCompatibility:
    """Test that missing novelty_flags.json doesn't break existing behavior."""

    def test_no_novelty_flags_file(self, tmp_path):
        """Without novelty_flags.json, gate should fall through to LLM check."""
        paper_ws = tmp_path / "paper_workspace"
        paper_ws.mkdir()
        (paper_ws / "literature_review.tex").write_text("\\section{Review}...")

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"feasible": true, "reason": "OK"}'

        gate = _make_gate(str(tmp_path))
        with patch("litellm.completion", return_value=mock_resp):
            state = {"lit_review_attempts": 0}
            result = gate(state)

        assert result["current_agent"] == "brainstorm_agent"
        assert result["lit_review_feasibility"]["feasible"] is True

    def test_no_lit_review_passes_through(self, tmp_path):
        """Without literature_review.tex, gate should pass through."""
        paper_ws = tmp_path / "paper_workspace"
        paper_ws.mkdir()

        gate = _make_gate(str(tmp_path))
        state = {"lit_review_attempts": 0}
        result = gate(state)

        assert result["current_agent"] == "brainstorm_agent"
        assert result["lit_review_feasibility"]["feasible"] is True

    def test_malformed_novelty_flags_ignored(self, tmp_path):
        """Malformed JSON in novelty_flags.json should be ignored gracefully."""
        paper_ws = tmp_path / "paper_workspace"
        paper_ws.mkdir()
        (paper_ws / "literature_review.tex").write_text("\\section{Review}...")
        (paper_ws / "novelty_flags.json").write_text("not valid json {{{")

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"feasible": true, "reason": "OK"}'

        gate = _make_gate(str(tmp_path))
        with patch("litellm.completion", return_value=mock_resp):
            state = {"lit_review_attempts": 0}
            result = gate(state)

        # Should fall through to LLM check, not crash
        assert result["current_agent"] == "brainstorm_agent"
