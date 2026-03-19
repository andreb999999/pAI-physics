"""Tests for DeepResearchNoveltyScanTool — graceful degradation and verdict extraction."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from consortium.toolkits.search.deep_research.deep_research_tool import (
    DeepResearchNoveltyScanTool,
)


class TestGracefulDegradation:
    """Tool should never raise — returns UNCERTAIN on any failure."""

    def test_missing_api_key(self):
        """With no API key, the tool should return UNCERTAIN, not raise."""
        tool = DeepResearchNoveltyScanTool()

        with patch("litellm.completion", side_effect=Exception("No API key set")):
            raw = tool._run("Some claim about convergence rates")

        result = json.loads(raw)
        assert result["verdict"] == "UNCERTAIN"
        assert result["confidence"] == "low"
        assert result["claim_text"] == "Some claim about convergence rates"

    def test_network_error(self):
        """Network errors should be caught gracefully."""
        tool = DeepResearchNoveltyScanTool()

        with patch("litellm.completion", side_effect=ConnectionError("timeout")):
            raw = tool._run("Theorem about PAC learning bounds")

        result = json.loads(raw)
        assert result["verdict"] == "UNCERTAIN"

    def test_empty_response(self):
        """Empty API response should not crash."""
        tool = DeepResearchNoveltyScanTool()

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = ""

        with patch("litellm.completion", return_value=mock_resp):
            raw = tool._run("Some claim")

        result = json.loads(raw)
        # Empty response → heuristic can't find anything → UNCERTAIN
        assert result["verdict"] in ("UNCERTAIN", "OPEN")


class TestVerdictExtraction:
    """Test the static verdict extraction from API responses."""

    def test_structured_verdict_known(self):
        text = "VERDICT: KNOWN\nThis was proven by Smith et al. (2019)."
        assert DeepResearchNoveltyScanTool._extract_verdict(text) == "KNOWN"

    def test_structured_verdict_open(self):
        text = "After exhaustive search...\nVERDICT: OPEN"
        assert DeepResearchNoveltyScanTool._extract_verdict(text) == "OPEN"

    def test_structured_verdict_equivalent(self):
        text = "VERDICT: EQUIVALENT_KNOWN"
        assert DeepResearchNoveltyScanTool._extract_verdict(text) == "EQUIVALENT_KNOWN"

    def test_structured_verdict_partial(self):
        text = "VERDICT: PARTIAL"
        assert DeepResearchNoveltyScanTool._extract_verdict(text) == "PARTIAL"

    def test_heuristic_has_been_proven(self):
        text = "This result has been proven by Wang (2020) in JMLR."
        assert DeepResearchNoveltyScanTool._extract_verdict(text) == "KNOWN"

    def test_heuristic_appears_novel(self):
        text = "The claim appears novel based on my search."
        assert DeepResearchNoveltyScanTool._extract_verdict(text) == "OPEN"

    def test_heuristic_equivalent_result(self):
        text = "An equivalent result was established under different terminology."
        assert DeepResearchNoveltyScanTool._extract_verdict(text) == "EQUIVALENT_KNOWN"

    def test_heuristic_partial_result(self):
        text = "Only a partial result exists for the special case of d=2."
        assert DeepResearchNoveltyScanTool._extract_verdict(text) == "PARTIAL"

    def test_no_signal_returns_uncertain(self):
        text = "I searched but the results are inconclusive."
        assert DeepResearchNoveltyScanTool._extract_verdict(text) == "UNCERTAIN"


class TestConfidenceExtraction:
    """Test confidence level extraction."""

    def test_high_confidence(self):
        assert DeepResearchNoveltyScanTool._extract_confidence("CONFIDENCE: HIGH") == "high"

    def test_low_confidence(self):
        assert DeepResearchNoveltyScanTool._extract_confidence("CONFIDENCE: low") == "low"

    def test_default_medium(self):
        assert DeepResearchNoveltyScanTool._extract_confidence("No confidence stated.") == "medium"


class TestSourceExtraction:
    """Test URL extraction from responses."""

    def test_extracts_urls(self):
        text = "See https://arxiv.org/abs/1904.00962 and https://mathoverflow.net/q/123456"
        sources = DeepResearchNoveltyScanTool._extract_sources(text)
        assert len(sources) == 2
        assert "https://arxiv.org/abs/1904.00962" in sources

    def test_deduplicates(self):
        text = "https://example.com and again https://example.com"
        sources = DeepResearchNoveltyScanTool._extract_sources(text)
        assert len(sources) == 1

    def test_no_urls(self):
        text = "No links here."
        assert DeepResearchNoveltyScanTool._extract_sources(text) == []
