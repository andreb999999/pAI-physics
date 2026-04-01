"""Tests for OpenRouterDeepResearchTool."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RESPONSE_JSON = json.dumps({
    "papers": [
        {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
            "year": 2017,
            "venue": "NeurIPS",
            "abstract": "We propose a new simple network architecture, the Transformer.",
            "arxiv_id": "1706.03762",
            "doi": "",
            "url": "https://arxiv.org/abs/1706.03762",
            "citation_count": 100000,
            "relevance_summary": "Foundational work introducing the Transformer architecture.",
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "authors": ["Jacob Devlin", "Ming-Wei Chang"],
            "year": 2019,
            "venue": "NAACL",
            "abstract": "We introduce BERT, a bidirectional transformer for language understanding.",
            "arxiv_id": "1810.04805",
            "doi": "",
            "url": "https://arxiv.org/abs/1810.04805",
            "citation_count": 80000,
            "relevance_summary": "Key follow-up applying transformers to pre-training.",
        },
    ],
    "search_summary": "Found seminal transformer papers.",
})


def _mock_completion_response(content: str):
    """Create a mock litellm.completion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpenRouterDeepResearchTool:
    """Unit tests for OpenRouterDeepResearchTool."""

    def setup_method(self):
        """Clear the response cache before each test to prevent cross-test pollution."""
        from consortium.toolkits.search.deep_research.openrouter_deep_research_tool import (
            clear_response_cache,
        )
        clear_response_cache()

    def _get_tool(self):
        from consortium.toolkits.search.deep_research.openrouter_deep_research_tool import (
            OpenRouterDeepResearchTool,
        )
        return OpenRouterDeepResearchTool(model_name="test-model")

    @patch("litellm.completion")
    def test_basic_search(self, mock_completion):
        """Successful search returns structured papers and BibTeX."""
        mock_completion.return_value = _mock_completion_response(SAMPLE_RESPONSE_JSON)
        tool = self._get_tool()

        result = json.loads(tool._run("transformer architecture"))

        assert result["total_found"] == 2
        assert len(result["papers"]) == 2
        assert result["papers"][0]["title"] == "Attention Is All You Need"
        assert result["papers"][0]["arxiv_id"] == "1706.03762"
        assert len(result["bibtex_entries"]) == 2
        assert "@" in result["bibtex_entries"][0]
        assert result["search_summary"] == "Found seminal transformer papers."

    @patch("litellm.completion")
    def test_markdown_wrapped_json(self, mock_completion):
        """Handles JSON wrapped in markdown code fences."""
        wrapped = f"```json\n{SAMPLE_RESPONSE_JSON}\n```"
        mock_completion.return_value = _mock_completion_response(wrapped)
        tool = self._get_tool()

        result = json.loads(tool._run("transformers"))
        assert result["total_found"] == 2

    @patch("litellm.completion")
    def test_deduplication(self, mock_completion):
        """Duplicate papers (same title) are removed."""
        dup_data = json.dumps({
            "papers": [
                {"title": "Same Paper Title", "authors": ["A"], "year": 2024,
                 "venue": "ICML", "abstract": "v1", "arxiv_id": "", "doi": "",
                 "url": "", "citation_count": 0, "relevance_summary": ""},
                {"title": "Same Paper Title", "authors": ["A"], "year": 2024,
                 "venue": "ICML", "abstract": "v2", "arxiv_id": "", "doi": "",
                 "url": "", "citation_count": 0, "relevance_summary": ""},
            ],
            "search_summary": "",
        })
        mock_completion.return_value = _mock_completion_response(dup_data)
        tool = self._get_tool()

        result = json.loads(tool._run("test"))
        assert result["total_found"] == 1

    @patch("litellm.completion")
    def test_api_failure_fallback(self, mock_completion):
        """When deep research API fails, falls back to CitationSearchTool."""
        mock_completion.side_effect = Exception("API Error")
        tool = self._get_tool()

        # The fallback itself may also fail (no S2_API_KEY), but should not raise
        result = json.loads(tool._run("test query"))
        # Should have some result structure (either fallback or empty)
        assert "papers" in result or "citations" in result or "error" in result

    @patch("litellm.completion")
    def test_bibtex_generation(self, mock_completion):
        """BibTeX entries are correctly formatted."""
        mock_completion.return_value = _mock_completion_response(SAMPLE_RESPONSE_JSON)
        tool = self._get_tool()

        result = json.loads(tool._run("test"))
        bibtex = result["bibtex_entries"][0]

        assert "@misc{vaswani2017," in bibtex
        assert "Attention Is All You Need" in bibtex
        assert "arXiv preprint arXiv:1706.03762" in bibtex

    def test_model_selection_constructor(self):
        """Constructor model_name takes precedence."""
        tool = self._get_tool()
        assert tool._get_model_id() == "test-model"

    @patch.dict(os.environ, {"DEEP_RESEARCH_MODEL": "openrouter/custom-model"})
    def test_model_selection_env_var(self):
        """Env var is used when no constructor model_name."""
        from consortium.toolkits.search.deep_research.openrouter_deep_research_tool import (
            OpenRouterDeepResearchTool,
        )
        tool = OpenRouterDeepResearchTool()
        assert tool._get_model_id() == "openrouter/custom-model"

    def test_model_selection_default(self):
        """Default model is perplexity/sonar-deep-research via OpenRouter."""
        from consortium.toolkits.search.deep_research.openrouter_deep_research_tool import (
            OpenRouterDeepResearchTool,
        )
        # Clear env var if set
        env = os.environ.copy()
        env.pop("DEEP_RESEARCH_MODEL", None)
        with patch.dict(os.environ, env, clear=True):
            tool = OpenRouterDeepResearchTool()
            assert tool._get_model_id() == "openrouter/perplexity/sonar-deep-research"

    @patch("litellm.completion")
    def test_empty_papers(self, mock_completion):
        """Empty paper list returns gracefully."""
        mock_completion.return_value = _mock_completion_response(
            json.dumps({"papers": [], "search_summary": "No papers found."})
        )
        tool = self._get_tool()

        result = json.loads(tool._run("obscure query"))
        assert result["total_found"] == 0
        assert result["papers"] == []
        assert result["bibtex_entries"] == []

    @patch("litellm.completion")
    def test_malformed_response_triggers_fallback(self, mock_completion):
        """Non-JSON response triggers fallback."""
        mock_completion.return_value = _mock_completion_response(
            "I found some papers but can't format them properly."
        )
        tool = self._get_tool()

        result = json.loads(tool._run("test"))
        # Should not crash — returns either fallback or empty result
        assert isinstance(result, dict)

    def test_normalize_authors_string(self):
        """Authors given as string are split into list."""
        tool = self._get_tool()
        papers = tool._normalize_papers([
            {"title": "Test Paper With Long Title", "authors": "Alice, Bob, Charlie",
             "year": 2024, "venue": "ICML"}
        ])
        assert papers[0]["authors"] == ["Alice", "Bob", "Charlie"]

    def test_arxiv_id_cleanup(self):
        """arXiv ID prefixes are stripped."""
        tool = self._get_tool()
        papers = tool._normalize_papers([
            {"title": "Test Paper With Long Title Here",
             "authors": ["A"], "year": 2024, "arxiv_id": "arXiv: 2301.12345"}
        ])
        assert papers[0]["arxiv_id"] == "2301.12345"
