"""
OpenRouterDeepResearchTool — comprehensive academic literature search via OpenRouter.

Uses deep-research models (Perplexity sonar-deep-research by default) through
OpenRouter to search across arXiv, Semantic Scholar, Google Scholar, and other
academic databases simultaneously.

All API calls go through litellm.completion() for budget tracking consistency.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Type

import litellm
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Module-level LRU response cache for deep research queries.
# Keyed by (query, max_papers, focus_area) hash. TTL = 30 min.
# ---------------------------------------------------------------------------
_RESPONSE_CACHE: Dict[str, tuple[float, str]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = 1800  # 30 minutes
_CACHE_MAX_ENTRIES = 256


def _cache_key(query: str, max_papers: int, focus_area: str) -> str:
    raw = f"{query.strip().lower()}|{max_papers}|{focus_area.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    with _CACHE_LOCK:
        entry = _RESPONSE_CACHE.get(key)
        if entry is None:
            return None
        ts, val = entry
        if time.time() - ts > _CACHE_TTL:
            del _RESPONSE_CACHE[key]
            return None
        return val


def _cache_put(key: str, value: str) -> None:
    with _CACHE_LOCK:
        if len(_RESPONSE_CACHE) >= _CACHE_MAX_ENTRIES:
            oldest_key = min(_RESPONSE_CACHE, key=lambda k: _RESPONSE_CACHE[k][0])
            del _RESPONSE_CACHE[oldest_key]
        _RESPONSE_CACHE[key] = (time.time(), value)


def clear_response_cache() -> None:
    """Clear the module-level response cache (useful for tests)."""
    with _CACHE_LOCK:
        _RESPONSE_CACHE.clear()


class OpenRouterDeepResearchInput(BaseModel):
    query: str = Field(
        description=(
            "Research query or topic to investigate. Can be a research question, "
            "paper title, method name, or broad topic. The tool searches across "
            "arXiv, Semantic Scholar, Google Scholar, and other academic databases."
        )
    )
    max_papers: int = Field(
        default=15,
        description="Target number of papers to find (default: 15).",
    )
    focus_area: str = Field(
        default="",
        description=(
            "Optional constraint to narrow the search: specific subfield, "
            "venue filter, time range, or methodology focus."
        ),
    )


_DEEP_RESEARCH_SYSTEM_PROMPT = """\
You are an expert academic literature search assistant. Your job is to find
real, published academic papers relevant to the given research query.

CRITICAL RULES:
1. Only return papers that ACTUALLY EXIST. Never invent titles, authors, or venues.
2. Include arXiv IDs (e.g., 2301.12345) and DOIs when available.
3. Prioritize high-quality venues: ICML, NeurIPS, ICLR, JMLR, AISTATS, COLT,
   Annals of Mathematics, JAMS, Ann. Statist., PTRF, Nature, Science, PNAS.
4. For each paper, provide a substantive summary (not just the abstract).
5. Include the most recent and most cited papers on the topic.

You MUST respond with a JSON object (no markdown fencing) with this exact structure:
{
  "papers": [
    {
      "title": "Full paper title",
      "authors": ["First Author", "Second Author"],
      "year": 2024,
      "venue": "Conference or Journal name",
      "abstract": "Brief abstract or summary of the paper",
      "arxiv_id": "2301.12345 or empty string if not on arXiv",
      "doi": "10.xxxx/... or empty string if unknown",
      "url": "Direct URL to paper page",
      "citation_count": 0,
      "relevance_summary": "2-3 sentences on why this paper is relevant to the query"
    }
  ],
  "search_summary": "Brief summary of what was found and any notable gaps"
}

Return up to {max_papers} papers, ordered by relevance. If fewer relevant papers
exist, return fewer. Quality over quantity.
"""


class OpenRouterDeepResearchTool(BaseTool):
    """Comprehensive academic literature search via OpenRouter deep research models."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "deep_literature_search"
    description: str = (
        "Performs comprehensive academic literature search using deep research models. "
        "Searches across arXiv, Semantic Scholar, Google Scholar, and other academic "
        "databases simultaneously. Returns structured paper metadata with titles, "
        "authors, venues, years, abstracts, arXiv IDs, URLs, BibTeX entries, and "
        "relevance summaries. Use this as your PRIMARY tool for finding academic "
        "papers — it replaces separate Semantic Scholar and arXiv queries with a "
        "single comprehensive search."
    )
    args_schema: Type[BaseModel] = OpenRouterDeepResearchInput

    model_name: Optional[str] = None

    def __init__(self, model_name: Optional[str] = None, **kwargs: Any):
        super().__init__(model_name=model_name, **kwargs)

    def _get_model_id(self) -> str:
        """Determine model ID: constructor param > env var > default."""
        if self.model_name:
            return self.model_name
        return os.getenv(
            "DEEP_RESEARCH_MODEL",
            "openrouter/perplexity/sonar-deep-research",
        )

    def _run(
        self,
        query: str,
        max_papers: int = 15,
        focus_area: str = "",
    ) -> str:
        """Search for academic papers using deep research models.

        Returns a JSON string with keys:
            query, papers, bibtex_entries, total_found, search_summary
        """
        # Check response cache first
        ck = _cache_key(query, max_papers, focus_area)
        cached = _cache_get(ck)
        if cached is not None:
            return cached

        model_id = self._get_model_id()

        user_prompt = f"RESEARCH QUERY: {query}"
        if focus_area:
            user_prompt += f"\n\nFOCUS AREA: {focus_area}"
        user_prompt += f"\n\nFind up to {max_papers} relevant academic papers."

        system_prompt = _DEEP_RESEARCH_SYSTEM_PROMPT.replace(
            "{max_papers}", str(max_papers)
        )

        try:
            resp = litellm.completion(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=16384,
            )
            raw_content = resp.choices[0].message.content or ""
        except Exception as e:
            print(
                f"[OpenRouterDeepResearchTool] API call failed ({e}), "
                f"falling back to CitationSearchTool."
            )
            return self._fallback_search(query, max_papers)

        # Parse the response
        papers = self._parse_response(raw_content)
        if papers is None:
            # Parsing failed entirely — try fallback
            print(
                "[OpenRouterDeepResearchTool] Failed to parse response, "
                "falling back to CitationSearchTool."
            )
            return self._fallback_search(query, max_papers)

        # Deduplicate
        papers = self._deduplicate(papers)

        # Limit to max_papers
        papers = papers[:max_papers]

        # Generate BibTeX entries
        bibtex_entries = []
        for paper in papers:
            bibtex = self._generate_bibtex(paper)
            if bibtex:
                bibtex_entries.append(bibtex)

        # Extract search summary
        search_summary = self._extract_search_summary(raw_content)

        result = json.dumps(
            {
                "query": query,
                "papers": papers,
                "bibtex_entries": bibtex_entries,
                "total_found": len(papers),
                "search_summary": search_summary,
            },
            indent=2,
        )
        _cache_put(ck, result)
        return result

    def _parse_response(self, raw: str) -> Optional[List[Dict[str, Any]]]:
        """Parse the deep research model response into structured paper data."""
        # Try direct JSON parse
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "papers" in data:
                return self._normalize_papers(data["papers"])
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code blocks
        json_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL
        )
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict) and "papers" in data:
                    return self._normalize_papers(data["papers"])
            except json.JSONDecodeError:
                pass

        # Try finding JSON object in the response
        brace_match = re.search(r"\{.*\"papers\"\s*:\s*\[.*\].*\}", raw, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                if isinstance(data, dict) and "papers" in data:
                    return self._normalize_papers(data["papers"])
            except json.JSONDecodeError:
                pass

        return None

    def _normalize_papers(self, papers: List[Any]) -> List[Dict[str, Any]]:
        """Normalize paper entries to a consistent schema."""
        normalized = []
        for p in papers:
            if not isinstance(p, dict):
                continue
            title = p.get("title", "").strip()
            if not title:
                continue
            paper = {
                "title": title,
                "authors": p.get("authors", []),
                "year": str(p.get("year", "")),
                "venue": p.get("venue", ""),
                "abstract": p.get("abstract", ""),
                "arxiv_id": p.get("arxiv_id", ""),
                "doi": p.get("doi", ""),
                "url": p.get("url", ""),
                "citation_count": p.get("citation_count", 0),
                "relevance_summary": p.get("relevance_summary", ""),
                "source": "deep_research",
            }
            # Clean up arxiv_id if it contains prefix
            if paper["arxiv_id"]:
                aid = re.sub(r"^(?:arxiv:?\s*)", "", paper["arxiv_id"], flags=re.IGNORECASE)
                paper["arxiv_id"] = aid.strip()
            # Ensure authors is a list of strings
            if isinstance(paper["authors"], str):
                paper["authors"] = [a.strip() for a in paper["authors"].split(",")]
            normalized.append(paper)
        return normalized

    def _deduplicate(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate papers based on normalized title."""
        seen = set()
        unique = []
        for paper in papers:
            key = re.sub(r"[^\w]", "", paper["title"].lower())
            if len(key) > 10 and key not in seen:
                seen.add(key)
                unique.append(paper)
        return unique

    def _extract_search_summary(self, raw: str) -> str:
        """Extract search_summary from the raw response."""
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data.get("search_summary", "")
        except (json.JSONDecodeError, TypeError):
            pass
        # Try from code block
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict):
                    return data.get("search_summary", "")
            except (json.JSONDecodeError, TypeError):
                pass
        return ""

    @staticmethod
    def _generate_bibtex(paper: Dict[str, Any]) -> Optional[str]:
        """Generate a BibTeX entry for a paper.

        Logic adapted from CitationSearchTool._generate_bibtex.
        """
        title = paper.get("title", "").strip()
        if not title:
            return None

        authors = paper.get("authors", ["Unknown"])
        first_author = authors[0] if authors else "Unknown"
        first_author_last = (
            first_author.split()[-1] if " " in first_author else first_author
        )
        year = paper.get("year", "")

        clean_author = re.sub(r"[^\w]", "", first_author_last.lower())
        citation_key = f"{clean_author}{year}"

        # Choose entry type
        arxiv_id = paper.get("arxiv_id", "")
        venue = paper.get("venue", "").lower()
        if arxiv_id:
            entry_type = "misc"
            note_field = f"arXiv preprint arXiv:{arxiv_id}"
        elif "conference" in venue or venue in {
            "icml", "neurips", "nips", "iclr", "aaai", "ijcai", "cvpr",
            "eccv", "iccv", "acl", "emnlp", "naacl", "aistats", "colt",
        }:
            entry_type = "inproceedings"
            note_field = ""
        elif "journal" in venue or venue in {
            "jmlr", "tmlr", "nature", "science", "pnas",
        }:
            entry_type = "article"
            note_field = ""
        else:
            entry_type = "misc"
            note_field = ""

        author_str = " and ".join(authors) if authors else "Unknown"

        lines = [f"@{entry_type}{{{citation_key},"]
        lines.append(f"  title = {{{title}}},")
        lines.append(f"  author = {{{author_str}}},")
        if year:
            lines.append(f"  year = {{{year}}},")
        if paper.get("venue"):
            raw_venue = paper["venue"]
            if entry_type == "inproceedings":
                lines.append(f"  booktitle = {{{raw_venue}}},")
            elif entry_type == "article":
                lines.append(f"  journal = {{{raw_venue}}},")
        if note_field:
            lines.append(f"  note = {{{note_field}}},")
        if paper.get("url"):
            lines.append(f"  url = {{{paper['url']}}},")
        if paper.get("doi"):
            lines.append(f"  doi = {{{paper['doi']}}},")
        lines.append("}")

        return "\n".join(lines)

    def _fallback_search(self, query: str, max_papers: int) -> str:
        """Fall back to CitationSearchTool when deep research fails."""
        try:
            from ...writeup.citation_search_tool import CitationSearchTool

            fallback = CitationSearchTool()
            return fallback._run(
                search_query=query,
                max_results=max_papers,
                search_source="both",
            )
        except Exception as e:
            print(f"[OpenRouterDeepResearchTool] Fallback also failed: {e}")
            return json.dumps(
                {
                    "query": query,
                    "papers": [],
                    "bibtex_entries": [],
                    "total_found": 0,
                    "search_summary": f"Search unavailable: {e}",
                }
            )
