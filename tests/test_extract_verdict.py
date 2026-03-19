"""Tests for _extract_verdict in persona_council.py."""

import pytest

from consortium.persona_council import _extract_verdict


class TestExtractVerdictStructured:
    """Structured VERDICT: marker parsing (preferred path)."""

    def test_accept_at_end(self):
        text = "Some analysis...\nVERDICT: ACCEPT"
        assert _extract_verdict(text) == "ACCEPT"

    def test_reject_at_end(self):
        text = "Some analysis...\nVERDICT: REJECT"
        assert _extract_verdict(text) == "REJECT"

    def test_final_verdict_accept(self):
        text = "Long evaluation...\nFINAL VERDICT: ACCEPT"
        assert _extract_verdict(text) == "ACCEPT"

    def test_case_insensitive(self):
        text = "Analysis...\nverdict: accept"
        assert _extract_verdict(text) == "ACCEPT"

    def test_structured_overrides_body(self):
        """Structured marker at end should win over body mentions."""
        text = "I ACCEPT the premise that...\n\nVERDICT: REJECT"
        assert _extract_verdict(text) == "REJECT"

    def test_structured_with_extra_spaces(self):
        text = "Analysis...\nVERDICT:   ACCEPT  \n"
        assert _extract_verdict(text) == "ACCEPT"


class TestExtractVerdictFallback:
    """Unstructured fallback scan with false-positive filtering."""

    def test_accept_in_body(self):
        text = "Based on my review, I ACCEPT this proposal."
        assert _extract_verdict(text) == "ACCEPT"

    def test_reject_in_body(self):
        text = "Based on my review, I REJECT this proposal."
        assert _extract_verdict(text) == "REJECT"

    def test_reject_the_premise_filtered(self):
        """'reject the premise' should be filtered out."""
        text = "I reject the premise that this is novel. Otherwise the work is solid."
        assert _extract_verdict(text) != "REJECT"

    def test_reject_the_assumption_filtered(self):
        text = "I reject the assumption that scale is sufficient."
        assert _extract_verdict(text) != "REJECT"

    def test_cannot_accept_filtered(self):
        text = "I cannot accept the argument without stronger evidence."
        assert _extract_verdict(text) != "ACCEPT"

    def test_would_reject_the_filtered(self):
        text = "Any reviewer would reject the claim without ablations."
        assert _extract_verdict(text) != "REJECT"


class TestExtractVerdictEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_string(self):
        assert _extract_verdict("") == "UNKNOWN"

    def test_no_verdict_keywords(self):
        text = "This is a thoughtful analysis of the research direction."
        assert _extract_verdict(text) == "UNKNOWN"

    def test_both_keywords_accept_first(self):
        """When both appear in body without structured marker, ACCEPT wins."""
        text = "I ACCEPT the theory track but REJECT the experiment plan."
        assert _extract_verdict(text) == "ACCEPT"

    def test_very_long_text_structured_at_end(self):
        """Structured marker should be found even in very long text."""
        text = "x" * 10000 + "\nVERDICT: REJECT"
        assert _extract_verdict(text) == "REJECT"
