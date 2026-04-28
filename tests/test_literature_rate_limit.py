"""Tests for shared literature provider rate limiting."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_provider_rate_gate_serializes_requests_and_persists_cooldown(tmp_path, monkeypatch):
    from consortium.toolkits.search.provider_rate_limit import (
        ProviderRateGate,
        ProviderRateLimitTimeout,
    )

    monkeypatch.setenv("CONSORTIUM_LIT_RATE_STATE_DIR", str(tmp_path / "lit_state"))
    monkeypatch.setenv("CONSORTIUM_LIT_MAX_WAIT_SEC", "0.20")
    monkeypatch.setenv("CONSORTIUM_SS_MIN_INTERVAL_SEC", "0")
    monkeypatch.setenv("CONSORTIUM_SS_COOLDOWN_SEC", "0.05")
    monkeypatch.setenv("CONSORTIUM_SS_COOLDOWN_MAX_SEC", "0.05")

    entered_first = threading.Event()
    release_first = threading.Event()
    timeline: list[tuple[str, str, float]] = []

    def worker(name: str):
        gate = ProviderRateGate("semantic_scholar")
        with gate.request(f"test {name}", max_wait_seconds=0.5):
            timeline.append((name, "enter", time.time()))
            if name == "first":
                entered_first.set()
                release_first.wait(timeout=1.0)
            timeline.append((name, "exit", time.time()))

    first = threading.Thread(target=worker, args=("first",))
    second = threading.Thread(target=worker, args=("second",))
    first.start()
    assert entered_first.wait(timeout=1.0)
    second.start()
    time.sleep(0.05)
    release_first.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    first_exit = next(ts for name, event, ts in timeline if name == "first" and event == "exit")
    second_enter = next(ts for name, event, ts in timeline if name == "second" and event == "enter")
    assert second_enter >= first_exit

    with ProviderRateGate("semantic_scholar").request("prime cooldown", max_wait_seconds=0.2) as lease:
        lease.mark_saturated("HTTP 429", retry_after_seconds=0.05)

    state_path = Path(tmp_path / "lit_state" / "semantic_scholar.json")
    state = json.loads(state_path.read_text())
    assert state["saturation_streak"] == 1
    assert state["cooldown_until"] > state["last_completed_at"]

    with pytest.raises(ProviderRateLimitTimeout):
        with ProviderRateGate("semantic_scholar").request("immediate retry", max_wait_seconds=0.0):
            pass


def test_citation_search_tool_waits_through_shared_semantic_scholar_cooldown(tmp_path, monkeypatch):
    from consortium.toolkits.search.provider_rate_limit import ProviderRateGate
    from consortium.toolkits.writeup.citation_search_tool import CitationSearchTool

    monkeypatch.setenv("CONSORTIUM_LIT_RATE_STATE_DIR", str(tmp_path / "lit_state"))
    monkeypatch.setenv("CONSORTIUM_LIT_MAX_WAIT_SEC", "0.30")
    monkeypatch.setenv("CONSORTIUM_SS_MIN_INTERVAL_SEC", "0")
    monkeypatch.setenv("CONSORTIUM_SS_COOLDOWN_SEC", "0.05")
    monkeypatch.setenv("CONSORTIUM_SS_COOLDOWN_MAX_SEC", "0.05")

    with ProviderRateGate("semantic_scholar").request("prime cooldown", max_wait_seconds=0.2) as lease:
        lease.mark_saturated("HTTP 429", retry_after_seconds=0.05)

    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.json.return_value = {
        "data": [
            {
                "title": "Test Paper",
                "authors": [{"name": "Alice Example"}],
                "year": 2024,
                "abstract": "Abstract text.",
                "citationCount": 5,
                "venue": "ICML",
                "externalIds": {"ArXiv": "2301.12345"},
                "url": "https://example.com/test-paper",
            }
        ]
    }
    monkeypatch.setattr(
        "consortium.toolkits.writeup.citation_search_tool.requests.get",
        lambda *args, **kwargs: response,
    )

    tool = CitationSearchTool()
    start = time.time()
    results = tool._search_semantic_scholar("test paper", 1)
    elapsed = time.time() - start

    assert elapsed >= 0.04
    assert len(results) == 1
    assert results[0]["title"] == "Test Paper"
