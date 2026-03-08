"""
Tests for consortium/budget.py — BudgetManager enforcement, ledger writes, and lock files.
"""

import json
import os
import pytest

from consortium.budget import BudgetManager, BudgetExceededError

_PRICING = {
    "claude-opus-4-6": {"input_per_1k": 0.015, "output_per_1k": 0.075},
    "gpt-5": {"input_per_1k": 0.01, "output_per_1k": 0.03},
}


def _make_manager(tmp_path, usd_limit=10.0, fail_closed=False):
    state = str(tmp_path / "budget_state.json")
    ledger = str(tmp_path / "budget_ledger.jsonl")
    lock = str(tmp_path / "budget.lock")
    return BudgetManager(
        usd_limit=usd_limit,
        pricing=_PRICING,
        state_path=state,
        ledger_path=ledger,
        lock_path=lock,
        hard_stop=True,
        fail_closed=fail_closed,
    )


class TestBudgetManagerBasics:
    def test_initial_state_zero(self, tmp_path):
        mgr = _make_manager(tmp_path)
        assert mgr.total_usd == 0.0

    def test_record_usage_accumulates_cost(self, tmp_path):
        mgr = _make_manager(tmp_path)
        cost = mgr.record_usage("claude-opus-4-6", prompt_tokens=1000, completion_tokens=500)
        # 1k input * 0.015 + 0.5k output * 0.075 = 0.015 + 0.0375 = 0.0525
        assert abs(cost - 0.0525) < 1e-6
        assert abs(mgr.total_usd - 0.0525) < 1e-6

    def test_record_usage_writes_ledger(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.record_usage("gpt-5", prompt_tokens=2000, completion_tokens=1000)
        ledger_path = str(tmp_path / "budget_ledger.jsonl")
        assert os.path.exists(ledger_path)
        with open(ledger_path) as f:
            line = json.loads(f.readline())
        assert line["model_id"] == "gpt-5"
        assert line["prompt_tokens"] == 2000
        assert line["completion_tokens"] == 1000

    def test_record_usage_persists_state(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.record_usage("gpt-5", prompt_tokens=1000, completion_tokens=1000)
        # Reload from disk
        mgr2 = _make_manager(tmp_path)
        assert abs(mgr2.total_usd - mgr.total_usd) < 1e-6

    def test_multiple_models_tracked_separately(self, tmp_path):
        mgr = _make_manager(tmp_path)
        mgr.record_usage("claude-opus-4-6", prompt_tokens=1000, completion_tokens=0)
        mgr.record_usage("gpt-5", prompt_tokens=1000, completion_tokens=0)
        assert "claude-opus-4-6" in mgr.by_model
        assert "gpt-5" in mgr.by_model


class TestBudgetEnforcement:
    def test_check_budget_passes_under_limit(self, tmp_path):
        mgr = _make_manager(tmp_path, usd_limit=100.0)
        mgr.record_usage("gpt-5", prompt_tokens=100, completion_tokens=100)
        mgr.check_budget()  # should not raise

    def test_check_budget_raises_at_limit(self, tmp_path):
        mgr = _make_manager(tmp_path, usd_limit=0.001)
        # record_usage creates lock file but does not raise
        mgr.record_usage("gpt-5", prompt_tokens=10000, completion_tokens=10000)
        # check_budget raises when total exceeds limit
        with pytest.raises(BudgetExceededError):
            mgr.check_budget()

    def test_lock_file_created_on_budget_exceeded(self, tmp_path):
        mgr = _make_manager(tmp_path, usd_limit=0.001)
        lock_path = str(tmp_path / "budget.lock")
        mgr.record_usage("gpt-5", prompt_tokens=10000, completion_tokens=10000)
        assert os.path.exists(lock_path)

    def test_lock_file_blocks_subsequent_calls(self, tmp_path):
        mgr = _make_manager(tmp_path, usd_limit=0.001)
        mgr.record_usage("gpt-5", prompt_tokens=10000, completion_tokens=10000)
        # check_budget should raise due to lock file
        with pytest.raises(BudgetExceededError):
            mgr.check_budget()

    def test_unknown_model_fail_closed_raises(self, tmp_path):
        mgr = _make_manager(tmp_path, fail_closed=True)
        with pytest.raises(BudgetExceededError, match="no pricing configured"):
            mgr.record_usage("unknown-model-xyz", prompt_tokens=100, completion_tokens=100)

    def test_unknown_model_fail_open_returns_zero_cost(self, tmp_path):
        mgr = _make_manager(tmp_path, fail_closed=False)
        cost = mgr.record_usage("unknown-model-xyz", prompt_tokens=100, completion_tokens=100)
        assert cost == 0.0
