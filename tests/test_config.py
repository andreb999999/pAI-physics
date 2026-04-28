"""
Tests for consortium/config.py — load_llm_config and filter_model_params.
"""

import os
import tempfile
import pytest
import yaml

from consortium.config import load_llm_config, filter_model_params


# ---------------------------------------------------------------------------
# load_llm_config
# ---------------------------------------------------------------------------

class TestLoadLlmConfig:
    def test_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert load_llm_config() is None

    def test_honors_consortium_llm_config_path_override(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg_path = tmp_path / "rigorous.yaml"
        cfg = {"main_agents": {"model": "claude-opus-4-6"}}
        cfg_path.write_text(yaml.dump(cfg))
        monkeypatch.setenv("CONSORTIUM_LLM_CONFIG_PATH", str(cfg_path))

        result = load_llm_config()

        assert result["main_agents"]["model"] == "claude-opus-4-6"

    def test_loads_valid_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = {"main_agents": {"model": "claude-sonnet-4-5", "reasoning_effort": "high"}}
        (tmp_path / ".llm_config.yaml").write_text(yaml.dump(cfg))
        result = load_llm_config()
        assert result["main_agents"]["model"] == "claude-sonnet-4-5"

    def test_returns_none_on_invalid_yaml(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".llm_config.yaml").write_text(": bad: yaml: [unclosed")
        result = load_llm_config()
        assert result is None


# ---------------------------------------------------------------------------
# filter_model_params — Claude branch
# ---------------------------------------------------------------------------

class TestFilterModelParamsClaude:
    def _make_call_capture(self):
        """Returns a decorator target that captures kwargs."""
        calls = []

        def fake_completion(*args, **kwargs):
            calls.append(kwargs)
            return "ok"

        return fake_completion, calls

    def test_removes_top_p_when_temperature_present(self):
        captured = {}

        @filter_model_params
        def completion(**kwargs):
            captured.update(kwargs)

        completion(model="claude-sonnet-4-5", temperature=0.7, top_p=0.9)
        assert "top_p" not in captured
        assert captured.get("temperature") == 0.7

    def test_budget_tokens_converted_to_thinking(self):
        captured = {}

        @filter_model_params
        def completion(**kwargs):
            captured.update(kwargs)

        completion(model="claude-sonnet-4-5", budget_tokens=8192, max_tokens=16000)
        assert "budget_tokens" not in captured
        assert captured["thinking"] == {"type": "enabled", "budget_tokens": 8192}

    def test_budget_tokens_zero_disables_thinking(self):
        captured = {}

        @filter_model_params
        def completion(**kwargs):
            captured.update(kwargs)

        completion(model="claude-sonnet-4-5", budget_tokens=0)
        assert captured["thinking"] == {"type": "disabled"}


# ---------------------------------------------------------------------------
# filter_model_params — GPT-5 branch
# ---------------------------------------------------------------------------

class TestFilterModelParamsGPT5:
    def test_max_tokens_replaced_with_max_completion_tokens(self):
        captured = {}

        @filter_model_params
        def completion(**kwargs):
            captured.update(kwargs)

        completion(model="gpt-5", max_tokens=4096)
        assert "max_tokens" not in captured
        assert captured["max_completion_tokens"] == 4096

    def test_unsupported_params_dropped(self):
        captured = {}

        @filter_model_params
        def completion(**kwargs):
            captured.update(kwargs)

        completion(model="gpt-5", temperature=0.5, top_p=0.9, stop=["END"])
        assert "temperature" not in captured
        assert "top_p" not in captured
        assert "stop" not in captured
