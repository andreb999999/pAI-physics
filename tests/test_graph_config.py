"""Tests for consortium.graph_config (ResearchGraphConfig and sub-configs)."""

import json

import pytest

from consortium.graph_config import (
    ArtifactEnforcementConfig,
    DualityCheckConfig,
    PersonaCouncilConfig,
    ResearchGraphConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTINEL_MODEL = object()


def _minimal_config(**overrides):
    """Return a config with only the two required fields."""
    kwargs = {"model": _SENTINEL_MODEL, "workspace_dir": "/tmp/ws"}
    kwargs.update(overrides)
    return ResearchGraphConfig(**kwargs)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    """All defaults must match the old 27-param function signature."""

    def test_top_level_defaults(self):
        cfg = _minimal_config()
        assert cfg.pipeline_mode == "default"
        assert cfg.enable_math_agents is False
        assert cfg.enable_milestone_gates is False
        assert cfg.adversarial_verification is False
        assert cfg.min_review_score == 8
        assert cfg.followup_max_iterations == 3
        assert cfg.manager_max_steps == 50
        assert cfg.authorized_imports is None
        assert cfg.summary_model_id == "claude-sonnet-4-6"
        assert cfg.checkpointer is None
        assert cfg.counsel_models is None
        assert cfg.budget_manager is None
        assert cfg.model_registry is None
        assert cfg.tree_search is None

    def test_artifact_defaults(self):
        cfg = _minimal_config()
        a = cfg.artifacts
        assert a.enforce_paper_artifacts is False
        assert a.enforce_editorial_artifacts is False
        assert a.require_pdf is False
        assert a.require_experiment_plan is False
        assert a.lit_review_max_attempts == 2

    def test_persona_council_defaults(self):
        cfg = _minimal_config()
        pc = cfg.persona_council
        assert pc.specs is None
        assert pc.debate_rounds == 3
        assert pc.synthesis_model == "claude-opus-4-6"
        assert pc.max_post_vote_retries == 1

    def test_duality_check_defaults(self):
        cfg = _minimal_config()
        dc = cfg.duality_check
        assert dc.enabled is True
        assert dc.model == "claude-opus-4-6"


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_roundtrip_minimal(self):
        cfg = _minimal_config()
        d = cfg.to_dict()
        # Ensure it's JSON-safe
        json.dumps(d)

        restored = ResearchGraphConfig.from_dict(d, model=_SENTINEL_MODEL)
        assert restored.workspace_dir == cfg.workspace_dir
        assert restored.pipeline_mode == cfg.pipeline_mode
        assert restored.min_review_score == cfg.min_review_score

    def test_roundtrip_full(self):
        cfg = ResearchGraphConfig(
            model=_SENTINEL_MODEL,
            workspace_dir="/tmp/full",
            pipeline_mode="full_research",
            enable_math_agents=True,
            enable_milestone_gates=True,
            adversarial_verification=True,
            min_review_score=6,
            followup_max_iterations=5,
            manager_max_steps=100,
            authorized_imports=["numpy", "scipy"],
            summary_model_id="gpt-4o",
            persona_council=PersonaCouncilConfig(
                specs=[{"role": "mathematician"}],
                debate_rounds=5,
                synthesis_model="gpt-4o",
                max_post_vote_retries=2,
            ),
            duality_check=DualityCheckConfig(enabled=False, model="gpt-4o"),
            artifacts=ArtifactEnforcementConfig(
                enforce_paper_artifacts=True,
                enforce_editorial_artifacts=True,
                require_pdf=True,
                require_experiment_plan=True,
                lit_review_max_attempts=4,
            ),
        )
        d = cfg.to_dict()
        json.dumps(d)  # JSON-safe

        restored = ResearchGraphConfig.from_dict(
            d, model=_SENTINEL_MODEL, checkpointer="ckpt"
        )
        assert restored.workspace_dir == "/tmp/full"
        assert restored.pipeline_mode == "full_research"
        assert restored.enable_math_agents is True
        assert restored.min_review_score == 6
        assert restored.authorized_imports == ["numpy", "scipy"]
        assert restored.summary_model_id == "gpt-4o"
        assert restored.checkpointer == "ckpt"  # runtime kwarg

        # Sub-configs
        assert restored.persona_council.debate_rounds == 5
        assert restored.persona_council.specs == [{"role": "mathematician"}]
        assert restored.duality_check.enabled is False
        assert restored.artifacts.require_pdf is True
        assert restored.artifacts.lit_review_max_attempts == 4


# ---------------------------------------------------------------------------
# Runtime field exclusion
# ---------------------------------------------------------------------------


class TestRuntimeExclusion:
    def test_runtime_fields_absent_from_dict(self):
        cfg = ResearchGraphConfig(
            model=_SENTINEL_MODEL,
            workspace_dir="/tmp/ws",
            checkpointer="ckpt",
            counsel_models=["a", "b"],
            budget_manager="bm",
            model_registry="mr",
        )
        d = cfg.to_dict()
        for key in ("model", "checkpointer", "counsel_models",
                     "budget_manager", "model_registry"):
            assert key not in d, f"runtime field {key!r} leaked into to_dict()"


# ---------------------------------------------------------------------------
# Nested sub-config structure
# ---------------------------------------------------------------------------


class TestNestedStructure:
    def test_sub_configs_are_nested_dicts(self):
        cfg = _minimal_config()
        d = cfg.to_dict()
        assert isinstance(d["persona_council"], dict)
        assert isinstance(d["duality_check"], dict)
        assert isinstance(d["artifacts"], dict)
        # tree_search is None by default so should be absent
        assert "tree_search" not in d

    def test_tree_search_included_when_set(self):
        """When tree_search has a to_dict(), it should appear in output."""

        class FakeTreeConfig:
            def to_dict(self):
                return {"enabled": True, "max_depth": 4}

        cfg = _minimal_config(tree_search=FakeTreeConfig())
        d = cfg.to_dict()
        assert d["tree_search"] == {"enabled": True, "max_depth": 4}
