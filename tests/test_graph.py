"""
Tests for consortium/graph.py — pipeline stage construction and routing.
"""

import pytest


class TestBuildPipelineStages:
    def test_base_pipeline_has_fourteen_stages(self):
        from consortium.graph import build_pipeline_stages
        stages = build_pipeline_stages(enable_math_agents=False)
        # DISCOVERY(3) + EXPERIMENT(5) + POST_TRACK(6) = 14
        assert len(stages) == 14

    def test_base_pipeline_stage_names(self):
        from consortium.graph import build_pipeline_stages
        stages = build_pipeline_stages(enable_math_agents=False)
        assert stages[0] == "ideation_agent"
        assert stages[-1] == "reviewer_agent"

    def test_math_pipeline_has_twenty_stages(self):
        from consortium.graph import build_pipeline_stages
        stages = build_pipeline_stages(enable_math_agents=True)
        # DISCOVERY(3) + MATH(6) + EXPERIMENT(5) + POST_TRACK(6) = 20
        assert len(stages) == 20

    def test_math_stages_inserted_after_discovery_before_experiment(self):
        from consortium.graph import build_pipeline_stages
        stages = build_pipeline_stages(enable_math_agents=True)
        # Math stages come after discovery, before experiment track
        planner_idx = stages.index("research_planner_agent")
        math_lit_idx = stages.index("math_literature_agent")
        exp_lit_idx = stages.index("experiment_literature_agent")
        assert math_lit_idx > planner_idx, "Math stages should follow discovery"
        assert math_lit_idx < exp_lit_idx, "Math stages should precede experiment track"

    def test_math_stages_before_resource_preparation(self):
        from consortium.graph import build_pipeline_stages
        stages = build_pipeline_stages(enable_math_agents=True)
        rp_idx = stages.index("resource_preparation_agent")
        assert stages.index("proof_transcription_agent") < rp_idx

    def test_all_base_stages_present_in_math_pipeline(self):
        from consortium.graph import build_pipeline_stages
        base = build_pipeline_stages(enable_math_agents=False)
        math = build_pipeline_stages(enable_math_agents=True)
        for stage in base:
            assert stage in math, f"Base stage '{stage}' missing from math pipeline"

    def test_no_duplicate_stages(self):
        from consortium.graph import build_pipeline_stages
        for enable_math in [False, True]:
            stages = build_pipeline_stages(enable_math)
            assert len(stages) == len(set(stages)), \
                f"Duplicate stages found in pipeline (math={enable_math}): {stages}"

    def test_stage_group_order(self):
        from consortium.graph import build_pipeline_stages
        stages = build_pipeline_stages(enable_math_agents=False)
        # Discovery stages come first
        assert stages[0] == "ideation_agent"
        assert stages[1] == "literature_review_agent"
        assert stages[2] == "research_planner_agent"
        # Experiment track follows
        assert stages[3] == "experiment_literature_agent"
        # Post-track stages end with reviewer
        assert stages[-1] == "reviewer_agent"
        assert "synthesis_literature_review_agent" in stages


class TestStageAliases:
    def test_canonical_stage_name_ideation(self):
        from consortium.runner import _STAGE_ALIASES
        assert _STAGE_ALIASES.get("ideation") == "ideation_agent"

    def test_canonical_stage_name_writeup(self):
        from consortium.runner import _STAGE_ALIASES
        assert _STAGE_ALIASES.get("writeup") == "writeup_agent"

    def test_canonical_stage_name_math_prover(self):
        from consortium.runner import _STAGE_ALIASES
        assert _STAGE_ALIASES.get("math_prover") == "math_prover_agent"

    def test_resolve_start_stage_index_valid(self):
        from consortium.runner import _resolve_start_stage_index
        from consortium.graph import build_pipeline_stages
        stages = build_pipeline_stages(enable_math_agents=False)
        idx = _resolve_start_stage_index("writeup", stages)
        assert stages[idx] == "writeup_agent"

    def test_resolve_start_stage_index_invalid_raises(self):
        from consortium.runner import _resolve_start_stage_index
        from consortium.graph import build_pipeline_stages
        stages = build_pipeline_stages(enable_math_agents=False)
        with pytest.raises(ValueError, match="Unknown --start-from-stage"):
            _resolve_start_stage_index("nonexistent_stage", stages)
