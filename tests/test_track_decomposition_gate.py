"""Tests for build_track_decomposition_gate_node in graph.py."""

import json
import os
import tempfile

import pytest


def _make_gate(workspace_dir, enable_math_agents=False):
    """Build a track_decomposition_gate_node with the given workspace."""
    from consortium.graph import build_track_decomposition_gate_node
    return build_track_decomposition_gate_node(workspace_dir, enable_math_agents)


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


class TestTrackDecompositionGate:
    """Test that the gate validates and corrects track_decomposition.json."""

    def test_missing_file_defaults_to_empirical(self, tmp_path):
        """Missing track_decomposition.json should default to empirical."""
        (tmp_path / "paper_workspace").mkdir()
        gate = _make_gate(str(tmp_path))
        result = gate({})
        td = result["track_decomposition"]
        assert td["recommended_track"] == "empirical"
        assert len(td["empirical_questions"]) == 1
        assert td["theory_questions"] == []
        assert result["agent_task"] is None

    def test_malformed_json_defaults_to_empirical(self, tmp_path):
        """Corrupt JSON should default to empirical."""
        paper_ws = tmp_path / "paper_workspace"
        paper_ws.mkdir()
        (paper_ws / "track_decomposition.json").write_text("{invalid json!!}")
        gate = _make_gate(str(tmp_path))
        result = gate({})
        td = result["track_decomposition"]
        assert td["recommended_track"] == "empirical"

    def test_theory_downgraded_when_math_disabled(self, tmp_path):
        """theory track should be downgraded to empirical when math agents disabled."""
        paper_ws = tmp_path / "paper_workspace"
        paper_ws.mkdir()
        _write_json(str(paper_ws / "track_decomposition.json"), {
            "theory_questions": ["Prove theorem X"],
            "empirical_questions": ["Test hypothesis Y"],
            "recommended_track": "theory",
            "rationale": "Theory-heavy research",
        })
        gate = _make_gate(str(tmp_path), enable_math_agents=False)
        result = gate({})
        td = result["track_decomposition"]
        assert td["recommended_track"] == "empirical"

    def test_both_downgraded_when_math_disabled(self, tmp_path):
        """both track should be downgraded to empirical when math agents disabled."""
        paper_ws = tmp_path / "paper_workspace"
        paper_ws.mkdir()
        _write_json(str(paper_ws / "track_decomposition.json"), {
            "theory_questions": ["Prove theorem X"],
            "empirical_questions": ["Test hypothesis Y"],
            "recommended_track": "both",
            "rationale": "Mixed research",
        })
        gate = _make_gate(str(tmp_path), enable_math_agents=False)
        result = gate({})
        td = result["track_decomposition"]
        assert td["recommended_track"] == "empirical"

    def test_theory_allowed_when_math_enabled(self, tmp_path):
        """theory track should pass through when math agents enabled."""
        paper_ws = tmp_path / "paper_workspace"
        paper_ws.mkdir()
        _write_json(str(paper_ws / "track_decomposition.json"), {
            "theory_questions": ["Prove theorem X"],
            "empirical_questions": [],
            "recommended_track": "theory",
            "rationale": "Pure theory",
        })
        gate = _make_gate(str(tmp_path), enable_math_agents=True)
        result = gate({})
        td = result["track_decomposition"]
        assert td["recommended_track"] == "theory"
        assert td["theory_questions"] == ["Prove theorem X"]

    def test_empty_theory_questions_recovered_from_goals(self, tmp_path):
        """Empty theory_questions should be recovered from research_goals.json."""
        paper_ws = tmp_path / "paper_workspace"
        paper_ws.mkdir()
        _write_json(str(paper_ws / "track_decomposition.json"), {
            "theory_questions": [],
            "empirical_questions": ["Test Y"],
            "recommended_track": "both",
            "rationale": "Mixed",
        })
        _write_json(str(paper_ws / "research_goals.json"), {
            "goals": [
                {"id": "G1", "description": "Prove convergence bound", "track": "theory"},
                {"id": "G2", "description": "Run ablation study", "track": "experiment"},
            ]
        })
        gate = _make_gate(str(tmp_path), enable_math_agents=True)
        result = gate({})
        td = result["track_decomposition"]
        assert len(td["theory_questions"]) == 1
        assert "G1" in td["theory_questions"][0]

    def test_empty_empirical_questions_recovered_from_goals(self, tmp_path):
        """Empty empirical_questions should be recovered from research_goals.json."""
        paper_ws = tmp_path / "paper_workspace"
        paper_ws.mkdir()
        _write_json(str(paper_ws / "track_decomposition.json"), {
            "theory_questions": [],
            "empirical_questions": [],
            "recommended_track": "empirical",
            "rationale": "Empirical only",
        })
        _write_json(str(paper_ws / "research_goals.json"), {
            "goals": [
                {"id": "G1", "description": "Run scaling experiments", "track": "experiment"},
            ]
        })
        gate = _make_gate(str(tmp_path))
        result = gate({})
        td = result["track_decomposition"]
        assert len(td["empirical_questions"]) == 1
        assert "G1" in td["empirical_questions"][0]

    def test_valid_decomposition_passes_through(self, tmp_path):
        """A valid track_decomposition.json should pass through unchanged."""
        paper_ws = tmp_path / "paper_workspace"
        paper_ws.mkdir()
        original = {
            "theory_questions": [],
            "empirical_questions": ["Test hypothesis Y", "Run ablation Z"],
            "recommended_track": "empirical",
            "rationale": "Empirical-focused study",
        }
        _write_json(str(paper_ws / "track_decomposition.json"), original)
        gate = _make_gate(str(tmp_path))
        result = gate({})
        td = result["track_decomposition"]
        assert td["recommended_track"] == "empirical"
        assert td["empirical_questions"] == original["empirical_questions"]

    def test_recovery_when_goals_also_absent(self, tmp_path):
        """When both theory_questions and research_goals.json are absent, should downgrade."""
        paper_ws = tmp_path / "paper_workspace"
        paper_ws.mkdir()
        _write_json(str(paper_ws / "track_decomposition.json"), {
            "theory_questions": [],
            "empirical_questions": ["Test Y"],
            "recommended_track": "both",
            "rationale": "Mixed",
        })
        # No research_goals.json exists
        gate = _make_gate(str(tmp_path), enable_math_agents=True)
        result = gate({})
        td = result["track_decomposition"]
        # Should downgrade to empirical since theory questions can't be recovered
        assert td["recommended_track"] == "empirical"

    def test_corrected_file_written_back_to_disk(self, tmp_path):
        """Corrections should be persisted back to disk."""
        paper_ws = tmp_path / "paper_workspace"
        paper_ws.mkdir()
        _write_json(str(paper_ws / "track_decomposition.json"), {
            "theory_questions": ["Prove X"],
            "empirical_questions": ["Test Y"],
            "recommended_track": "theory",
            "rationale": "Theory",
        })
        gate = _make_gate(str(tmp_path), enable_math_agents=False)
        gate({})
        with open(str(paper_ws / "track_decomposition.json")) as f:
            on_disk = json.load(f)
        assert on_disk["recommended_track"] == "empirical"
