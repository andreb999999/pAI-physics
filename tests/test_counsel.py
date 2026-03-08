"""
Tests for consortium/counsel.py — counsel model specs and sandbox utilities.
"""

import pytest


class TestDefaultCounselModelSpecs:
    def test_default_specs_is_list(self):
        from consortium.counsel import DEFAULT_COUNSEL_MODEL_SPECS
        assert isinstance(DEFAULT_COUNSEL_MODEL_SPECS, list)

    def test_default_specs_has_multiple_models(self):
        from consortium.counsel import DEFAULT_COUNSEL_MODEL_SPECS
        assert len(DEFAULT_COUNSEL_MODEL_SPECS) >= 2, \
            "Counsel mode should use at least 2 models for debate"

    def test_each_spec_has_model_field(self):
        from consortium.counsel import DEFAULT_COUNSEL_MODEL_SPECS
        for spec in DEFAULT_COUNSEL_MODEL_SPECS:
            assert "model" in spec, f"Counsel model spec missing 'model' key: {spec}"

    def test_specs_include_diverse_providers(self):
        from consortium.counsel import DEFAULT_COUNSEL_MODEL_SPECS
        models = [s["model"] for s in DEFAULT_COUNSEL_MODEL_SPECS]
        model_str = " ".join(models).lower()
        # Should include at least two different provider families
        providers = []
        if any("claude" in m or "anthropic" in m for m in [model_str]):
            providers.append("anthropic")
        if any("gpt" in m or "openai" in m for m in [model_str]):
            providers.append("openai")
        if any("gemini" in m or "google" in m for m in [model_str]):
            providers.append("google")
        assert len(providers) >= 2, \
            "Counsel mode should include models from at least 2 providers"


class TestCounselImport:
    def test_create_counsel_models_importable(self):
        from consortium.counsel import create_counsel_models
        assert callable(create_counsel_models)

    def test_counsel_module_has_expected_exports(self):
        import consortium.counsel as counsel
        assert hasattr(counsel, "DEFAULT_COUNSEL_MODEL_SPECS")
        assert hasattr(counsel, "create_counsel_models")
