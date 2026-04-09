"""Smoke tests for high-value documentation references."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_referenced_docs_files_exist():
    expected_paths = [
        "README.md",
        "docs/architecture.md",
        "docs/engaging_setup.md",
        "CAMPAIGN_QUICKSTART.md",
        "OpenClaw_Use_Guide.md",
        "examples/quickstart/campaign.yaml",
        "examples/quickstart/task.txt",
        "scripts/launch_openclaw_gateway.sh",
        "campaign_template.yaml",
    ]

    for rel_path in expected_paths:
        assert (REPO_ROOT / rel_path).exists(), f"Missing documented file: {rel_path}"


def test_docs_do_not_reference_stale_openpi_or_old_config_terms():
    doc_paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "architecture.md",
        REPO_ROOT / "docs" / "engaging_setup.md",
        REPO_ROOT / "CAMPAIGN_QUICKSTART.md",
        REPO_ROOT / "OpenClaw_Use_Guide.md",
    ]

    banned_terms = [
        "OpenPI",
        "~/.openpi",
        "counsel_enabled",
        "planning.base_task:",
        "campaign.yaml at the root",
    ]

    for path in doc_paths:
        text = path.read_text()
        for term in banned_terms:
            assert term not in text, f"{path.name} still contains stale term: {term}"


def test_onboarding_docs_center_the_msc_cli():
    doc_paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "examples" / "quickstart" / "README.md",
    ]

    required_terms = [
        "msc setup",
        "msc run",
    ]
    banned_terms = [
        "python launch_multiagent.py",
        "./poggioaimsc run",
    ]

    for path in doc_paths:
        text = path.read_text()
        for term in required_terms:
            assert term in text, f"{path.name} should mention canonical CLI flow: {term}"
        for term in banned_terms:
            assert term not in text, f"{path.name} should not center legacy entrypoint: {term}"
