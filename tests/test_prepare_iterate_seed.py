from __future__ import annotations

import json
from pathlib import Path

from consortium.iterate import validate_iterate_dir
from consortium.campaign.spec import Stage
from scripts.campaign_heartbeat import _validate_artifact_content
from scripts import prepare_iterate_seed as seed_tool


def _sample_review_text() -> str:
    return """NeurIPS 2026 Reviewer Notes
Review 1 (Score: 6 - Weak Accept)
Summary:
Review one summary.
Strengths:
- Strong theorem.
Weaknesses:
- A6a is underspecified.
Questions for Rebuttal:
1. Can you provide an explicit A6a bound?
Review 2 (Score: 5 - Marginally Below Acceptance Threshold)
Summary:
Review two summary.
Strengths:
- Good structure.
Weaknesses:
- The title overclaims implicit bias.
Questions for Rebuttal:
1. What regularizer is actually minimized?
Review 3 (Score: 7 - Accept)
Summary:
Review three summary.
Strengths:
- Nice mechanism story.
Weaknesses:
- Need transformer evidence.
Questions for Rebuttal:
1. Can you add GPT-2 Small style evidence?
"""


def test_prepare_iterate_seed_generates_curated_iterate_dir(tmp_path, monkeypatch):
    latest_pdf = tmp_path / "latest.pdf"
    latest_pdf.write_bytes(b"%PDF-1.4\nlatest")
    latest_tex = tmp_path / "latest.tex"
    latest_tex.write_text(
        "\\title{The Implicit Bias of the Muon Optimizer: Spectral Flattening, Concurrent Acquisition, and Batch Scaling}\n"
        "These results explain why Muon benefits language models.\n"
        "A6a is approximate.\n"
        "weight decay remains tricky.\n"
        "momentum caveat.\n"
        "Gronich max-margin.\n"
    )
    reviews_pdf = tmp_path / "reviews.pdf"
    reviews_pdf.write_bytes(b"%PDF-1.4\nreviews")
    reviewed_submission_pdf = tmp_path / "reviewed.pdf"
    reviewed_submission_pdf.write_bytes(b"%PDF-1.4\nreviewed")
    verdict_json = tmp_path / "review_verdict.json"
    verdict_json.write_text(
        json.dumps(
            {
                "overall_score": 8,
                "hard_blockers": [],
                "must_fix_actions": [
                    {"action": "Restore figures in the final PDF."}
                ],
                "nice_to_fix_actions": [
                    {"action": "Address unresolved citation placeholders."}
                ],
                "iteration_assessment": {
                    "regression_issues": ["Missing visual figures in the layout check."]
                },
            }
        )
    )
    report_tex = tmp_path / "review_report.tex"
    report_tex.write_text("Missing visuals. Multiple question marks detected (16).")
    output_dir = tmp_path / "seed"

    monkeypatch.setattr(seed_tool, "extract_pdf_text", lambda path: _sample_review_text())

    seed_tool.prepare_iterate_seed(
        latest_paper_pdf=latest_pdf,
        latest_paper_tex=latest_tex,
        reviews_pdf=reviews_pdf,
        reviewed_submission_pdf=reviewed_submission_pdf,
        internal_verdict_json=verdict_json,
        internal_report_tex=report_tex,
        output_dir=output_dir,
        supplemental_markdown=[],
        force=False,
    )

    assert (output_dir / "paper.pdf").exists()
    assert (output_dir / "neurips_reviews.md").exists()
    assert (output_dir / "review_alignment.md").exists()
    assert (output_dir / "human_directive.md").exists()
    assert (output_dir / "internal_v4_checks.md").exists()
    assert (output_dir / "seed_manifest.json").exists()

    pdfs = sorted(path.name for path in output_dir.glob("*.pdf"))
    assert pdfs == ["paper.pdf"]

    iterate_info = validate_iterate_dir(str(output_dir))
    assert iterate_info["paper_tex"] is None
    assert iterate_info["paper_pdf"].endswith("paper.pdf")
    assert len(iterate_info["feedback_files"]) == 4

    reviews_md = (output_dir / "neurips_reviews.md").read_text()
    assert "## Review 1" in reviews_md
    assert "## Review 2" in reviews_md
    assert "## Review 3" in reviews_md
    assert "## Consolidated Action Items" in reviews_md

    manifest = json.loads((output_dir / "seed_manifest.json").read_text())
    assert manifest["canonical_prior_paper"]["path"] == str(latest_pdf)


def test_v5_artifact_validators_reject_old_overclaim_language(tmp_path):
    paper_workspace = tmp_path / "paper_workspace"
    paper_workspace.mkdir()
    (paper_workspace / "final_paper.tex").write_text(
        "\\title{The Implicit Bias of the Muon Optimizer: Spectral Flattening, Concurrent Acquisition, and Batch Scaling}\n"
        "These results explain why Muon benefits language models.\n"
        "This section mentions hypothesis, weight decay, and A6a.\n"
    )

    stage = Stage.from_dict(
        {
            "id": "iterate_v5",
            "task_file": "task.txt",
            "success_artifacts": {"required": ["paper_workspace/final_paper.tex"]},
            "artifact_validators": {
                "paper_workspace/final_paper.tex": {
                    "must_contain": ["A6a", "weight decay", "hypothesis"],
                    "must_not_contain": [
                        "The Implicit Bias of the Muon Optimizer: Spectral Flattening, Concurrent Acquisition, and Batch Scaling",
                        "These results explain why Muon benefits language models",
                    ],
                }
            },
        }
    )

    errors = _validate_artifact_content(str(tmp_path), stage)

    assert any("contains forbidden content 'The Implicit Bias of the Muon Optimizer" in err for err in errors)
    assert any("contains forbidden content 'These results explain why Muon benefits language models'" in err for err in errors)
