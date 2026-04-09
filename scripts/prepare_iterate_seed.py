#!/usr/bin/env python3
"""Prepare a curated iterate seed directory from a paper PDF and review artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List

from pypdf import PdfReader

DEFAULT_LATEST_PAPER_PDF = Path(
    "/orcd/scratch/orcd/012/mabdel03/AI_Researcher/MSc_Internal/results/"
    "muon_v4_iterate/iterate_v4/paper_workspace/final_paper.pdf"
)
DEFAULT_LATEST_PAPER_TEX = Path(
    "/orcd/scratch/orcd/012/mabdel03/AI_Researcher/MSc_Internal/results/"
    "muon_v4_iterate/iterate_v4/paper_workspace/final_paper.tex"
)
DEFAULT_REVIEWS_PDF = Path(
    "/orcd/scratch/orcd/012/mabdel03/AI_Researcher/muon_v4_iterate/NeurIPS_Muon_Reviews.pdf"
)
DEFAULT_REVIEWED_SUBMISSION_PDF = Path(
    "/orcd/scratch/orcd/012/mabdel03/AI_Researcher/muon_v4_iterate/muon_v4.pdf"
)
DEFAULT_INTERNAL_VERDICT_JSON = Path(
    "/orcd/scratch/orcd/012/mabdel03/AI_Researcher/MSc_Internal/results/"
    "muon_v4_iterate/iterate_v4/paper_workspace/review_verdict.json"
)
DEFAULT_INTERNAL_REPORT_TEX = Path(
    "/orcd/scratch/orcd/012/mabdel03/AI_Researcher/MSc_Internal/results/"
    "muon_v4_iterate/iterate_v4/paper_workspace/review_report.tex"
)
DEFAULT_OUTPUT_DIR = Path(
    "/orcd/scratch/orcd/012/mabdel03/AI_Researcher/muon_v5_iterate_seed"
)


@dataclass(frozen=True)
class ParsedReview:
    review_id: str
    score: str
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    questions: List[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def clean_review_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\nPage \d+\nNeurIPS 2026 Reviewer Notes\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_heading_block(body: str, heading: str, next_headings: Iterable[str]) -> str:
    next_headings = list(next_headings)
    if next_headings:
        escaped_next = "|".join(re.escape(item) for item in next_headings)
        pattern = re.compile(
            rf"{re.escape(heading)}:\s*\n(.*?)(?=\n(?:{escaped_next}):\s*\n|\Z)",
            re.DOTALL,
        )
    else:
        pattern = re.compile(rf"{re.escape(heading)}:\s*\n(.*)\Z", re.DOTALL)
    match = pattern.search(body)
    return match.group(1).strip() if match else ""


def _to_bullets(block: str) -> List[str]:
    items: List[str] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[\-\u2022]\s*", "", line)
        line = re.sub(r"^\d+\.\s*", "", line)
        items.append(line)
    return items


def parse_neurips_reviews(text: str) -> List[ParsedReview]:
    cleaned = clean_review_text(text)
    matches = list(
        re.finditer(r"Review\s+(\d+)\s+\(Score:\s*([^\)]+)\)\s*\n", cleaned)
    )
    reviews: List[ParsedReview] = []
    headings = ["Summary", "Strengths", "Weaknesses", "Questions for Rebuttal"]

    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
        body = cleaned[start:end].strip()
        summary = _extract_heading_block(body, "Summary", headings[1:])
        strengths = _to_bullets(_extract_heading_block(body, "Strengths", headings[2:]))
        weaknesses = _to_bullets(_extract_heading_block(body, "Weaknesses", headings[3:]))
        questions = _to_bullets(_extract_heading_block(body, "Questions for Rebuttal", []))
        reviews.append(
            ParsedReview(
                review_id=match.group(1),
                score=match.group(2).strip(),
                summary=summary,
                strengths=strengths,
                weaknesses=weaknesses,
                questions=questions,
            )
        )

    if not reviews:
        raise ValueError("Could not parse NeurIPS review sections from the review PDF text.")
    return reviews


def build_neurips_reviews_markdown(reviews: List[ParsedReview]) -> str:
    numeric_scores = []
    for review in reviews:
        match = re.match(r"(\d+)", review.score)
        if match:
            numeric_scores.append(int(match.group(1)))
    avg = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0

    lines = [
        "# NeurIPS Review Packet",
        "",
        "## Overall Score Summary",
        "",
        f"- Review count: {len(reviews)}",
        f"- Average score: {avg:.2f}",
        "- Raw scores: " + ", ".join(f"R{r.review_id}={r.score}" for r in reviews),
        "- Decision signal: mixed-to-positive external feedback, but with major requests for narrower claims and stronger evidence.",
        "",
    ]

    for review in reviews:
        lines.extend(
            [
                f"## Review {review.review_id}",
                "",
                f"**Score:** {review.score}",
                "",
                "### Summary",
                "",
                review.summary or "No summary extracted.",
                "",
                "### Strengths",
                "",
            ]
        )
        lines.extend(f"- {item}" for item in review.strengths)
        lines.extend(["", "### Weaknesses", ""])
        lines.extend(f"- {item}" for item in review.weaknesses)
        lines.extend(["", "### Questions For Rebuttal", ""])
        lines.extend(f"- {item}" for item in review.questions)
        lines.append("")

    lines.extend(
        [
            "## Consolidated Action Items",
            "",
            "- Either derive an explicit A6a/shared-frame error bound or demote the theorem framing so the limitation is front-and-center.",
            "- Retitle and reframe the paper unless the revision truly produces a regression-side solution characterization.",
            "- Keep H3 as a hypothesis by default and only upgrade it if direct transformer-side evidence is added.",
            "- Reconcile the block and weight-decay story with practical Muon usage in language-model training.",
            "- Tighten the B_crit discussion around momentum and anisotropic noise, including explicit limitations if a sharper derivation is unavailable.",
            "- Clarify the relationship to Fan/Gronich/Sato and other concurrent work so the regression-side novelty is scoped precisely.",
            "- Prefer layerwise transformer evidence, checkpoint reuse, and spectral measurements over hand-wavy scale claims.",
            "",
        ]
    )

    return "\n".join(lines).rstrip() + "\n"


def build_review_alignment_markdown(latest_final_tex: str) -> str:
    title_overclaim_open = "The Implicit Bias of the Muon Optimizer" in latest_final_tex
    strong_h3_claim_open = "These results explain why Muon benefits language models" in latest_final_tex
    mentions_a6a = "A6a" in latest_final_tex or "shared-frame" in latest_final_tex
    mentions_weight_decay = "weight decay" in latest_final_tex.lower()
    mentions_concurrent_work = "Gronich" in latest_final_tex or "max-margin" in latest_final_tex
    mentions_momentum = "momentum" in latest_final_tex.lower()

    def yes_no(value: bool) -> str:
        return "yes" if value else "no"

    lines = [
        "# Review Alignment",
        "",
        "The NeurIPS reviews were written for `muon_v4_iterate/muon_v4.pdf`, while v5 will revise the later completed v4 campaign output at `results/muon_v4_iterate/iterate_v4/paper_workspace/final_paper.pdf`.",
        "",
        "This mapping prevents the v5 iterate cycle from redundantly re-fixing issues that the later v4 draft already narrowed, while still treating unresolved reviewer asks as first-class requirements.",
        "",
        "| Reviewer Thread | Status | Latest v4 Evidence | v5 Requirement |",
        "| --- | --- | --- | --- |",
        f"| A6a / shared-frame weakness | {'already_partially_addressed in latest v4' if mentions_a6a else 'still open'} | shared-frame caveat present: {yes_no(mentions_a6a)} | Try to derive an explicit bound; otherwise demote the theorem framing further. |",
        f"| Title overclaim / missing regression regularizer | {'still open' if title_overclaim_open else 'already_partially_addressed in latest v4'} | old title still present: {yes_no(title_overclaim_open)} | Retitle unless the revision truly produces a converged-solution characterization. |",
        f"| Block-to-language-model bridging gap | {'still open' if strong_h3_claim_open else 'already_partially_addressed in latest v4'} | strong H3 payoff wording still present: {yes_no(strong_h3_claim_open)} | Keep H3 as a hypothesis unless new direct evidence upgrades it. |",
        f"| Weight decay tension | {'already_partially_addressed in latest v4' if mentions_weight_decay else 'still open'} | dedicated weight-decay text present: {yes_no(mentions_weight_decay)} | Reconcile ATSR degradation with practical Muon usage, not just warn practitioners. |",
        f"| B_crit anisotropy / momentum caveat | {'already_partially_addressed in latest v4' if mentions_momentum else 'still open'} | momentum caveat present: {yes_no(mentions_momentum)} | Add explicit limitation language or a sharper derivation. |",
        f"| Concurrent-work positioning | {'already_partially_addressed in latest v4' if mentions_concurrent_work else 'still open'} | classification-side concurrent work cited: {yes_no(mentions_concurrent_work)} | Tighten the novelty claim around the regression-side story. |",
        "| Transformer-scale evidence gap | requires new evidence | current v4 relies on NanoGPT-scale and argumentation | Prefer layerwise spectral measurements or checkpoint-based evidence; if unavailable, narrow the paper. |",
        "",
    ]
    return "\n".join(lines)


def build_human_directive_markdown() -> str:
    return (
        "# Human Directive\n\n"
        "These are binding project-level choices for the v5 iterate campaign.\n\n"
        "- Baseline: the canonical prior paper is the latest completed v4 campaign output, not the earlier reviewed submission PDF.\n"
        "- Revision scope: this is a full revision, not a cosmetic rebuttal pass.\n"
        "- H1/H2/H3 framing: H1 remains a supported mechanism/result; H2 remains supported but must be qualified by momentum/noise caveats; H3 stays a hypothesis unless the campaign produces new direct evidence strong enough to upgrade it.\n"
        "- Title discipline: unless the run discovers a real regression-side solution characterization, the paper must stop claiming a full implicit-bias characterization in the title and opening thesis.\n"
        "- Decision rule: if a stronger formal result is achievable, include it; if not, narrow the claim; never preserve an overclaim just because it existed in v4.\n"
        "- Evidence bar: prefer layerwise spectral measurements, transformer checkpoint reuse, or other direct evidence over vague scale claims. If medium-scale evidence is not credible, narrow the paper rather than hand-wave.\n"
    )


def extract_internal_review_items(verdict_payload: dict, report_text: str) -> tuple[list[str], list[str], list[str]]:
    must_fix = []
    for item in verdict_payload.get("must_fix_actions", []):
        action = item.get("action", "").strip()
        if action:
            must_fix.append(action)

    nice_to_fix = []
    for item in verdict_payload.get("nice_to_fix_actions", []):
        action = item.get("action", "").strip()
        if action:
            nice_to_fix.append(action)

    notable_findings = []
    regression_issues = verdict_payload.get("iteration_assessment", {}).get("regression_issues", [])
    notable_findings.extend(issue.strip() for issue in regression_issues if issue.strip())
    if "question marks" in report_text.lower():
        notable_findings.append("Review report flagged unresolved citation/question-mark placeholders.")
    if "no visual figures" in report_text.lower() or "missing visuals" in report_text.lower():
        notable_findings.append("Review report flagged missing or dropped visual figures in the compiled PDF.")

    return must_fix, nice_to_fix, notable_findings


def build_internal_v4_checks_markdown(verdict_payload: dict, report_text: str) -> str:
    must_fix, nice_to_fix, notable_findings = extract_internal_review_items(verdict_payload, report_text)
    lines = [
        "# Internal v4 Checks",
        "",
        "This summary captures the internal review-stage issues carried out during the completed v4 campaign. These checks are weaker than the external NeurIPS reviews, but they are still useful guardrails for v5.",
        "",
        f"- Internal review score: {verdict_payload.get('overall_score', 'unknown')}",
        f"- Hard blockers: {len(verdict_payload.get('hard_blockers', []))}",
        "",
        "## Notable Findings",
        "",
    ]
    if notable_findings:
        lines.extend(f"- {item}" for item in notable_findings)
    else:
        lines.append("- No additional notable findings were extracted.")

    lines.extend(["", "## Must-Fix Items", ""])
    if must_fix:
        lines.extend(f"- {item}" for item in must_fix)
    else:
        lines.append("- No must-fix items recorded.")

    lines.extend(["", "## Nice-To-Fix Items", ""])
    if nice_to_fix:
        lines.extend(f"- {item}" for item in nice_to_fix)
    else:
        lines.append("- No nice-to-fix items recorded.")

    lines.extend(
        [
            "",
            "## v5 Interpretation",
            "",
            "- Treat the figure/layout issue as unresolved until the v5 PDF is visually inspected or otherwise validated as containing the main figures.",
            "- Treat citation/question-mark cleanup as a live editorial requirement, even if the exact count changes during revision.",
            "- Use these internal checks as supporting guardrails, not as a substitute for the external NeurIPS review packet.",
            "",
        ]
    )
    return "\n".join(lines)


def file_metadata(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256_file(path),
    }


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def prepare_iterate_seed(
    *,
    latest_paper_pdf: Path,
    latest_paper_tex: Path,
    reviews_pdf: Path,
    reviewed_submission_pdf: Path,
    internal_verdict_json: Path,
    internal_report_tex: Path,
    output_dir: Path,
    supplemental_markdown: Iterable[Path],
    force: bool,
) -> Path:
    for path in [
        latest_paper_pdf,
        latest_paper_tex,
        reviews_pdf,
        reviewed_submission_pdf,
        internal_verdict_json,
        internal_report_tex,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(latest_paper_pdf, output_dir / "paper.pdf")

    review_text = extract_pdf_text(reviews_pdf)
    reviews = parse_neurips_reviews(review_text)
    write_text(output_dir / "neurips_reviews.md", build_neurips_reviews_markdown(reviews))

    latest_final_tex = latest_paper_tex.read_text(encoding="utf-8", errors="replace")
    write_text(output_dir / "review_alignment.md", build_review_alignment_markdown(latest_final_tex))
    write_text(output_dir / "human_directive.md", build_human_directive_markdown())

    verdict_payload = json.loads(internal_verdict_json.read_text(encoding="utf-8"))
    report_text = internal_report_tex.read_text(encoding="utf-8", errors="replace")
    write_text(
        output_dir / "internal_v4_checks.md",
        build_internal_v4_checks_markdown(verdict_payload, report_text),
    )

    copied_supplemental = []
    for source in supplemental_markdown:
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(source)
        if source.suffix.lower() != ".md":
            raise ValueError(f"Supplemental markdown inputs must end in .md: {source}")
        destination = output_dir / source.name
        if destination.exists():
            raise ValueError(f"Supplemental markdown filename collides with generated file: {source.name}")
        shutil.copy2(source, destination)
        copied_supplemental.append(destination)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed_dir": str(output_dir),
        "canonical_prior_paper": file_metadata(latest_paper_pdf),
        "canonical_prior_paper_tex": file_metadata(latest_paper_tex),
        "review_packet_pdf": file_metadata(reviews_pdf),
        "review_target_submission_pdf": file_metadata(reviewed_submission_pdf),
        "internal_review_verdict_json": file_metadata(internal_verdict_json),
        "internal_review_report_tex": file_metadata(internal_report_tex),
        "supplemental_markdown": [file_metadata(path) for path in copied_supplemental],
        "generated_files": {
            name: file_metadata(output_dir / name)
            for name in [
                "paper.pdf",
                "neurips_reviews.md",
                "review_alignment.md",
                "human_directive.md",
                "internal_v4_checks.md",
            ]
        },
    }
    (output_dir / "seed_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latest-paper-pdf", type=Path, default=DEFAULT_LATEST_PAPER_PDF)
    parser.add_argument("--latest-paper-tex", type=Path, default=DEFAULT_LATEST_PAPER_TEX)
    parser.add_argument("--reviews-pdf", type=Path, default=DEFAULT_REVIEWS_PDF)
    parser.add_argument("--reviewed-submission-pdf", type=Path, default=DEFAULT_REVIEWED_SUBMISSION_PDF)
    parser.add_argument("--internal-verdict-json", type=Path, default=DEFAULT_INTERNAL_VERDICT_JSON)
    parser.add_argument("--internal-report-tex", type=Path, default=DEFAULT_INTERNAL_REPORT_TEX)
    parser.add_argument("--supplemental-markdown", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Overwrite the output directory if it exists.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = prepare_iterate_seed(
        latest_paper_pdf=args.latest_paper_pdf,
        latest_paper_tex=args.latest_paper_tex,
        reviews_pdf=args.reviews_pdf,
        reviewed_submission_pdf=args.reviewed_submission_pdf,
        internal_verdict_json=args.internal_verdict_json,
        internal_report_tex=args.internal_report_tex,
        output_dir=args.output_dir,
        supplemental_markdown=args.supplemental_markdown,
        force=args.force,
    )
    print(f"Prepared iterate seed: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
