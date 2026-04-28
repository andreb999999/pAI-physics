from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "docs" / "pAI - theoretical physics network.svg"


@dataclass(frozen=True)
class Box:
    name: str
    x: int
    y: int
    w: int
    h: int
    label: str
    subtitle: str = ""
    fill: str = "#eef3ff"
    stroke: str = "#284b8f"
    radius: int = 18
    dashed: bool = False
    text_color: str = "#0f172a"
    subtitle_color: str = "#334155"
    title_size: int = 16
    subtitle_size: int = 12
    font_weight: str = "700"


@dataclass(frozen=True)
class Arrow:
    start: tuple[int, int]
    end: tuple[int, int]
    label: str
    dashed: bool = False
    color: str = "#334155"


def _multiline_text(text: str, x: int, start_y: float, font_size: int, color: str, weight: str) -> str:
    lines = text.split("\n") if text else []
    line_height = font_size + 4
    parts: list[str] = []
    for idx, line in enumerate(lines):
        parts.append(
            f'<text x="{x}" y="{start_y + font_size + idx * line_height:.1f}" '
            f'fill="{color}" font-size="{font_size}" font-weight="{weight}" '
            'font-family="Segoe UI, Arial, sans-serif" text-anchor="middle">'
            f'{html.escape(line)}</text>'
        )
    return "\n".join(parts)


def render_box(box: Box) -> str:
    dash_attr = ' stroke-dasharray="10 8"' if box.dashed else ""
    title_lines = box.label.split("\n")
    subtitle_lines = box.subtitle.split("\n") if box.subtitle else []
    title_h = len(title_lines) * (box.title_size + 4)
    subtitle_h = len(subtitle_lines) * (box.subtitle_size + 4)
    gap = 6 if subtitle_lines else 0
    total_h = title_h + gap + subtitle_h
    top_y = box.y + (box.h - total_h) / 2

    return (
        f'<rect x="{box.x}" y="{box.y}" width="{box.w}" height="{box.h}" '
        f'rx="{box.radius}" ry="{box.radius}" fill="{box.fill}" stroke="{box.stroke}" '
        f'stroke-width="2.5"{dash_attr} />\n'
        + _multiline_text(
            box.label,
            box.x + box.w // 2,
            top_y,
            box.title_size,
            box.text_color,
            box.font_weight,
        )
        + (
            "\n"
            + _multiline_text(
                box.subtitle,
                box.x + box.w // 2,
                top_y + title_h + gap,
                box.subtitle_size,
                box.subtitle_color,
                "500",
            )
            if subtitle_lines
            else ""
        )
    )


def render_arrow(arrow: Arrow) -> str:
    dash_attr = ' stroke-dasharray="8 6"' if arrow.dashed else ""
    mid_x = (arrow.start[0] + arrow.end[0]) / 2
    mid_y = (arrow.start[1] + arrow.end[1]) / 2
    label_svg = (
        f'<text x="{mid_x:.1f}" y="{mid_y - 10:.1f}" fill="{arrow.color}" font-size="11" '
        'font-family="Segoe UI, Arial, sans-serif" text-anchor="middle">'
        f'{html.escape(arrow.label)}</text>\n'
    )
    return (
        label_svg
        + f'<line x1="{arrow.start[0]}" y1="{arrow.start[1]}" x2="{arrow.end[0]}" y2="{arrow.end[1]}" '
        f'stroke="{arrow.color}" stroke-width="3" marker-end="url(#arrowhead)"{dash_attr} />'
    )


def build_svg() -> str:
    boxes = [
        Box("prompt", 560, 30, 280, 90, "Research prompt", "User-provided task or question", fill="#1d4ed8", stroke="#1e3a8a", text_color="#ffffff", subtitle_color="#dbeafe", title_size=20, subtitle_size=12),
        Box("persona", 525, 145, 350, 108, "persona_council", "Generates viewpoints and challenges assumptions", fill="#dbeafe"),
        Box("lit", 525, 285, 350, 108, "literature_review_agent", "Collects prior work and constraints from references", fill="#dbeafe"),
        Box("lit_gate", 525, 425, 350, 108, "lit_review_gate", "Checks if evidence is feasible and sufficient to proceed", fill="#e2e8f0"),
        Box("brainstorm", 525, 565, 350, 108, "brainstorm_agent", "Proposes candidate directions and hypotheses", fill="#dbeafe"),
        Box("goals_entry", 525, 705, 350, 108, "formalize_goals_entry", "Converts ideas into structured goal handoff", fill="#dcfce7", stroke="#166534"),
        Box("goals", 525, 845, 350, 108, "formalize_goals_agent", "Defines explicit objectives and success criteria", fill="#dcfce7", stroke="#166534"),
        Box("plan", 525, 985, 350, 108, "research_plan_writeup_agent", "Writes the concrete execution plan", fill="#dcfce7", stroke="#166534"),
        Box("track_gate", 525, 1125, 350, 108, "track_decomposition_gate", "Validates decomposition into executable tracks", fill="#f8fafc"),
        Box("milestone", 525, 1265, 350, 108, "milestone_goals", "Creates stage milestones and dependencies", fill="#f8fafc"),
        Box("router", 525, 1405, 350, 108, "track_router", "Dispatches work to theory and experiment branches", fill="#f8fafc"),
        Box("merge", 525, 2225, 350, 108, "track_merge", "Combines branch outputs into unified state", fill="#f8fafc"),
        Box("verify", 525, 2365, 350, 108, "verify_completion", "Routes to synthesize, rework, or rethink", fill="#f8fafc"),
        Box("results", 525, 2505, 350, 108, "formalize_results_agent", "Synthesizes accepted findings and claims", fill="#fde68a", stroke="#92400e"),
        Box("duality", 525, 2645, 350, 108, "duality_check", "Checks theory and empirical consistency", fill="#fef3c7", stroke="#92400e"),
        Box("duality_gate", 525, 2785, 350, 108, "duality_gate", "Decides pass or follow-up investigation", fill="#f8fafc"),
        Box("resource", 525, 2925, 350, 108, "resource_preparation_agent", "Prepares citations, figures, and references", fill="#ede9fe", stroke="#6d28d9"),
        Box("writeup", 525, 3065, 350, 108, "writeup_agent", "Drafts manuscript sections and integrates artifacts", fill="#ede9fe", stroke="#6d28d9"),
        Box("proofread", 525, 3205, 350, 108, "proofreading_agent", "Improves clarity and language quality", fill="#ede9fe", stroke="#6d28d9"),
        Box("reviewer", 525, 3345, 350, 108, "reviewer_agent", "Scores output quality and requests revision if needed", fill="#ede9fe", stroke="#6d28d9"),
        Box("validation", 525, 3485, 350, 108, "validation_gate", "Runs final artifact and quality validations", fill="#f8fafc"),
        Box("final", 525, 3625, 350, 116, "Final artifacts", "run_summary.json, budget_state.json, paper_workspace/", fill="#14532d", stroke="#14532d", text_color="#ffffff", subtitle_color="#dcfce7", title_size=18, subtitle_size=12),
        Box("theory_group", 980, 1490, 380, 780, "Theory track (optional)", "Formal reasoning and verification branch", fill="#fafafa", stroke="#94a3b8", dashed=True, title_size=18, subtitle_size=12),
        Box("math_lit", 1035, 1565, 270, 110, "math_literature_agent", "Collects theorem and proof background", fill="#f8fafc", stroke="#64748b", title_size=14, subtitle_size=11),
        Box("math_prop", 1035, 1695, 270, 110, "math_proposer_agent", "Proposes formal claims and structure", fill="#f8fafc", stroke="#64748b", title_size=14, subtitle_size=11),
        Box("math_prov", 1035, 1825, 270, 110, "math_prover_agent", "Drafts proofs and key derivations", fill="#f8fafc", stroke="#64748b", title_size=14, subtitle_size=11),
        Box("math_rig", 1035, 1955, 270, 110, "math_rigorous_verifier_agent", "Checks symbolic rigor and logic", fill="#f8fafc", stroke="#64748b", title_size=13, subtitle_size=11),
        Box("math_emp", 1035, 2085, 270, 110, "math_empirical_verifier_agent", "Runs numeric or computational checks", fill="#f8fafc", stroke="#64748b", title_size=13, subtitle_size=11),
        Box("proof_tx", 1035, 2215, 270, 110, "proof_transcription_agent", "Converts proofs into paper artifacts", fill="#f8fafc", stroke="#64748b", title_size=13, subtitle_size=11),
        Box("exp_group", 40, 1490, 380, 650, "Experiment track", "Empirical design, run, and verification branch", fill="#fafafa", stroke="#94a3b8", dashed=True, title_size=18, subtitle_size=12),
        Box("exp_lit", 95, 1565, 270, 110, "experiment_literature_agent", "Finds baselines and empirical precedents", fill="#fff7ed", stroke="#c2410c", title_size=13, subtitle_size=11),
        Box("exp_design", 95, 1695, 270, 110, "experiment_design_agent", "Specifies setup, controls, and metrics", fill="#fff7ed", stroke="#c2410c", title_size=13, subtitle_size=11),
        Box("exp_run", 95, 1825, 270, 110, "experimentation_agent", "Executes planned experiments", fill="#fff7ed", stroke="#c2410c", title_size=14, subtitle_size=11),
        Box("exp_verify", 95, 1955, 270, 110, "experiment_verification_agent", "Validates outputs and result quality", fill="#fff7ed", stroke="#c2410c", title_size=12, subtitle_size=11),
        Box("exp_tx", 95, 2085, 270, 110, "experiment_transcription_agent", "Packages findings into structured artifacts", fill="#fff7ed", stroke="#c2410c", title_size=12, subtitle_size=11),
        Box("followup", 980, 2790, 380, 108, "followup_lit_review", "Performs targeted follow-up research", fill="#fee2e2", stroke="#b91c1c"),
    ]

    box_map = {box.name: box for box in boxes}

    def top(name: str) -> tuple[int, int]:
        box = box_map[name]
        return (box.x + box.w // 2, box.y)

    def bottom(name: str) -> tuple[int, int]:
        box = box_map[name]
        return (box.x + box.w // 2, box.y + box.h)

    def left(name: str) -> tuple[int, int]:
        box = box_map[name]
        return (box.x, box.y + box.h // 2)

    def right(name: str) -> tuple[int, int]:
        box = box_map[name]
        return (box.x + box.w, box.y + box.h // 2)

    arrows = [
        Arrow(bottom("prompt"), top("persona"), label="run starts"),
        Arrow(bottom("persona"), top("lit"), label="viewpoints synthesized"),
        Arrow(bottom("lit"), top("lit_gate"), label="literature pass complete"),
        Arrow(bottom("lit_gate"), top("brainstorm"), label="if feasible"),
        Arrow(left("lit_gate"), (420, 478), label="if infeasible"),
        Arrow((420, 478), left("persona"), label="reframe objective", dashed=True),
        Arrow(bottom("brainstorm"), top("goals_entry"), label="candidate plans ready"),
        Arrow(bottom("goals_entry"), top("goals"), label="handoff normalized"),
        Arrow(bottom("goals"), top("plan"), label="goals accepted"),
        Arrow(bottom("plan"), top("track_gate"), label="plan drafted"),
        Arrow(bottom("track_gate"), top("milestone"), label="decomposition valid"),
        Arrow(bottom("milestone"), top("router"), label="milestones emitted"),
        Arrow(left("router"), top("exp_lit"), label="route: experiment"),
        Arrow(right("router"), top("math_lit"), label="route: theory"),
        Arrow(bottom("math_lit"), top("math_prop"), label="proof context ready"),
        Arrow(bottom("math_prop"), top("math_prov"), label="claims proposed"),
        Arrow(bottom("math_prov"), top("math_rig"), label="proof draft complete"),
        Arrow(bottom("math_rig"), top("math_emp"), label="rigor checks pass"),
        Arrow(bottom("math_emp"), top("proof_tx"), label="numeric checks pass"),
        Arrow(bottom("exp_lit"), top("exp_design"), label="baselines identified"),
        Arrow(bottom("exp_design"), top("exp_run"), label="design approved"),
        Arrow(bottom("exp_run"), top("exp_verify"), label="runs complete"),
        Arrow(bottom("exp_verify"), top("exp_tx"), label="results validated"),
        Arrow(right("exp_tx"), left("merge"), label="ingest experiment artifacts"),
        Arrow(left("proof_tx"), right("merge"), label="ingest theory artifacts"),
        Arrow(bottom("merge"), top("verify"), label="both tracks merged"),
        Arrow(bottom("verify"), top("results"), label="if complete"),
        Arrow(left("verify"), (430, 2418), label="if fundamental rethink"),
        Arrow((430, 2418), left("brainstorm"), label="return to ideation", dashed=True),
        Arrow(right("verify"), (970, 2418), label="if partial rework"),
        Arrow((970, 2418), right("goals"), label="refine goals", dashed=True),
        Arrow(bottom("results"), top("duality"), label="results formalized"),
        Arrow(bottom("duality"), top("duality_gate"), label="duality checked"),
        Arrow(bottom("duality_gate"), top("resource"), label="if pass"),
        Arrow(right("duality_gate"), left("followup"), label="if follow-up needed"),
        Arrow(bottom("resource"), top("writeup"), label="assets assembled"),
        Arrow(bottom("writeup"), top("proofread"), label="draft compiled"),
        Arrow(bottom("proofread"), top("reviewer"), label="language polished"),
        Arrow(bottom("reviewer"), top("validation"), label="review complete"),
        Arrow(bottom("validation"), top("final"), label="if all checks pass"),
        Arrow(right("validation"), (980, 3538), label="if revisions required"),
        Arrow((1170, 3538), (1170, 2905), label="revision path", dashed=True),
        Arrow((1170, 2905), top("followup"), label="enter follow-up review", dashed=True),
        Arrow(bottom("followup"), (1170, 2945), label="follow-up complete"),
        Arrow((1170, 2945), (1170, 618), label="loop to ideation", dashed=True),
        Arrow((1170, 618), right("brainstorm"), label="resume at brainstorm", dashed=True),
    ]

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="3780" viewBox="0 0 1400 3780">',
        '<defs>',
        '<marker id="arrowhead" markerWidth="12" markerHeight="8" refX="10" refY="4" orient="auto">',
        '<polygon points="0 0, 12 4, 0 8" fill="#334155" />',
        '</marker>',
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">',
        '<feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#94a3b8" flood-opacity="0.25"/>',
        '</filter>',
        '</defs>',
        '<rect width="1400" height="3780" fill="#f8fafc" />',
        '<text x="700" y="24" fill="#0f172a" font-size="28" font-weight="700" font-family="Segoe UI, Arial, sans-serif" text-anchor="middle">pAI - theoretical physics network</text>',
    ]

    for box in boxes:
        parts.append(f'<g filter="url(#shadow)">{render_box(box)}</g>')
    for arrow in arrows:
        parts.append(render_arrow(arrow))
    parts.append('</svg>')
    return "\n".join(parts)


def main() -> None:
    OUTPUT_PATH.write_text(build_svg(), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()