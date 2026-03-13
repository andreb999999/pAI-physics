"""
Planning counsel prompts — system prompt and task builders for dynamic campaign planning.

The planning counsel analyzes a research plan (from a completed discovery phase) and
outputs a structured campaign decomposition: how many theory/experiment stages are needed,
their dependency ordering, task prompts, and artifact requirements.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Stage type templates — default CLI args, artifacts, and memory dirs
# ---------------------------------------------------------------------------

STAGE_ARG_TEMPLATES: Dict[str, List[str]] = {
    "theory": [
        "--pipeline-mode", "full_research",
        "--enable-math-agents", "--enable-counsel",
        "--enable-tree-search",
    ],
    "experiment": [
        "--pipeline-mode", "full_research",
        "--enable-counsel",
    ],
    "paper": [
        "--pipeline-mode", "full_research",
        "--require-pdf", "--enable-counsel",
    ],
}

STAGE_ARTIFACT_TEMPLATES: Dict[str, dict] = {
    "theory": {
        "success_artifacts": {
            "required": ["math_workspace/claim_graph.json"],
            "optional": ["math_workspace/proofs/", "math_workspace/checks/"],
        },
        "memory_dirs": ["math_workspace/"],
    },
    "experiment": {
        "success_artifacts": {
            "required": ["experiment_results.json", "experiment_analysis.md"],
            "optional": ["experiment_workspace/"],
        },
        "memory_dirs": ["experiment_workspace/"],
    },
    "paper": {
        "success_artifacts": {
            "required": ["final_paper.tex", "final_paper.pdf"],
            "optional": ["paper_workspace/"],
        },
        "memory_dirs": [],
    },
}


# ---------------------------------------------------------------------------
# System prompt for the planning counsel
# ---------------------------------------------------------------------------

CAMPAIGN_PLANNING_SYSTEM_PROMPT = """\
You are a research campaign architect. Your task is to analyze a completed research plan \
and decompose it into a sequence of campaign stages that will be executed by an autonomous \
multi-agent research pipeline.

## Available Stage Types

Each stage runs a full 21-agent research pipeline. The stage type determines which \
capabilities are enabled:

### theory
- Runs the full pipeline with math agents enabled (formal proof, verification, claim graphs)
- Best for: proving theorems, deriving bounds, formal analysis, mathematical modeling
- Produces: claim_graph.json, formal proofs, verification reports

### experiment
- Runs the full pipeline with experiment agents enabled (design, execution, verification)
- Best for: empirical validation, benchmarks, ablation studies, data analysis
- Produces: experiment_results.json, experiment_analysis.md, plots, code

### paper
- Runs the full pipeline focused on paper writing with PDF generation
- Must be the FINAL stage — depends on all prior stages
- Produces: final_paper.tex, final_paper.pdf

## Design Principles

1. **Dependency-driven ordering**: If theory T informs experiment E, then E depends_on T. \
If experimental results should refine theory, create a second theory stage that depends on \
the experiment.

2. **Parallelism where possible**: Independent investigations (e.g., two unrelated \
theoretical questions) should NOT depend on each other — they can run simultaneously.

3. **Iterative refinement**: For complex research, use theory→experiment→theory cycles \
where each stage builds on prior results.

4. **Minimal stages**: Don't create more stages than necessary. A single theory stage can \
address multiple related theoretical questions. Only split into separate stages when \
questions are genuinely independent or when results from one stage must feed into another.

5. **Context flow**: Use context_from to specify which prior stages' artifacts should be \
copied into a stage's workspace. This enables downstream stages to build on upstream results.

## Output Format

You MUST output valid JSON matching this exact schema:

```json
{
  "stages": [
    {
      "id": "theory1",
      "stage_type": "theory",
      "description": "Brief description of what this stage investigates",
      "task_prompt": "Detailed task prompt that the research agents will follow...",
      "depends_on": [],
      "context_from": [],
      "research_questions": ["What specific questions does this stage address?"]
    }
  ],
  "rationale": "Explain why this decomposition was chosen",
  "research_questions": ["Top-level questions for the entire campaign"]
}
```

## Rules
- Stage IDs must be unique and use the format: {type}{number} (e.g., theory1, experiment1, theory2)
- The paper stage is ALWAYS the last stage and must depend on ALL other stages
- context_from entries must reference valid stage IDs
- depends_on entries must reference valid stage IDs
- The dependency graph must be acyclic (a DAG)
- Each stage's task_prompt should be detailed enough for agents to work autonomously \
(2-4 paragraphs minimum)
- task_prompt should describe the research objectives, methodology, expected outputs, \
and how this stage relates to the broader campaign
"""


# ---------------------------------------------------------------------------
# Task prompt builder
# ---------------------------------------------------------------------------

def build_planning_task(
    research_plan_text: str,
    track_decomposition: Optional[dict] = None,
    constraints: Optional[dict] = None,
) -> str:
    """Build the user-facing task prompt for the planning counsel.

    Args:
        research_plan_text: Full text of the research plan (from discovery phase).
        track_decomposition: The track_decomposition.json from discovery (if available).
        constraints: Optional constraints dict (max_stages, max_parallel, etc.).

    Returns:
        Formatted task prompt string.
    """
    parts = [
        "## Research Plan\n",
        "The following research plan was produced by a discovery phase that included "
        "ideation, literature review, and research planning:\n",
        "---\n",
        research_plan_text.strip(),
        "\n---\n",
    ]

    if track_decomposition:
        parts.append("\n## Track Decomposition\n")
        parts.append(
            "The research planner also produced the following decomposition of the "
            "research into theory and empirical questions:\n"
        )
        parts.append("```json\n")
        parts.append(json.dumps(track_decomposition, indent=2))
        parts.append("\n```\n")

    parts.append("\n## Your Task\n")
    parts.append(
        "Analyze this research plan and produce a campaign stage decomposition. "
        "Determine:\n"
        "1. How many theory stages and experiment stages are needed\n"
        "2. The dependency ordering between them (what can run in parallel, "
        "what must be sequential)\n"
        "3. Detailed task prompts for each stage\n"
        "4. Always include a final paper stage that depends on all other stages\n"
    )

    if constraints:
        parts.append("\n## Constraints\n")
        if constraints.get("max_stages"):
            parts.append(
                f"- Maximum {constraints['max_stages']} non-paper stages allowed\n"
            )
        if constraints.get("max_parallel"):
            parts.append(
                f"- Maximum {constraints['max_parallel']} stages may run simultaneously\n"
            )

    parts.append(
        "\nOutput your response as a single JSON object matching the schema "
        "described in the system prompt. Do not include any text outside the JSON."
    )

    return "".join(parts)
