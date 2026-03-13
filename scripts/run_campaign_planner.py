#!/usr/bin/env python3
"""
Campaign planner — standalone script that runs the planning counsel.

Launched by the campaign runner as a subprocess (instead of launch_multiagent.py)
for the planning_counsel stage. Reads discovery phase output from the workspace,
runs a multi-model debate to determine the campaign structure, and writes
campaign_plan.json and campaign_plan_review.md.

Usage:
    python scripts/run_campaign_planner.py --workspace /path/to/planning_counsel/workspace
    python scripts/run_campaign_planner.py --workspace /path --planning-config '{"max_stages": 4}'
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure consortium is importable from repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from consortium.campaign.planner import (
    format_plan_for_review,
    run_campaign_planning_counsel,
)
from consortium.campaign.spec import PlanningConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run campaign planning counsel.")
    p.add_argument(
        "--workspace",
        required=True,
        help="Path to the planning_counsel stage workspace "
             "(should contain discovery phase artifacts via context_from).",
    )
    p.add_argument(
        "--planning-config",
        default="{}",
        help="JSON string or path to file with PlanningConfig overrides.",
    )
    p.add_argument(
        "--max-debate-rounds",
        type=int,
        default=3,
        help="Number of debate rounds (default: 3).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-model timeout in seconds (default: 600).",
    )
    return p.parse_args()


def _load_planning_config(raw: str) -> PlanningConfig:
    """Load PlanningConfig from a JSON string or file path."""
    if os.path.isfile(raw):
        with open(raw) as f:
            d = json.load(f)
    else:
        d = json.loads(raw)
    return PlanningConfig.from_dict(d)


def _read_file(path: str) -> str:
    """Read a file and return its contents, or empty string if missing."""
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return ""


def main() -> int:
    args = parse_args()
    workspace = os.path.abspath(args.workspace)

    print(f"[planner] Workspace: {workspace}")

    # Load planning config
    planning_config = _load_planning_config(args.planning_config)

    # Read discovery phase outputs from the workspace
    # These were copied in via context_from: discovery_plan
    research_plan_text = ""
    for candidate in [
        os.path.join(workspace, "paper_workspace", "research_plan.tex"),
        os.path.join(workspace, "paper_workspace", "research_plan.md"),
        os.path.join(workspace, "research_plan.tex"),
    ]:
        content = _read_file(candidate)
        if content:
            research_plan_text = content
            print(f"[planner] Loaded research plan from: {candidate}")
            break

    if not research_plan_text:
        print("[planner] ERROR: No research plan found in workspace.")
        return 1

    # Load track decomposition (optional but valuable)
    track_decomposition = None
    for candidate in [
        os.path.join(workspace, "paper_workspace", "track_decomposition.json"),
        os.path.join(workspace, "track_decomposition.json"),
    ]:
        content = _read_file(candidate)
        if content:
            try:
                track_decomposition = json.loads(content)
                print(f"[planner] Loaded track decomposition from: {candidate}")
            except json.JSONDecodeError:
                print(f"[planner] Warning: invalid JSON in {candidate}")
            break

    # Also try to load research_plan_tasks.json for extra context
    tasks_text = ""
    for candidate in [
        os.path.join(workspace, "paper_workspace", "research_plan_tasks.json"),
        os.path.join(workspace, "research_plan_tasks.json"),
    ]:
        content = _read_file(candidate)
        if content:
            tasks_text = content
            print(f"[planner] Loaded research tasks from: {candidate}")
            break

    # Append tasks to research plan if available
    if tasks_text:
        research_plan_text += (
            "\n\n## Research Tasks (from discovery phase)\n\n"
            + tasks_text
        )

    # Run the planning counsel
    print("[planner] Running planning counsel debate...")
    try:
        plan = run_campaign_planning_counsel(
            research_plan_text=research_plan_text,
            track_decomposition=track_decomposition,
            planning_config=planning_config,
            max_debate_rounds=args.max_debate_rounds,
            model_timeout_seconds=args.timeout,
        )
    except ValueError as e:
        print(f"[planner] ERROR: {e}")
        return 1

    # Write campaign_plan.json
    plan_path = os.path.join(workspace, "campaign_plan.json")
    plan_dict = {
        "stages": [
            {
                "id": s.id,
                "stage_type": s.stage_type,
                "description": s.description,
                "task_prompt": s.task_prompt,
                "depends_on": s.depends_on,
                "context_from": s.context_from,
                "research_questions": s.research_questions,
                "estimated_budget_usd": s.estimated_budget_usd,
            }
            for s in plan.stages
        ],
        "rationale": plan.rationale,
        "total_estimated_budget_usd": plan.total_estimated_budget_usd,
        "research_questions": plan.research_questions,
    }
    with open(plan_path, "w") as f:
        json.dump(plan_dict, f, indent=2)
    print(f"[planner] Written: {plan_path}")

    # Write human-readable review
    review_path = os.path.join(workspace, "campaign_plan_review.md")
    review_md = format_plan_for_review(plan)
    with open(review_path, "w") as f:
        f.write(review_md)
    print(f"[planner] Written: {review_path}")

    # Summary
    stage_types = {}
    for s in plan.stages:
        stage_types[s.stage_type] = stage_types.get(s.stage_type, 0) + 1
    type_summary = ", ".join(f"{v} {k}" for k, v in sorted(stage_types.items()))
    print(f"[planner] Campaign plan: {len(plan.stages)} stages ({type_summary})")
    print(f"[planner] Rationale: {plan.rationale[:200]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
