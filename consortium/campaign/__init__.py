"""
Campaign manager — autonomous multi-stage research campaign orchestration.

A campaign is a sequence of pipeline runs (e.g. theory → experiments → paper)
where each stage's artifacts feed the next. This package provides:

  spec.py           — load/validate campaign.yaml
  status.py         — campaign_status.json read/write + artifact presence checking
  runner.py         — subprocess launcher with cross-run context enrichment
  memory.py         — post-stage artifact distillation into markdown summaries
  notify.py         — Slack/Telegram/stdout notification dispatch
  repair_agent.py   — autonomous failure repair via Claude Code agents
  planner.py        — dynamic campaign planning via multi-model counsel
  planner_prompt.py — planning counsel system prompts and templates

Entry point for external orchestrators (e.g. OpenClaw):
  scripts/campaign_heartbeat.py
"""

from .spec import CampaignSpec, PlanningConfig, RepairConfig, Stage, load_spec
from .status import CampaignStatus, read_status, write_status, check_stage_artifacts, is_pid_alive
from .runner import launch_stage, build_task_prompt
from .memory import distill_stage_memory
from .notify import notify
from .repair_agent import attempt_repair, submit_slurm_repair, poll_slurm_repair, RepairResult
from .budget_manager import (
    CampaignBudgetManager,
    DEGRADATION_PROFILES,
    apply_degradation_profile,
)
from .planner import (
    CampaignPlan,
    PlannedStage,
    format_plan_for_review,
    generate_task_files,
    load_campaign_plan,
    plan_to_stages,
    run_campaign_planning_counsel,
    validate_campaign_plan,
)

__all__ = [
    "CampaignSpec", "PlanningConfig", "RepairConfig", "Stage", "load_spec",
    "CampaignStatus", "read_status", "write_status",
    "check_stage_artifacts", "is_pid_alive",
    "launch_stage", "build_task_prompt",
    "distill_stage_memory",
    "notify",
    "attempt_repair", "submit_slurm_repair", "poll_slurm_repair", "RepairResult",
    "CampaignBudgetManager",
    "DEGRADATION_PROFILES",
    "apply_degradation_profile",
    "CampaignPlan", "PlannedStage",
    "format_plan_for_review", "generate_task_files",
    "load_campaign_plan", "plan_to_stages",
    "run_campaign_planning_counsel", "validate_campaign_plan",
]
