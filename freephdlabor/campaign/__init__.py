"""
Campaign manager — autonomous multi-stage research campaign orchestration.

A campaign is a sequence of pipeline runs (e.g. theory → experiments → paper)
where each stage's artifacts feed the next. This package provides:

  spec.py    — load/validate campaign.yaml
  status.py  — campaign_status.json read/write + artifact presence checking
  runner.py  — subprocess launcher with cross-run context enrichment
  memory.py  — post-stage artifact distillation into markdown summaries
  notify.py  — Slack/Telegram/stdout notification dispatch

Entry point for external orchestrators (e.g. OpenClaw):
  scripts/campaign_heartbeat.py
"""

from .spec import CampaignSpec, Stage, load_spec
from .status import CampaignStatus, read_status, write_status, check_stage_artifacts, is_pid_alive
from .runner import launch_stage, build_task_prompt
from .memory import distill_stage_memory
from .notify import notify

__all__ = [
    "CampaignSpec", "Stage", "load_spec",
    "CampaignStatus", "read_status", "write_status",
    "check_stage_artifacts", "is_pid_alive",
    "launch_stage", "build_task_prompt",
    "distill_stage_memory",
    "notify",
]
