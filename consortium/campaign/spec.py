"""
Campaign spec loader — parses and validates campaign.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class Stage:
    id: str
    task_file: str
    args: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    context_from: List[str] = field(default_factory=list)
    memory_dirs: List[str] = field(default_factory=list)
    # success_artifacts: {"required": [...], "optional": [...]}
    success_artifacts: dict = field(default_factory=lambda: {"required": [], "optional": []})

    @classmethod
    def from_dict(cls, d: dict) -> "Stage":
        depends_on = d.get("depends_on", [])
        if isinstance(depends_on, str):
            depends_on = [depends_on]

        context_from = d.get("context_from", [])
        if isinstance(context_from, str):
            context_from = [context_from]

        artifacts = d.get("success_artifacts", {})
        if isinstance(artifacts, list):
            # shorthand: just a list of required files
            artifacts = {"required": artifacts, "optional": []}
        else:
            artifacts.setdefault("required", [])
            artifacts.setdefault("optional", [])

        return cls(
            id=d["id"],
            task_file=d["task_file"],
            args=d.get("args", []),
            depends_on=depends_on,
            context_from=context_from,
            memory_dirs=d.get("memory_dirs", []),
            success_artifacts=artifacts,
        )


@dataclass
class NotificationConfig:
    slack_webhook: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    ntfy_topic: Optional[str] = None
    ntfy_server: Optional[str] = None  # defaults to https://ntfy.sh
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_from_number: Optional[str] = None
    sms_to_number: Optional[str] = None
    on_stage_complete: bool = True
    on_failure: bool = True
    on_heartbeat: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "NotificationConfig":
        return cls(
            ntfy_topic=_expand_env(d.get("ntfy_topic")),
            ntfy_server=_expand_env(d.get("ntfy_server")),
            slack_webhook=_expand_env(d.get("slack_webhook")),
            telegram_bot_token=_expand_env(d.get("telegram_bot_token")),
            telegram_chat_id=_expand_env(d.get("telegram_chat_id")),
            twilio_account_sid=_expand_env(d.get("twilio_account_sid")),
            twilio_auth_token=_expand_env(d.get("twilio_auth_token")),
            twilio_from_number=_expand_env(d.get("twilio_from_number")),
            sms_to_number=_expand_env(d.get("sms_to_number")),
            on_stage_complete=d.get("on_stage_complete", True),
            on_failure=d.get("on_failure", True),
            on_heartbeat=d.get("on_heartbeat", False),
        )


@dataclass
class RepairConfig:
    """Configuration for autonomous failure repair via Claude Code agents."""
    enabled: bool = False
    max_attempts: int = 2
    launcher: str = "local"             # "local" (blocking) or "slurm" (async)
    claude_binary: str = "auto"         # "auto" to search, or explicit path
    model: Optional[str] = "claude-opus-4-6"  # strongest model for repair
    effort: str = "max"                  # Claude Code effort level (low/medium/high/max)
    budget_usd: float = 10.0            # per-attempt budget cap
    timeout_seconds: int = 600           # hard timeout per repair attempt
    allowed_actions: List[str] = field(default_factory=lambda: [
        "edit_code",
        "fix_config",
        "generate_missing_artifacts",
        "install_dependencies",
    ])
    retry_delay_seconds: int = 10       # pause before retrying after repair
    # --- Two-phase plan-then-execute settings ---
    two_phase: bool = True              # enable plan→review→execute flow
    plan_model: Optional[str] = None    # model for planning (defaults to self.model)
    plan_effort: Optional[str] = None   # effort for planning (defaults to self.effort)
    plan_budget_usd: float = 5.0        # budget for the read-only planning phase
    plan_timeout_seconds: int = 300     # timeout for planning phase
    review_model: str = "claude-opus-4-6"  # model OpenClaw uses to judge the plan
    review_temperature: float = 0.2     # low temp for deterministic review
    min_review_score: int = 7           # plan must score >= this (1-10) to proceed

    @classmethod
    def from_dict(cls, d: dict) -> "RepairConfig":
        return cls(
            enabled=d.get("enabled", False),
            max_attempts=int(d.get("max_attempts", 2)),
            launcher=d.get("launcher", "local"),
            claude_binary=d.get("claude_binary", "auto"),
            model=d.get("model", "claude-opus-4-6"),
            effort=d.get("effort", "max"),
            budget_usd=float(d.get("budget_usd", 10.0)),
            timeout_seconds=int(d.get("timeout_seconds", 600)),
            allowed_actions=d.get("allowed_actions", [
                "edit_code", "fix_config",
                "generate_missing_artifacts", "install_dependencies",
            ]),
            retry_delay_seconds=int(d.get("retry_delay_seconds", 10)),
            two_phase=d.get("two_phase", True),
            plan_model=d.get("plan_model"),
            plan_effort=d.get("plan_effort"),
            plan_budget_usd=float(d.get("plan_budget_usd", 5.0)),
            plan_timeout_seconds=int(d.get("plan_timeout_seconds", 300)),
            review_model=d.get("review_model", "claude-opus-4-6"),
            review_temperature=float(d.get("review_temperature", 0.2)),
            min_review_score=int(d.get("min_review_score", 7)),
        )


@dataclass
class CampaignSpec:
    name: str
    workspace_root: str
    stages: List[Stage]
    notification: NotificationConfig
    repair: RepairConfig = field(default_factory=RepairConfig)
    heartbeat_interval_minutes: int = 30

    def stage(self, stage_id: str) -> Optional[Stage]:
        for s in self.stages:
            if s.id == stage_id:
                return s
        return None

    def stage_ids(self) -> List[str]:
        return [s.id for s in self.stages]


def load_spec(path: str) -> CampaignSpec:
    """Load and validate a campaign.yaml file. Raises ValueError on bad config."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    name = raw.get("name") or os.path.basename(os.path.dirname(os.path.abspath(path)))
    workspace_root = raw.get("workspace_root", "results/campaign")

    stages_raw = raw.get("stages", [])
    if not stages_raw:
        raise ValueError(f"campaign.yaml at {path} has no stages defined.")

    stages = [Stage.from_dict(s) for s in stages_raw]

    # Validate dependency references
    ids = {s.id for s in stages}
    for s in stages:
        for dep in s.depends_on:
            if dep not in ids:
                raise ValueError(f"Stage '{s.id}' depends_on unknown stage '{dep}'.")
        for ctx in s.context_from:
            if ctx not in ids:
                raise ValueError(f"Stage '{s.id}' context_from unknown stage '{ctx}'.")

    notification = NotificationConfig.from_dict(raw.get("notification", {}))
    repair = RepairConfig.from_dict(raw.get("repair", {}))

    return CampaignSpec(
        name=name,
        workspace_root=workspace_root,
        stages=stages,
        notification=notification,
        repair=repair,
        heartbeat_interval_minutes=int(raw.get("heartbeat_interval_minutes", 30)),
    )


def _expand_env(value: Optional[str]) -> Optional[str]:
    """Expand ${VAR_NAME} environment variable references in config strings."""
    if not value:
        return value
    if value.startswith("${") and value.endswith("}"):
        var = value[2:-1]
        return os.environ.get(var)
    return value
