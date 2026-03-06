"""
Campaign spec loader — parses and validates campaign.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

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
    on_stage_complete: bool = True
    on_failure: bool = True
    on_heartbeat: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "NotificationConfig":
        return cls(
            slack_webhook=_expand_env(d.get("slack_webhook")),
            telegram_bot_token=_expand_env(d.get("telegram_bot_token")),
            telegram_chat_id=_expand_env(d.get("telegram_chat_id")),
            on_stage_complete=d.get("on_stage_complete", True),
            on_failure=d.get("on_failure", True),
            on_heartbeat=d.get("on_heartbeat", False),
        )


@dataclass
class CampaignSpec:
    name: str
    workspace_root: str
    stages: List[Stage]
    notification: NotificationConfig
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

    return CampaignSpec(
        name=name,
        workspace_root=workspace_root,
        stages=stages,
        notification=notification,
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
