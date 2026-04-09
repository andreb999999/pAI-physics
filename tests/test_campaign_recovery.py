"""Regression tests for the Muon campaign recovery path."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from consortium.campaign.budget_manager import CampaignBudgetManager
from consortium.campaign.runner import launch_stage, launch_stage_slurm
from consortium.campaign.spec import Stage, load_spec
from consortium.campaign.status import CampaignStatus


def test_proofreading_and_reviewer_tools_cannot_generate_replacement_papers(tmp_path):
    from consortium.agents.proofreading_agent import get_tools as proofreading_tools
    from consortium.agents.reviewer_agent import get_tools as reviewer_tools

    proofreading_names = {tool.name for tool in proofreading_tools(str(tmp_path), "claude-opus-4-6")}
    reviewer_names = {tool.name for tool in reviewer_tools(str(tmp_path), "claude-opus-4-6")}

    assert "latex_generator" not in proofreading_names
    assert "delete_file_or_folder" not in proofreading_names
    assert "latex_generator" not in reviewer_names
    assert "delete_file_or_folder" not in reviewer_names


def test_counsel_creation_is_blocked_when_disabled(monkeypatch, tmp_path):
    from consortium.counsel import create_counsel_node

    monkeypatch.setenv("CONSORTIUM_COUNSEL_DISABLED", "1")

    try:
        create_counsel_node(
            system_prompt="test",
            tools=[],
            agent_name="proofreading_agent",
            workspace_dir=str(tmp_path),
            counsel_models=[object()],
        )
    except RuntimeError as exc:
        assert "while counsel is disabled" in str(exc)
    else:
        raise AssertionError("create_counsel_node should fail when counsel is disabled")


def test_counsel_sandbox_skips_checkpoint_sidecars(tmp_path):
    from consortium.counsel import _populate_sandbox

    workspace = tmp_path / "workspace"
    sandbox = tmp_path / "sandbox"
    workspace.mkdir()
    (workspace / "checkpoints.db").write_text("base")
    (workspace / "checkpoints.db-wal").write_text("wal")
    (workspace / "checkpoints.db-shm").write_text("shm")
    (workspace / "paper_workspace").mkdir()
    (workspace / "paper_workspace" / "brainstorm.md").write_text("keep me")

    _populate_sandbox(str(workspace), str(sandbox))

    assert not (sandbox / "checkpoints.db").exists()
    assert not (sandbox / "checkpoints.db-wal").exists()
    assert not (sandbox / "checkpoints.db-shm").exists()
    assert (sandbox / "paper_workspace" / "brainstorm.md").exists()


def test_tree_search_branch_skips_checkpoint_sidecars(tmp_path):
    from consortium.tree_search.workspace_fork import populate_branch

    workspace = tmp_path / "workspace"
    branch = tmp_path / "branch"
    workspace.mkdir()
    (workspace / "checkpoints.db").write_text("base")
    (workspace / "checkpoints.db-wal").write_text("wal")
    (workspace / "checkpoints.db-shm").write_text("shm")
    (workspace / "notes.txt").write_text("keep me")

    populate_branch(str(workspace), str(branch))

    assert not (branch / "checkpoints.db").exists()
    assert not (branch / "checkpoints.db-wal").exists()
    assert not (branch / "checkpoints.db-shm").exists()
    assert (branch / "notes.txt").exists()


def test_budget_manager_separates_current_and_archived_attempt_spend(tmp_path):
    campaign_dir = tmp_path / "campaign"
    current_ws = campaign_dir / "iterate_v4"
    archived_ws = campaign_dir / "archived_attempts" / "iterate_v4_failed_run_4"
    current_ws.mkdir(parents=True)
    archived_ws.mkdir(parents=True)

    (current_ws / "budget_state.json").write_text(json.dumps({"total_usd": 12.5}))
    (archived_ws / "budget_state.json").write_text(json.dumps({"total_usd": 7.25}))
    (campaign_dir / "campaign_status.json").write_text(
        json.dumps(
            {
                "campaign_name": "Muon",
                "spec_file": "campaign_muon_v4_iterate.yaml",
                "stages": {
                    "iterate_v4": {
                        "status": "in_progress",
                        "workspace": str(current_ws),
                    }
                },
            }
        )
    )

    mgr = CampaignBudgetManager(str(campaign_dir), usd_limit=100.0)

    assert mgr.current_attempt_spent == 12.5
    assert mgr.archived_spent == 7.25
    assert mgr.campaign_lifetime_spent == 19.75
    summary = mgr.to_dict()
    assert summary["current_attempt_usd"] == 12.5
    assert summary["campaign_lifetime_usd"] == 19.75


def test_stage_status_tracks_attempt_and_logs():
    status = CampaignStatus({"campaign_name": "Muon", "spec_file": "spec.yaml", "stages": {}})
    status.mark_in_progress(
        "iterate_v4",
        "/tmp/workspace",
        12345,
        attempt_id=4,
        stdout_log="logs/iterate_v4_attempt_4_stdout.log",
        stderr_log="logs/iterate_v4_attempt_4_stderr.log",
    )

    assert status.stage_attempt_id("iterate_v4") == 4
    assert status.stage_stdout_log("iterate_v4") == "logs/iterate_v4_attempt_4_stdout.log"
    assert status.stage_stderr_log("iterate_v4") == "logs/iterate_v4_attempt_4_stderr.log"


def test_mark_pending_retry_preserves_previous_attempt_id():
    status = CampaignStatus(
        {
            "campaign_name": "Muon",
            "spec_file": "spec.yaml",
            "stages": {
                "iterate_v4": {
                    "status": "failed",
                    "attempt_id": 6,
                    "stdout_log": "logs/iterate_v4_attempt_6_stdout.log",
                    "stderr_log": "logs/iterate_v4_attempt_6_stderr.log",
                    "fail_reason": "boom",
                    "missing_artifacts": ["paper_workspace/brainstorm.json"],
                }
            },
        }
    )

    status.mark_pending_retry("iterate_v4")

    assert status.stage_status("iterate_v4") == "pending"
    assert status.stage_attempt_id("iterate_v4") == 6
    assert status.stage_stdout_log("iterate_v4") is None
    assert status.stage_stderr_log("iterate_v4") is None


def test_campaign_heartbeat_validators_reject_muon_stub_workspace(tmp_path):
    from scripts.campaign_heartbeat import _validate_artifact_content

    pw = tmp_path / "paper_workspace"
    pw.mkdir()
    (pw / "paper_contract.json").write_text(json.dumps({"required_terms": ["Muon", "H1", "H2", "H3"]}))
    (pw / "final_paper.tex").write_text(
        "\\documentclass{article}\n"
        "\\title{Research Paper Title}\n"
        "\\author{Author Names}\n"
        "TODO\n"
    )
    (pw / "final_paper.pdf").write_bytes(b"%PDF-1.4\nsmall")
    (pw / "review_verdict.json").write_text(
        json.dumps({"overall_score": 7, "hard_blockers": ["B1"], "must_fix_actions": []})
    )

    stage = Stage.from_dict(
        {
            "id": "iterate_v4",
            "task_file": "task.txt",
            "success_artifacts": {
                "required": [
                    "paper_workspace/final_paper.tex",
                    "paper_workspace/final_paper.pdf",
                    "paper_workspace/review_verdict.json",
                ]
            },
            "artifact_validators": {
                "paper_workspace/final_paper.tex": {
                    "must_contain": ["Muon", "H1", "H2", "H3"],
                    "must_not_contain": ["Research Paper Title", "Author Names", "TODO"],
                },
                "paper_workspace/final_paper.pdf": {"min_size_bytes": 20000},
                "paper_workspace/review_verdict.json": {
                    "json_required_keys": ["overall_score", "hard_blockers", "must_fix_actions"],
                    "json_min_numeric": {"overall_score": 8},
                    "json_max_list_length": {"hard_blockers": 0},
                },
            },
        }
    )

    errors = _validate_artifact_content(str(tmp_path), stage)

    assert any("missing required content 'Muon'" in err for err in errors)
    assert any("contains forbidden content 'Research Paper Title'" in err for err in errors)
    assert any("too small" in err for err in errors)
    assert any("below minimum" in err for err in errors)
    assert any("too long" in err for err in errors)


def test_recursive_workspace_signal_counts_as_activity(tmp_path):
    from scripts.campaign_cli import _check_stage_liveness

    campaign_dir = tmp_path / "campaign"
    workspace = campaign_dir / "iterate_v4"
    nested = workspace / "deep" / "nested"
    nested.mkdir(parents=True)
    (nested / ".progress_heartbeat").write_text(json.dumps({"ts": 9999999999}))

    status = CampaignStatus(
        {
            "campaign_name": "Muon",
            "spec_file": "campaign.yaml",
            "stages": {
                "iterate_v4": {
                    "status": "in_progress",
                    "workspace": str(workspace),
                    "pid": None,
                    "slurm_job_id": None,
                    "attempt_id": 1,
                    "stdout_log": None,
                    "stderr_log": None,
                }
            },
        }
    )

    result = _check_stage_liveness(status, "iterate_v4", str(campaign_dir))

    assert result["workspace_active"] is True
    assert result["overall"] in {"likely_alive", "unknown"}


def test_campaign_launch_detaches_stage_and_keeps_attempt_logs(tmp_path, monkeypatch):
    campaign_dir = tmp_path / "campaign"
    workspace_root = campaign_dir
    task_file = tmp_path / "task.txt"
    launcher = tmp_path / "launch_multiagent.py"
    task_file.write_text("revise the paper")
    launcher.write_text("print('placeholder launcher')\n")
    campaign_dir.mkdir()

    stage = Stage.from_dict(
        {
            "id": "iterate_v4",
            "task_file": str(task_file),
            "env": {"CONSORTIUM_SS_COOLDOWN_SEC": "120"},
            "args": [
                "--iterate-start-stage",
                "literature_review_agent",
                "--callback_port",
                "5601",
            ],
            "success_artifacts": {"required": []},
        }
    )
    spec = type(
        "Spec",
        (),
        {
            "workspace_root": str(workspace_root),
            "budget_usd": 0.0,
            "counsel_model_timeout_seconds": 222,
        },
    )()
    status = CampaignStatus(
        {
            "campaign_name": "Muon",
            "spec_file": "campaign.yaml",
            "stages": {"iterate_v4": {"attempt_id": 4}},
        }
    )

    captured: dict[str, object] = {}

    class DummyProc:
        pid = 43210

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return DummyProc()

    monkeypatch.setattr("consortium.campaign.runner.subprocess.Popen", fake_popen)

    proc = launch_stage(
        stage=stage,
        spec=spec,
        status=status,
        campaign_dir=str(campaign_dir),
        launcher=str(launcher),
    )

    assert proc.pid == 43210
    kwargs = captured["kwargs"]
    env = kwargs["env"]
    assert env["CONSORTIUM_LOG_TO_FILES"] == "0"
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["CONSORTIUM_SS_COOLDOWN_SEC"] == "120"
    assert env["COUNSEL_MODEL_TIMEOUT_SECONDS"] == "222"
    assert env["CONSORTIUM_LIT_RATE_STATE_DIR"].endswith("iterate_v4/.lit_rate_state")
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["start_new_session"] is True
    assert Path(proc.stdout_log_path).name == "iterate_v4_attempt_5_stdout.log"
    assert Path(proc.stderr_log_path).name == "iterate_v4_attempt_5_stderr.log"


def test_stage_env_is_loaded_from_campaign_spec(tmp_path, monkeypatch):
    task_file = tmp_path / "task.txt"
    task_file.write_text("revise the paper")
    campaign_file = tmp_path / "campaign.yaml"
    monkeypatch.setenv("LIT_WAIT", "1800")
    campaign_file.write_text(
        "name: Muon\n"
        f'workspace_root: "{tmp_path / "results"}"\n'
        "stages:\n"
        "  - id: iterate_v4\n"
        f'    task_file: "{task_file}"\n'
        "    env:\n"
        '      CONSORTIUM_LIT_MAX_WAIT_SEC: "${LIT_WAIT}"\n'
        '      CONSORTIUM_SS_COOLDOWN_SEC: "120"\n'
    )

    spec = load_spec(str(campaign_file))
    stage = spec.stage("iterate_v4")

    assert stage is not None
    assert stage.env["CONSORTIUM_LIT_MAX_WAIT_SEC"] == "1800"
    assert stage.env["CONSORTIUM_SS_COOLDOWN_SEC"] == "120"


def test_launch_stage_slurm_exports_stage_env_and_timeout(tmp_path, monkeypatch):
    campaign_dir = tmp_path / "campaign"
    task_file = tmp_path / "task.txt"
    launcher = tmp_path / "launch_multiagent.py"
    task_file.write_text("revise the paper")
    launcher.write_text("print('placeholder launcher')\n")
    campaign_dir.mkdir()

    stage = Stage.from_dict(
        {
            "id": "iterate_v4",
            "task_file": str(task_file),
            "env": {"CONSORTIUM_SS_COOLDOWN_SEC": "120"},
            "args": ["--iterate-start-stage", "literature_review_agent"],
            "success_artifacts": {"required": []},
        }
    )
    spec = type(
        "Spec",
        (),
        {
            "workspace_root": str(campaign_dir),
            "budget_usd": 0.0,
            "planning": None,
            "counsel_model_timeout_seconds": 444,
        },
    )()
    status = CampaignStatus(
        {
            "campaign_name": "Muon",
            "spec_file": "campaign.yaml",
            "stages": {"iterate_v4": {"attempt_id": 1}},
        }
    )

    captured: dict[str, str] = {}

    class DummyResult:
        returncode = 0
        stdout = "Submitted batch job 321\n"
        stderr = ""

    def fake_run(cmd, **kwargs):
        script_path = Path(cmd[1])
        captured["script"] = script_path.read_text()
        return DummyResult()

    monkeypatch.setattr("consortium.campaign.runner.subprocess.run", fake_run)

    job_id = launch_stage_slurm(
        stage=stage,
        spec=spec,
        status=status,
        campaign_dir=str(campaign_dir),
        launcher=str(launcher),
    )

    assert job_id == 321
    assert "export CONSORTIUM_SS_COOLDOWN_SEC=120" in captured["script"]
    assert "export COUNSEL_MODEL_TIMEOUT_SECONDS=444" in captured["script"]
    assert "export CONSORTIUM_LIT_RATE_STATE_DIR=" in captured["script"]


def test_run_heartbeat_cools_down_provider_saturation_without_repair(tmp_path, monkeypatch):
    from scripts.campaign_heartbeat import run_heartbeat

    campaign_dir = tmp_path / "campaign"
    workspace = campaign_dir / "iterate_v5"
    task_file = tmp_path / "task.txt"
    stdout_log = campaign_dir / "logs" / "iterate_v5_stdout.log"
    stderr_log = campaign_dir / "logs" / "iterate_v5_stderr.log"
    campaign_dir.mkdir()
    workspace.mkdir()
    stdout_log.parent.mkdir()
    task_file.write_text("continue from literature review")
    stdout_log.write_text(
        "literature_review_agent starting\n"
        "Semantic Scholar rate limit exceeded\n"
        "[literature-rate-limit] Provider 'semantic_scholar' saturated; cooling down\n"
        "arXiv request timed out\n"
    )
    stderr_log.write_text("")

    stage = Stage.from_dict(
        {
            "id": "iterate_v5",
            "task_file": str(task_file),
            "success_artifacts": {"required": ["paper_workspace/literature_review.tex"]},
        }
    )
    status = CampaignStatus(
        {
            "campaign_name": "Muon",
            "spec_file": "campaign.yaml",
            "stages": {
                "iterate_v5": {
                    "status": "in_progress",
                    "workspace": str(workspace),
                    "pid": 999999999,
                    "slurm_job_id": None,
                    "attempt_id": 4,
                    "stdout_log": str(stdout_log),
                    "stderr_log": str(stderr_log),
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "completed_at": None,
                    "missing_artifacts": [],
                    "fail_reason": None,
                }
            },
        }
    )
    spec = type(
        "Spec",
        (),
        {
            "name": "Muon",
            "stages": [stage],
            "notification": type("Notification", (), {})(),
            "repair": type(
                "Repair",
                (),
                {
                    "enabled": True,
                    "backoff_base_seconds": 0,
                    "backoff_max_seconds": 0,
                    "repairing_timeout_seconds": 3600,
                    "auto_retry_on_timeout": False,
                    "escalation_timeout_minutes": 60,
                    "max_attempts": 2,
                },
            )(),
            "budget_usd": 0.0,
            "max_campaign_hours": 0.0,
            "planning": None,
            "max_idle_ticks": 8,
        },
    )()

    notices: list[str] = []

    class DummyProc:
        pid = 42424
        attempt_id = 5
        stdout_log_path = str(campaign_dir / "logs" / "iterate_v5_attempt_5_stdout.log")
        stderr_log_path = str(campaign_dir / "logs" / "iterate_v5_attempt_5_stderr.log")
        stage_workspace = str(workspace)

    monkeypatch.setattr("scripts.campaign_heartbeat._preflight_api_check", lambda: True)
    monkeypatch.setattr("scripts.campaign_heartbeat.notify", lambda msg, _cfg: notices.append(msg))
    monkeypatch.setattr("scripts.campaign_heartbeat.notify_stage_launched", lambda *args, **kwargs: None)
    monkeypatch.setattr("scripts.campaign_heartbeat.notify_stage_complete", lambda *args, **kwargs: None)
    monkeypatch.setattr("scripts.campaign_heartbeat.notify_stage_failed", lambda *args, **kwargs: None)
    monkeypatch.setattr("scripts.campaign_heartbeat.notify_heartbeat", lambda *args, **kwargs: None)
    monkeypatch.setattr("scripts.campaign_heartbeat.distill_stage_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "scripts.campaign_heartbeat._try_repair",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("repair should not run")),
    )
    monkeypatch.setattr("scripts.campaign_heartbeat.launch_stage", lambda *args, **kwargs: DummyProc())

    first_exit = run_heartbeat(spec, status, str(campaign_dir))
    assert first_exit == 1
    assert status.stage_status("iterate_v5") == "failed"
    assert status.raw()["stages"]["iterate_v5"]["provider_saturation_log"]
    assert any("cooling down on external literature providers" in msg for msg in notices)

    second_exit = run_heartbeat(spec, status, str(campaign_dir))
    assert second_exit == 3
    assert status.stage_status("iterate_v5") == "in_progress"
    assert status.stage_pid("iterate_v5") == 42424


def test_manual_failed_status_preserves_existing_reason_and_supports_override(
    tmp_path, monkeypatch
):
    from scripts import campaign_cli

    task_file = tmp_path / "task.txt"
    task_file.write_text("do work")
    campaign_dir = tmp_path / "results"
    campaign_dir.mkdir()
    campaign_file = tmp_path / "campaign.yaml"
    campaign_file.write_text(
        "name: Muon\n"
        f'workspace_root: "{campaign_dir}"\n'
        "planning:\n"
        "  enabled: false\n"
        "stages:\n"
        "  - id: iterate_v4\n"
        f'    task_file: "{task_file}"\n'
    )

    status_path = campaign_dir / "campaign_status.json"
    status_path.write_text(
        json.dumps(
            {
                "campaign_name": "Muon",
                "spec_file": str(campaign_file),
                "stages": {
                    "iterate_v4": {
                        "status": "failed",
                        "workspace": str(campaign_dir / "iterate_v4"),
                        "pid": None,
                        "slurm_job_id": None,
                        "attempt_id": 6,
                        "stdout_log": None,
                        "stderr_log": None,
                        "started_at": None,
                        "completed_at": None,
                        "missing_artifacts": ["paper_workspace/brainstorm.json"],
                        "fail_reason": "formalize_goals_agent requires brainstorm artifacts",
                    }
                },
            }
        )
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign_cli.py",
            "--campaign",
            str(campaign_file),
            "set-stage-status",
            "iterate_v4",
            "failed",
        ],
    )
    assert campaign_cli.main() == 0

    preserved = json.loads(status_path.read_text())
    assert (
        preserved["stages"]["iterate_v4"]["fail_reason"]
        == "formalize_goals_agent requires brainstorm artifacts"
    )
    assert preserved["stages"]["iterate_v4"]["missing_artifacts"] == [
        "paper_workspace/brainstorm.json"
    ]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign_cli.py",
            "--campaign",
            str(campaign_file),
            "set-stage-status",
            "iterate_v4",
            "failed",
            "--reason",
            "operator override",
        ],
    )
    assert campaign_cli.main() == 0

    overridden = json.loads(status_path.read_text())
    assert overridden["stages"]["iterate_v4"]["fail_reason"] == "operator override"
    assert overridden["stages"]["iterate_v4"]["missing_artifacts"] == [
        "paper_workspace/brainstorm.json"
    ]
