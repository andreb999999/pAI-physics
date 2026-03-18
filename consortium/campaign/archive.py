"""
Campaign archival and clean-slate enforcement.

Provides functions to:
- Archive completed/failed campaigns (move YAML + results + task files to archive/)
- Verify a new campaign starts from a clean slate
- Migrate legacy *_DEPRECATED.yaml files into the archive
"""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Optional

import yaml

from .status import read_status, COMPLETED, FAILED, PENDING, IN_PROGRESS

# Status values that indicate a stage is "done" (archivable)
_DONE_STATUSES = {COMPLETED, FAILED}
# Status values that indicate a stage is actively running
_ACTIVE_STATUSES = {IN_PROGRESS, "repairing"}

ARCHIVE_DIR = "archive"


def archive_campaign(
    yaml_path: str,
    repo_root: str,
    force: bool = False,
    dry_run: bool = False,
) -> dict:
    """Archive a campaign by moving its YAML, results, and task files to archive/.

    Returns a summary dict with archive_path, manifest, and moved files.
    Raises ValueError if the campaign has active stages (unless force=True).
    """
    yaml_path = os.path.abspath(yaml_path)
    repo_root = os.path.abspath(repo_root)

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    name = raw.get("name", os.path.basename(yaml_path))
    workspace_root = raw.get("workspace_root", "")
    if not os.path.isabs(workspace_root):
        workspace_root = os.path.join(os.path.dirname(yaml_path), workspace_root)

    # Derive slug from workspace dir name
    slug = os.path.basename(workspace_root.rstrip("/"))

    # Read campaign status if available
    status_data = {}
    stage_summary = {}
    if os.path.isdir(workspace_root):
        try:
            status = read_status(workspace_root)
            status_data = status.raw()
            for sid, sdata in status_data.get("stages", {}).items():
                stage_summary[sid] = sdata.get("status", "unknown")
        except Exception:
            pass

    # Safety gate: refuse if any stage is active
    active_stages = [
        sid for sid, st in stage_summary.items() if st in _ACTIVE_STATUSES
    ]
    if active_stages and not force:
        raise ValueError(
            f"Campaign has active stages: {active_stages}. "
            f"Use --force to archive anyway."
        )

    # Build archive directory name
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_name = f"{slug}_{ts}"
    archive_dir = os.path.join(repo_root, ARCHIVE_DIR, archive_name)

    moved_files = []

    if dry_run:
        # Report what would be moved
        if os.path.isdir(workspace_root):
            moved_files.append(f"results: {workspace_root} -> {archive_dir}/results/")
        moved_files.append(f"yaml: {yaml_path} -> {archive_dir}/{os.path.basename(yaml_path)}")

        # Check for generated task files (namespaced or legacy flat)
        gen_dir = os.path.join(repo_root, "automation_tasks", "generated", slug)
        gen_flat = os.path.join(repo_root, "automation_tasks", "generated")
        if os.path.isdir(gen_dir):
            moved_files.append(f"task_files: {gen_dir}/ -> {archive_dir}/automation_tasks/")
        elif os.path.isdir(gen_flat):
            flat_files = _find_legacy_task_files(gen_flat, stage_summary.keys())
            if flat_files:
                moved_files.append(
                    f"task_files (legacy flat): {len(flat_files)} files -> {archive_dir}/automation_tasks/"
                )

        return {
            "archived": False,
            "dry_run": True,
            "archive_path": archive_dir,
            "slug": slug,
            "stage_summary": stage_summary,
            "would_move": moved_files,
        }

    # Create archive directory
    os.makedirs(archive_dir, exist_ok=True)

    # 1. Move results workspace
    if os.path.isdir(workspace_root):
        dest = os.path.join(archive_dir, "results")
        shutil.move(workspace_root, dest)
        moved_files.append(f"results -> {dest}")

    # 2. Move campaign YAML
    yaml_dest = os.path.join(archive_dir, os.path.basename(yaml_path))
    shutil.move(yaml_path, yaml_dest)
    moved_files.append(f"yaml -> {yaml_dest}")

    # 3. Move generated task files (namespaced first, then legacy flat)
    gen_dir = os.path.join(repo_root, "automation_tasks", "generated", slug)
    tasks_dest = os.path.join(archive_dir, "automation_tasks")
    if os.path.isdir(gen_dir):
        shutil.move(gen_dir, tasks_dest)
        moved_files.append(f"task_files -> {tasks_dest}")
    else:
        gen_flat = os.path.join(repo_root, "automation_tasks", "generated")
        flat_files = _find_legacy_task_files(gen_flat, stage_summary.keys())
        if flat_files:
            os.makedirs(tasks_dest, exist_ok=True)
            for fp in flat_files:
                shutil.move(fp, os.path.join(tasks_dest, os.path.basename(fp)))
            moved_files.append(f"task_files (legacy) -> {tasks_dest}")

    # 4. Extract token ledger lines for this campaign
    ledger_path = os.path.join(
        repo_root, ".local", "private_token_usage", "api_token_calls.jsonl"
    )
    if os.path.isfile(ledger_path):
        _extract_token_ledger(ledger_path, slug, archive_dir)

    # 5. Write manifest
    manifest = {
        "campaign_name": name,
        "slug": slug,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "original_yaml": yaml_path,
        "original_workspace": workspace_root,
        "stage_summary": stage_summary,
        "moved_files": moved_files,
    }

    # Try to include budget total
    budget_path = os.path.join(archive_dir, "results", "campaign_budget.json")
    if os.path.isfile(budget_path):
        try:
            with open(budget_path) as f:
                budget = json.load(f)
            manifest["total_spent_usd"] = budget.get("total_spent_usd", 0)
        except Exception:
            pass

    manifest_path = os.path.join(archive_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "archived": True,
        "archive_path": archive_dir,
        "slug": slug,
        "manifest": manifest,
        "moved_files": moved_files,
    }


def verify_clean_slate(slug: str, repo_root: str) -> list[str]:
    """Verify that a new campaign can start from a clean state.

    Returns a list of violation strings. Empty list = clean.
    """
    violations = []

    # 1. Results directory must not exist (or be empty)
    results_dir = os.path.join(repo_root, "results", slug)
    if os.path.isdir(results_dir):
        contents = os.listdir(results_dir)
        if contents:
            violations.append(
                f"Results directory exists and is not empty: results/{slug}/ "
                f"({len(contents)} items)"
            )

    # 2. Generated task files must not exist
    gen_dir = os.path.join(repo_root, "automation_tasks", "generated", slug)
    if os.path.isdir(gen_dir):
        contents = os.listdir(gen_dir)
        if contents:
            violations.append(
                f"Generated task files exist: automation_tasks/generated/{slug}/ "
                f"({len(contents)} files)"
            )

    # 3. No active campaigns (running/repairing stages)
    active = _find_active_campaigns(repo_root)
    if active:
        for yf, stages in active.items():
            violations.append(
                f"Active campaign detected: {yf} has running stages: {stages}"
            )

    # 4. No stale sentinel files in target workspace
    sentinel_patterns = [".idle_ticks", ".api_credit_alert"]
    budget_alert_prefix = ".budget_alert_"
    if os.path.isdir(results_dir):
        for fname in os.listdir(results_dir):
            if fname in sentinel_patterns or fname.startswith(budget_alert_prefix):
                violations.append(
                    f"Stale sentinel file in workspace: results/{slug}/{fname}"
                )

    return violations


def migrate_deprecated_yamls(
    repo_root: str, dry_run: bool = False
) -> list[str]:
    """Migrate legacy *_DEPRECATED.yaml files into archive/.

    Returns list of migrated campaign names.
    """
    migrated = []

    for fname in sorted(os.listdir(repo_root)):
        if not fname.startswith("campaign_") or not fname.endswith("_DEPRECATED.yaml"):
            continue

        yaml_path = os.path.join(repo_root, fname)

        # Extract slug: campaign_v1_DEPRECATED.yaml -> v1
        slug_match = re.match(r"campaign_(.+)_DEPRECATED\.yaml$", fname)
        if not slug_match:
            continue
        slug = slug_match.group(1)

        # Load workspace_root from YAML
        try:
            with open(yaml_path) as f:
                raw = yaml.safe_load(f)
            workspace_root = raw.get("workspace_root", f"results/{slug}")
            if not os.path.isabs(workspace_root):
                workspace_root = os.path.join(repo_root, workspace_root)
        except Exception:
            workspace_root = os.path.join(repo_root, "results", slug)

        # Use file mtime for archive timestamp (original campaign date)
        try:
            mtime = os.path.getmtime(yaml_path)
            ts = datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y%m%d")
        except Exception:
            ts = "unknown"

        archive_name = f"{slug}_{ts}"
        archive_dir = os.path.join(repo_root, ARCHIVE_DIR, archive_name)

        if dry_run:
            migrated.append(f"{fname} -> archive/{archive_name}/")
            continue

        os.makedirs(archive_dir, exist_ok=True)

        # Move results workspace if it exists
        if os.path.isdir(workspace_root):
            shutil.move(workspace_root, os.path.join(archive_dir, "results"))

        # Move the YAML
        shutil.move(yaml_path, os.path.join(archive_dir, fname))

        # Write a minimal manifest
        manifest = {
            "campaign_name": raw.get("name", slug),
            "slug": slug,
            "migrated_from": "deprecated_yaml",
            "archived_at": datetime.now(timezone.utc).isoformat(),
            "original_yaml": yaml_path,
            "original_workspace": workspace_root,
        }
        with open(os.path.join(archive_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        migrated.append(slug)

    return migrated


def auto_archive_finished(
    repo_root: str, dry_run: bool = False
) -> list[str]:
    """Find and archive all finished campaigns (all stages completed or failed).

    Returns list of archived campaign names.
    """
    archived = []

    for fname in sorted(os.listdir(repo_root)):
        if not fname.startswith("campaign_") or not fname.endswith(".yaml"):
            continue
        if fname.endswith("_DEPRECATED.yaml"):
            continue

        yaml_path = os.path.join(repo_root, fname)

        try:
            with open(yaml_path) as f:
                raw = yaml.safe_load(f)
            workspace_root = raw.get("workspace_root", "")
            if not os.path.isabs(workspace_root):
                workspace_root = os.path.join(repo_root, workspace_root)
        except Exception:
            continue

        if not os.path.isdir(workspace_root):
            continue

        # Read status and check if all stages are done
        try:
            status = read_status(workspace_root)
            stages = status.raw().get("stages", {})
        except Exception:
            continue

        if not stages:
            continue

        all_done = all(
            s.get("status") in _DONE_STATUSES for s in stages.values()
        )
        any_active = any(
            s.get("status") in _ACTIVE_STATUSES for s in stages.values()
        )

        if all_done and not any_active:
            if dry_run:
                archived.append(f"{fname} (finished)")
            else:
                try:
                    result = archive_campaign(yaml_path, repo_root)
                    archived.append(result.get("slug", fname))
                except Exception as exc:
                    print(f"[archive] Failed to archive {fname}: {exc}")

    return archived


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _find_active_campaigns(repo_root: str) -> dict[str, list[str]]:
    """Return {yaml_filename: [active_stage_ids]} for campaigns with running stages."""
    active = {}
    for fname in os.listdir(repo_root):
        if not fname.startswith("campaign_") or not fname.endswith(".yaml"):
            continue
        if fname.endswith("_DEPRECATED.yaml"):
            continue

        yaml_path = os.path.join(repo_root, fname)
        try:
            with open(yaml_path) as f:
                raw = yaml.safe_load(f)
            workspace_root = raw.get("workspace_root", "")
            if not os.path.isabs(workspace_root):
                workspace_root = os.path.join(repo_root, workspace_root)

            if not os.path.isdir(workspace_root):
                continue

            status = read_status(workspace_root)
            active_stages = [
                sid
                for sid, sdata in status.raw().get("stages", {}).items()
                if sdata.get("status") in _ACTIVE_STATUSES
            ]
            if active_stages:
                active[fname] = active_stages
        except Exception:
            continue

    return active


def _find_legacy_task_files(gen_dir: str, stage_ids) -> list[str]:
    """Find flat task files in gen_dir matching known stage IDs."""
    found = []
    if not os.path.isdir(gen_dir):
        return found
    for sid in stage_ids:
        task_file = os.path.join(gen_dir, f"{sid}_task.txt")
        if os.path.isfile(task_file):
            found.append(task_file)
    return found


def _extract_token_ledger(
    ledger_path: str, slug: str, archive_dir: str
) -> None:
    """Extract token ledger lines matching the campaign slug into the archive."""
    try:
        out_path = os.path.join(archive_dir, "token_ledger.jsonl")
        count = 0
        with open(ledger_path) as fin, open(out_path, "w") as fout:
            for line in fin:
                if slug in line:
                    fout.write(line)
                    count += 1
        if count == 0:
            os.remove(out_path)
    except Exception:
        pass
