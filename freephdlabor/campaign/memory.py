"""
Campaign memory — distill completed stage artifacts into a markdown summary.

After each stage completes, distill_stage_memory() reads the key output files
from the stage workspace and writes a concise summary to:
    campaign_dir/memory/<stage_id>_summary.md

This summary is then prepended to the next stage's task prompt by runner.py,
giving downstream agents cross-run context without inflating their full context
windows with raw artifact files.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from .spec import Stage


# Maximum characters to include from any single artifact file
_MAX_ARTIFACT_CHARS = 4000


def distill_stage_memory(
    stage: Stage,
    workspace: str,
    campaign_dir: str,
) -> str:
    """
    Read key artifacts from workspace, build a compact markdown summary,
    and write it to campaign_dir/memory/<stage_id>_summary.md.

    Returns the summary text.
    """
    os.makedirs(os.path.join(campaign_dir, "memory"), exist_ok=True)
    out_path = os.path.join(campaign_dir, "memory", f"{stage.id}_summary.md")

    sections: list[str] = [f"# Stage Summary: {stage.id}\n"]

    # ----------------------------------------------------------------
    # Required artifacts — include a short excerpt of each
    # ----------------------------------------------------------------
    for rel_path in stage.success_artifacts.get("required", []):
        full_path = os.path.join(workspace, rel_path.rstrip("/"))
        if os.path.isfile(full_path):
            excerpt = _read_excerpt(full_path)
            sections.append(f"## {rel_path}\n\n{excerpt}\n")
        elif os.path.isdir(full_path):
            listing = _list_dir(full_path)
            sections.append(f"## {rel_path}/ (directory)\n\n{listing}\n")

    # ----------------------------------------------------------------
    # Memory dirs — walk and include short excerpts
    # ----------------------------------------------------------------
    for mem_dir in stage.memory_dirs:
        dir_path = os.path.join(workspace, mem_dir.rstrip("/"))
        if not os.path.isdir(dir_path):
            continue
        sections.append(f"## Memory directory: {mem_dir}\n")
        for fname in sorted(os.listdir(dir_path))[:20]:
            fpath = os.path.join(dir_path, fname)
            if os.path.isfile(fpath):
                excerpt = _read_excerpt(fpath, max_chars=1500)
                sections.append(f"### {fname}\n\n{excerpt}\n")

    # ----------------------------------------------------------------
    # Budget ledger — always include if present
    # ----------------------------------------------------------------
    budget_path = os.path.join(workspace, "budget_state.json")
    if os.path.exists(budget_path):
        try:
            with open(budget_path) as f:
                budget = json.load(f)
            total = budget.get("total_cost_usd", budget.get("total_spent_usd", "unknown"))
            sections.append(f"## Budget\n\nTotal cost: ${total}\n")
        except Exception:
            pass

    # ----------------------------------------------------------------
    # Token usage summary — include if present
    # ----------------------------------------------------------------
    token_path = os.path.join(workspace, "run_token_usage.json")
    if os.path.exists(token_path):
        try:
            with open(token_path) as f:
                usage = json.load(f)
            prompt_t = usage.get("total_prompt_tokens", 0)
            comp_t = usage.get("total_completion_tokens", 0)
            sections.append(
                f"## Token Usage\n\n"
                f"Prompt tokens: {prompt_t:,}  |  Completion tokens: {comp_t:,}\n"
            )
        except Exception:
            pass

    # ----------------------------------------------------------------
    # STATUS.txt — always include
    # ----------------------------------------------------------------
    status_path = os.path.join(workspace, "STATUS.txt")
    if os.path.exists(status_path):
        try:
            with open(status_path) as f:
                status_txt = f.read().strip()
            sections.append(f"## Pipeline Status\n\n{status_txt}\n")
        except Exception:
            pass

    summary = "\n".join(sections)
    with open(out_path, "w") as f:
        f.write(summary)

    print(f"[campaign:memory] Distilled {stage.id} summary → {out_path}")
    return summary


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _read_excerpt(path: str, max_chars: int = _MAX_ARTIFACT_CHARS) -> str:
    """Read a file and return a truncated excerpt, with JSON pretty-printed."""
    try:
        with open(path) as f:
            raw = f.read()
    except Exception as e:
        return f"[Could not read: {e}]"

    # Pretty-print JSON if possible
    if path.endswith(".json"):
        try:
            parsed = json.loads(raw)
            raw = json.dumps(parsed, indent=2)
        except Exception:
            pass

    if len(raw) <= max_chars:
        return f"```\n{raw}\n```"
    return f"```\n{raw[:max_chars]}\n... [truncated, {len(raw)} chars total]\n```"


def _list_dir(path: str, max_entries: int = 30) -> str:
    """Return a formatted directory listing."""
    try:
        entries = sorted(os.listdir(path))
        if len(entries) > max_entries:
            shown = entries[:max_entries]
            return "```\n" + "\n".join(shown) + f"\n... (+{len(entries) - max_entries} more)\n```"
        return "```\n" + "\n".join(entries) + "\n```"
    except Exception as e:
        return f"[Could not list: {e}]"
