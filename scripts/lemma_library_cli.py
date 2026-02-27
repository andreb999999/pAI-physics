#!/usr/bin/env python3
"""
Incremental lemma-library helper (standalone, no runtime deps on smolagents).

Maintains:
- <workspace>/lemma_library_index.json (authoritative incremental store)
- <workspace>/lemma_library.md (human-readable synchronized view)
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_tier(value: Optional[str]) -> str:
    v = (value or "").strip().lower().replace(" ", "")
    if v in {"0", "tier0"}:
        return "tier0"
    if v in {"1", "tier1"}:
        return "tier1"
    if v in {"2", "tier2"}:
        return "tier2"
    if v in {"3", "tier3"}:
        return "tier3"
    return "tier1"


def normalize_status(value: Optional[str]) -> str:
    v = (value or "").strip().lower()
    if v in {"active", "deprecated", "draft"}:
        return v
    return "active"


def parse_tags(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    val = json.loads(raw)
    if not isinstance(val, list):
        raise ValueError("--tags-json must be a JSON array")
    return [str(x) for x in val]


def ensure_workspace(path: str) -> str:
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def index_path(workspace: str) -> str:
    return os.path.join(workspace, "lemma_library_index.json")


def markdown_path(workspace: str) -> str:
    return os.path.join(workspace, "lemma_library.md")


def default_entries() -> List[Dict[str, Any]]:
    t = now_iso()
    return [
        {
            "id": "L_smooth_descent_standard",
            "tier": "tier1",
            "statement": "Descent inequality for L-smooth objectives.",
            "conditions": "Objective is differentiable and L-smooth.",
            "source": "Canonical optimization texts (set exact source per project).",
            "usage_notes": "Use for one-step progress bounds.",
            "tags": ["area:optimization", "origin:library"],
            "status": "active",
            "usage_count": 0,
            "created_at": t,
            "updated_at": t,
            "last_used_at": "",
        },
        {
            "id": "L_operator_nuclear_duality",
            "tier": "tier1",
            "statement": "<A,B> <= ||A||_2 ||B||_* for compatible real matrices.",
            "conditions": "Finite-dimensional real matrices with compatible shapes.",
            "source": "Standard matrix analysis references.",
            "usage_notes": "Use in operator-vs-nuclear norm arguments.",
            "tags": ["area:matrix", "origin:library"],
            "status": "active",
            "usage_count": 0,
            "created_at": t,
            "updated_at": t,
            "last_used_at": "",
        },
    ]


def load_index(workspace: str) -> Dict[str, Any]:
    p = index_path(workspace)
    if not os.path.exists(p):
        idx = {
            "version": 1,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "entries": default_entries(),
        }
        save_index(workspace, idx)
        sync_markdown(workspace, idx)
        return idx
    with open(p, "r", encoding="utf-8") as f:
        idx = json.load(f)
    if "entries" not in idx or not isinstance(idx["entries"], list):
        raise ValueError("Invalid lemma_library_index.json: missing entries list")
    return idx


def save_index(workspace: str, idx: Dict[str, Any]) -> None:
    idx["updated_at"] = now_iso()
    with open(index_path(workspace), "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)


def sync_markdown(workspace: str, idx: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Standard Lemma Library (Math Fast Path)")
    lines.append("")
    lines.append("This file is synchronized from `lemma_library_index.json`.")
    lines.append("Use this as a readable view; update entries incrementally via CLI/tool actions.")
    lines.append("")
    lines.append("Policy:")
    lines.append("- Tier 0 (primitive): inline only, usually no claim node.")
    lines.append("- Tier 1 (standard ML-theory): prefer library-backed claim nodes (`origin:library`).")
    lines.append("- Tier 2 (specialized known): explicit conditions and source required.")
    lines.append("- Tier 3 (novel): full proof workflow required.")
    lines.append("")

    entries = idx.get("entries", [])
    if not entries:
        lines.append("_No entries yet._")
    else:
        for e in entries:
            lines.append(f"## {e.get('id','')}")
            lines.append(f"- tier: {e.get('tier','tier1')}")
            lines.append(f"- status: {e.get('status','active')}")
            lines.append(f"- statement: {e.get('statement','')}")
            lines.append(f"- conditions: {e.get('conditions','')}")
            lines.append(f"- source: {e.get('source','')}")
            lines.append(f"- usage_notes: {e.get('usage_notes','')}")
            tags = e.get("tags", [])
            lines.append(f"- tags: {', '.join(str(t) for t in tags) if tags else ''}")
            lines.append(f"- usage_count: {e.get('usage_count',0)}")
            lines.append(f"- last_used_at: {e.get('last_used_at','')}")
            lines.append("")

    with open(markdown_path(workspace), "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def find_entry(idx: Dict[str, Any], lemma_id: str) -> Optional[Dict[str, Any]]:
    for e in idx.get("entries", []):
        if str(e.get("id", "")) == str(lemma_id):
            return e
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Manage math lemma library incrementally")
    p.add_argument("--workspace", default="math_workspace", help="Workspace dir (absolute or relative)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List lemma entries")
    p_list.add_argument("--limit", type=int, default=None)

    p_get = sub.add_parser("get", help="Get one lemma entry")
    p_get.add_argument("--lemma-id", required=True)

    p_upsert = sub.add_parser("upsert", help="Create or update a lemma entry")
    p_upsert.add_argument("--lemma-id", required=True)
    p_upsert.add_argument("--tier", default=None)
    p_upsert.add_argument("--statement", default=None)
    p_upsert.add_argument("--conditions", default=None)
    p_upsert.add_argument("--source", default=None)
    p_upsert.add_argument("--usage-notes", default=None)
    p_upsert.add_argument("--status", default=None)
    p_upsert.add_argument("--tags-json", default=None)

    p_touch = sub.add_parser("touch", help="Increment usage count for a lemma")
    p_touch.add_argument("--lemma-id", required=True)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    workspace = ensure_workspace(args.workspace)
    idx = load_index(workspace)

    if args.cmd == "list":
        entries = idx.get("entries", [])
        if args.limit is not None:
            entries = entries[: max(0, int(args.limit))]
        print(json.dumps({"success": True, "action": "list", "entries": entries}, indent=2))
        return 0

    if args.cmd == "get":
        e = find_entry(idx, args.lemma_id)
        if e is None:
            print(json.dumps({"success": False, "error": f"lemma '{args.lemma_id}' not found"}, indent=2))
            return 1
        print(json.dumps({"success": True, "action": "get", "lemma": e}, indent=2))
        return 0

    if args.cmd == "upsert":
        e = find_entry(idx, args.lemma_id)
        created = e is None
        if created:
            if not args.statement:
                print(json.dumps({"success": False, "error": "--statement is required for new lemma"}, indent=2))
                return 1
            t = now_iso()
            e = {
                "id": args.lemma_id,
                "tier": "tier1",
                "statement": "",
                "conditions": "",
                "source": "",
                "usage_notes": "",
                "tags": [],
                "status": "active",
                "usage_count": 0,
                "created_at": t,
                "updated_at": t,
                "last_used_at": "",
            }
            idx.setdefault("entries", []).append(e)

        if args.tier is not None:
            e["tier"] = normalize_tier(args.tier)
        if args.statement is not None:
            e["statement"] = args.statement
        if args.conditions is not None:
            e["conditions"] = args.conditions
        if args.source is not None:
            e["source"] = args.source
        if args.usage_notes is not None:
            e["usage_notes"] = args.usage_notes
        if args.status is not None:
            e["status"] = normalize_status(args.status)
        if args.tags_json is not None:
            e["tags"] = parse_tags(args.tags_json)

        e["updated_at"] = now_iso()
        save_index(workspace, idx)
        sync_markdown(workspace, idx)
        print(json.dumps({"success": True, "action": "upsert", "created": created, "lemma": e}, indent=2))
        return 0

    if args.cmd == "touch":
        e = find_entry(idx, args.lemma_id)
        if e is None:
            print(json.dumps({"success": False, "error": f"lemma '{args.lemma_id}' not found"}, indent=2))
            return 1
        e["usage_count"] = int(e.get("usage_count", 0) or 0) + 1
        e["last_used_at"] = now_iso()
        e["updated_at"] = now_iso()
        save_index(workspace, idx)
        sync_markdown(workspace, idx)
        print(json.dumps({"success": True, "action": "touch", "lemma": e}, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
