"""
Validation helpers for paper-to-claim traceability.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Set


def _looks_like_theorem_label(key: str) -> bool:
    k = str(key).strip().lower()
    return bool(
        re.match(
            r"^(thm|theorem|lem|lemma|prop|proposition|cor|corollary|def|definition|claim|result|eq|equation)[:_].+",
            k,
        )
    )


def _extract_claim_refs(payload: Any) -> Set[str]:
    refs: Set[str] = set()

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                cid = item.get("source_claim_id") or item.get("claim_id") or item.get("source_claim")
                if cid:
                    refs.add(str(cid))
            elif isinstance(item, str):
                refs.add(item)
        return refs

    if not isinstance(payload, dict):
        return refs

    # Common schema: {"entries": [{...}]}
    entries = payload.get("entries")
    if isinstance(entries, list):
        refs.update(_extract_claim_refs(entries))

    # Common schema: {"claims": {...}} or {"claims": [{...}]}
    claims_block = payload.get("claims")
    if isinstance(claims_block, list):
        refs.update(_extract_claim_refs(claims_block))
    elif isinstance(claims_block, dict):
        # Could be mapping claim_id -> metadata
        for key, value in claims_block.items():
            if isinstance(value, dict):
                cid = value.get("source_claim_id") or value.get("claim_id") or key
                if cid:
                    refs.add(str(cid))
            elif isinstance(value, str):
                refs.add(str(value))
            else:
                refs.add(str(key))

    # Optional top-level single-entry schema.
    top_level_cid = payload.get("source_claim_id") or payload.get("claim_id")
    if top_level_cid:
        refs.add(str(top_level_cid))

    # Top-level mapping schema: {"thm:main": "T_main"}.
    for key, value in payload.items():
        if key in {"entries", "claims", "source_claim_id", "claim_id", "version", "generated_at", "notes", "schema"}:
            continue
        if isinstance(value, str) and _looks_like_theorem_label(key):
            refs.add(str(value))
        elif isinstance(value, dict) and _looks_like_theorem_label(key):
            cid = value.get("source_claim_id") or value.get("claim_id")
            if cid:
                refs.add(str(cid))

    return refs


def validate_claim_traceability(workspace_dir: str) -> Dict[str, Any]:
    """
    Validate paper claim traceability against math claim graph.

    Rules:
    - claim_traceability.json must exist and parse.
    - Every referenced claim must exist in claim_graph.json.
    - Every referenced claim must be accepted.
    - Every must_accept claim must appear in traceability references.
    """
    paper_traceability_path = os.path.join(
        workspace_dir, "paper_workspace", "claim_traceability.json"
    )
    graph_path = os.path.join(workspace_dir, "math_workspace", "claim_graph.json")

    errors: List[str] = []

    if not os.path.exists(paper_traceability_path):
        return {
            "traceability_present": False,
            "graph_present": os.path.exists(graph_path),
            "is_valid": False,
            "errors": ["missing paper_workspace/claim_traceability.json"],
            "referenced_claim_ids": [],
            "must_accept_claim_ids": [],
        }

    if not os.path.exists(graph_path):
        return {
            "traceability_present": True,
            "graph_present": False,
            "is_valid": False,
            "errors": ["missing math_workspace/claim_graph.json"],
            "referenced_claim_ids": [],
            "must_accept_claim_ids": [],
        }

    try:
        with open(paper_traceability_path, "r", encoding="utf-8") as f:
            traceability_payload = json.load(f)
    except Exception as e:
        return {
            "traceability_present": True,
            "graph_present": True,
            "is_valid": False,
            "errors": [f"failed to parse claim_traceability.json: {e}"],
            "referenced_claim_ids": [],
            "must_accept_claim_ids": [],
        }

    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            graph_payload = json.load(f)
    except Exception as e:
        return {
            "traceability_present": True,
            "graph_present": True,
            "is_valid": False,
            "errors": [f"failed to parse claim_graph.json: {e}"],
            "referenced_claim_ids": [],
            "must_accept_claim_ids": [],
        }

    claims = graph_payload.get("claims", [])
    if not isinstance(claims, list):
        return {
            "traceability_present": True,
            "graph_present": True,
            "is_valid": False,
            "errors": ["invalid claim graph: 'claims' must be a list"],
            "referenced_claim_ids": [],
            "must_accept_claim_ids": [],
        }

    id_to_status: Dict[str, str] = {}
    must_accept_claim_ids: List[str] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        cid = claim.get("id")
        if not cid:
            continue
        cid_str = str(cid)
        id_to_status[cid_str] = str(claim.get("status", "")).strip()
        if bool(claim.get("must_accept")):
            must_accept_claim_ids.append(cid_str)

    referenced_claim_ids = sorted(_extract_claim_refs(traceability_payload))

    if not referenced_claim_ids:
        errors.append("claim_traceability.json has no referenced claim ids")

    for cid in referenced_claim_ids:
        if cid not in id_to_status:
            errors.append(f"traceability references unknown claim id: {cid}")
            continue
        if id_to_status[cid] != "accepted":
            errors.append(
                f"traceability references non-accepted claim id: {cid} (status={id_to_status[cid]})"
            )

    referenced_set = set(referenced_claim_ids)
    for cid in must_accept_claim_ids:
        if cid not in referenced_set:
            errors.append(f"must_accept claim missing from traceability: {cid}")

    return {
        "traceability_present": True,
        "graph_present": True,
        "is_valid": len(errors) == 0,
        "errors": errors,
        "referenced_claim_ids": referenced_claim_ids,
        "must_accept_claim_ids": sorted(must_accept_claim_ids),
    }
