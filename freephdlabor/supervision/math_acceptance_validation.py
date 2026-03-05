"""
Validation helpers for theorem-oriented claim acceptance criteria.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List


def _safe_id(claim_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", str(claim_id))


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except Exception:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _latest(
    rows: List[Dict[str, Any]],
    predicate,
) -> Dict[str, Any] | None:
    """
    Return the latest row matching a predicate using file order.
    """
    for row in reversed(rows):
        try:
            if predicate(row):
                return row
        except Exception:
            continue
    return None


def validate_math_acceptance(workspace_dir: str) -> Dict[str, Any]:
    """
    Validate accepted math claims are backed by auditable artifacts.

    Criteria enforced:
    - Every accepted claim has a proof file.
    - Every accepted claim has a symbolic audit pass record.
    - Every accepted claim has numeric evidence (pass or explicit waive).
    - Every accepted claim depends only on accepted dependencies.
    - Every must_accept claim is accepted.
    """
    math_dir = os.path.join(workspace_dir, "math_workspace")
    graph_path = os.path.join(math_dir, "claim_graph.json")

    if not os.path.exists(graph_path):
        return {
            "graph_present": False,
            "is_valid": True,
            "errors": [],
            "accepted_claim_ids": [],
            "must_accept_claim_ids": [],
        }

    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            graph = json.load(f)
    except Exception as e:
        return {
            "graph_present": True,
            "is_valid": False,
            "errors": [f"failed to parse claim graph: {e}"],
            "accepted_claim_ids": [],
            "must_accept_claim_ids": [],
        }

    claims = graph.get("claims", [])
    if not isinstance(claims, list):
        return {
            "graph_present": True,
            "is_valid": False,
            "errors": ["invalid claim graph: 'claims' must be a list"],
            "accepted_claim_ids": [],
            "must_accept_claim_ids": [],
        }

    id_to_claim: Dict[str, Dict[str, Any]] = {}
    for claim in claims:
        if isinstance(claim, dict) and claim.get("id"):
            id_to_claim[str(claim["id"])] = claim

    # Prevent artifact ambiguity: different claim IDs must not map to same safe filename.
    safe_id_to_claim_ids: Dict[str, List[str]] = {}
    for cid in id_to_claim.keys():
        safe = _safe_id(cid)
        safe_id_to_claim_ids.setdefault(safe, []).append(cid)

    accepted_claim_ids = [
        str(c.get("id"))
        for c in claims
        if isinstance(c, dict) and str(c.get("status", "")).strip() == "accepted" and c.get("id")
    ]
    must_accept_claim_ids = [
        str(c.get("id"))
        for c in claims
        if isinstance(c, dict) and bool(c.get("must_accept")) and c.get("id")
    ]

    errors: List[str] = []

    for safe_id, claim_ids in safe_id_to_claim_ids.items():
        if len(claim_ids) > 1:
            errors.append(
                f"claim_id collision after sanitization ({safe_id}): {', '.join(sorted(claim_ids))}"
            )

    for cid in must_accept_claim_ids:
        claim = id_to_claim.get(cid)
        if not claim or claim.get("status") != "accepted":
            errors.append(f"must_accept claim not accepted: {cid}")

    for cid in accepted_claim_ids:
        claim = id_to_claim.get(cid, {})
        safe_cid = _safe_id(cid)

        proof_path = os.path.join(math_dir, "proofs", f"{safe_cid}.md")
        if not os.path.exists(proof_path):
            errors.append(f"accepted claim missing proof file: {cid}")

        checks_path = os.path.join(math_dir, "checks", f"{safe_cid}.jsonl")
        checks = _read_jsonl(checks_path)
        if not checks:
            errors.append(f"accepted claim missing checks log: {cid}")
            continue

        latest_symbolic = _latest(
            checks, lambda row: str(row.get("check_kind", "")).strip() == "symbolic_rigor_audit"
        )
        if not latest_symbolic:
            errors.append(f"accepted claim missing symbolic_rigor_audit record: {cid}")
        elif str(latest_symbolic.get("verdict", "")).strip().lower() != "pass":
            errors.append(
                f"accepted claim latest symbolic_rigor_audit is not pass: {cid} "
                f"(verdict={latest_symbolic.get('verdict')})"
            )

        latest_numeric = _latest(
            checks,
            lambda row: (
                str(row.get("kind", "")).strip() == "numeric_check"
                or str(row.get("check_kind", "")).strip()
                in {"numeric_protocol_summary", "numeric_check_waived"}
            ),
        )
        if not latest_numeric:
            errors.append(f"accepted claim missing numeric evidence record: {cid}")
        else:
            numeric_kind = str(latest_numeric.get("kind", "")).strip()
            numeric_check_kind = str(latest_numeric.get("check_kind", "")).strip()
            numeric_verdict = str(latest_numeric.get("verdict", "")).strip().lower()
            if numeric_kind == "numeric_check":
                if numeric_verdict != "pass":
                    errors.append(
                        f"accepted claim latest numeric_check is not pass: {cid} "
                        f"(verdict={latest_numeric.get('verdict')})"
                    )
            elif numeric_check_kind in {"numeric_protocol_summary", "numeric_check_waived"}:
                if numeric_verdict not in {"pass", "waived"}:
                    errors.append(
                        f"accepted claim latest {numeric_check_kind} is neither pass nor waived: {cid} "
                        f"(verdict={latest_numeric.get('verdict')})"
                    )
            else:
                errors.append(f"accepted claim has malformed numeric evidence record: {cid}")

        deps = [str(d) for d in claim.get("depends_on", [])]
        for dep in deps:
            if dep == cid:
                errors.append(f"accepted claim has self-dependency: {cid}")
                continue
            dep_claim = id_to_claim.get(dep)
            if dep_claim is None:
                errors.append(f"accepted claim depends on missing claim: {cid} -> {dep}")
                continue
            if dep_claim.get("status") != "accepted":
                errors.append(
                    f"accepted claim depends on non-accepted dependency: {cid} -> {dep} (status={dep_claim.get('status')})"
                )

    return {
        "graph_present": True,
        "is_valid": len(errors) == 0,
        "errors": errors,
        "accepted_claim_ids": accepted_claim_ids,
        "must_accept_claim_ids": must_accept_claim_ids,
    }
