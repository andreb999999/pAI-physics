"""
MathClaimGraphTool - manage theorem/lemma claim graphs for theory workflows.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from smolagents import Tool


class MathClaimGraphTool(Tool):
    name = "math_claim_graph_tool"
    description = """
    Manage a shared math claim graph at math_workspace/claim_graph.json.

    Use this tool to initialize the graph, add/update claims, connect dependencies,
    validate structural consistency, and list progress by status.

    Actions:
    - init
    - add_claim
    - update_claim
    - set_status
    - add_dependency
    - get_claim
    - list_claims
    - validate_graph
    """

    inputs = {
        "action": {
            "type": "string",
            "description": "Action name (init, add_claim, update_claim, set_status, add_dependency, get_claim, list_claims, validate_graph)",
        },
        "claim_id": {
            "type": "string",
            "description": "Unique claim id (required for claim-specific actions)",
            "nullable": True,
        },
        "statement": {
            "type": "string",
            "description": "Claim statement in plain text or LaTeX",
            "nullable": True,
        },
        "assumptions_json": {
            "type": "string",
            "description": "JSON array of assumptions, e.g. [\"A1\",\"A2\"]",
            "nullable": True,
        },
        "depends_on_json": {
            "type": "string",
            "description": "JSON array of dependency claim ids",
            "nullable": True,
        },
        "tags_json": {
            "type": "string",
            "description": "JSON array of tags",
            "nullable": True,
        },
        "status": {
            "type": "string",
            "description": "Claim status (proposed, proved_draft, verified_symbolic, verified_numeric, accepted, rejected)",
            "nullable": True,
        },
        "notes": {
            "type": "string",
            "description": "Optional notes for the claim",
            "nullable": True,
        },
        "must_accept": {
            "type": "boolean",
            "description": "Whether this claim is required for run completion",
            "nullable": True,
        },
        "workspace_subdir": {
            "type": "string",
            "description": "Workspace subdir where claim_graph.json lives (default: math_workspace)",
            "nullable": True,
        },
    }

    output_type = "string"

    _VALID_STATUSES = {
        "proposed",
        "proved_draft",
        "verified_symbolic",
        "verified_numeric",
        "accepted",
        "rejected",
    }

    def __init__(self, working_dir: Optional[str] = None):
        super().__init__()
        self.working_dir = os.path.abspath(working_dir) if working_dir else None

    def forward(
        self,
        action: str,
        claim_id: Optional[str] = None,
        statement: Optional[str] = None,
        assumptions_json: Optional[str] = None,
        depends_on_json: Optional[str] = None,
        tags_json: Optional[str] = None,
        status: Optional[str] = None,
        notes: Optional[str] = None,
        must_accept: Optional[bool] = None,
        workspace_subdir: Optional[str] = "math_workspace",
    ) -> str:
        try:
            action = (action or "").strip()
            if not action:
                return self._error("'action' is required")

            workspace_dir = self._resolve_workspace_dir(workspace_subdir or "math_workspace")
            os.makedirs(workspace_dir, exist_ok=True)
            graph_path = os.path.join(workspace_dir, "claim_graph.json")

            if action == "init":
                graph = self._load_graph(graph_path)
                self._save_graph(graph_path, graph)
                lemma_library_path = self._ensure_lemma_library(workspace_dir)
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "graph_path": graph_path,
                        "lemma_library_path": lemma_library_path,
                        "claim_count": len(graph["claims"]),
                    },
                    indent=2,
                )

            graph = self._load_graph(graph_path)

            if action == "add_claim":
                if not claim_id or not statement:
                    return self._error("'claim_id' and 'statement' are required for add_claim")
                if self._find_claim(graph, claim_id) is not None:
                    return self._error(f"claim '{claim_id}' already exists")

                claim = {
                    "id": claim_id,
                    "statement": statement,
                    "assumptions": self._parse_json_list(assumptions_json),
                    "depends_on": self._parse_json_list(depends_on_json),
                    "tags": self._parse_json_list(tags_json),
                    "status": status if status in self._VALID_STATUSES else "proposed",
                    "notes": notes or "",
                    "must_accept": bool(must_accept),
                    "created_at": self._now_iso(),
                    "updated_at": self._now_iso(),
                }
                graph["claims"].append(claim)
                self._save_graph(graph_path, graph)
                return json.dumps({"success": True, "action": action, "claim": claim}, indent=2)

            if action == "update_claim":
                if not claim_id:
                    return self._error("'claim_id' is required for update_claim")
                claim = self._find_claim(graph, claim_id)
                if claim is None:
                    return self._error(f"claim '{claim_id}' not found")

                if statement is not None:
                    claim["statement"] = statement
                if assumptions_json is not None:
                    claim["assumptions"] = self._parse_json_list(assumptions_json)
                if depends_on_json is not None:
                    claim["depends_on"] = self._parse_json_list(depends_on_json)
                if tags_json is not None:
                    claim["tags"] = self._parse_json_list(tags_json)
                if notes is not None:
                    claim["notes"] = notes
                if must_accept is not None:
                    claim["must_accept"] = bool(must_accept)
                if status is not None:
                    if status not in self._VALID_STATUSES:
                        return self._error(
                            f"invalid status '{status}'. valid: {sorted(self._VALID_STATUSES)}"
                        )
                    claim["status"] = status

                claim["updated_at"] = self._now_iso()
                self._save_graph(graph_path, graph)
                return json.dumps({"success": True, "action": action, "claim": claim}, indent=2)

            if action == "set_status":
                if not claim_id or not status:
                    return self._error("'claim_id' and 'status' are required for set_status")
                if status not in self._VALID_STATUSES:
                    return self._error(f"invalid status '{status}'. valid: {sorted(self._VALID_STATUSES)}")

                claim = self._find_claim(graph, claim_id)
                if claim is None:
                    return self._error(f"claim '{claim_id}' not found")
                claim["status"] = status
                claim["updated_at"] = self._now_iso()
                self._save_graph(graph_path, graph)
                return json.dumps({"success": True, "action": action, "claim": claim}, indent=2)

            if action == "add_dependency":
                if not claim_id or not depends_on_json:
                    return self._error("'claim_id' and 'depends_on_json' are required for add_dependency")
                claim = self._find_claim(graph, claim_id)
                if claim is None:
                    return self._error(f"claim '{claim_id}' not found")

                deps = self._parse_json_list(depends_on_json)
                for dep in deps:
                    if dep not in claim["depends_on"]:
                        claim["depends_on"].append(dep)
                claim["updated_at"] = self._now_iso()
                self._save_graph(graph_path, graph)
                return json.dumps({"success": True, "action": action, "claim": claim}, indent=2)

            if action == "get_claim":
                if not claim_id:
                    return self._error("'claim_id' is required for get_claim")
                claim = self._find_claim(graph, claim_id)
                if claim is None:
                    return self._error(f"claim '{claim_id}' not found")
                return json.dumps({"success": True, "action": action, "claim": claim}, indent=2)

            if action == "list_claims":
                counts: Dict[str, int] = {}
                for claim in graph["claims"]:
                    s = claim.get("status", "proposed")
                    counts[s] = counts.get(s, 0) + 1
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "graph_path": graph_path,
                        "claim_count": len(graph["claims"]),
                        "status_counts": counts,
                        "claims": graph["claims"],
                    },
                    indent=2,
                )

            if action == "validate_graph":
                validation = self._validate_graph(graph)
                return json.dumps({"success": True, "action": action, **validation}, indent=2)

            return self._error(f"unknown action '{action}'")

        except Exception as e:
            return self._error(f"math claim graph tool failed: {e}")

    def _resolve_workspace_dir(self, workspace_subdir: str) -> str:
        if self.working_dir:
            abs_working = os.path.abspath(self.working_dir)
            if os.path.isabs(workspace_subdir):
                target = os.path.abspath(workspace_subdir)
            else:
                target = os.path.abspath(os.path.join(abs_working, workspace_subdir))
            try:
                within_root = os.path.commonpath([abs_working, target]) == abs_working
            except ValueError:
                within_root = False
            if not within_root:
                raise PermissionError(
                    f"Access denied: workspace_subdir '{workspace_subdir}' resolves outside '{abs_working}'"
                )
            return target

        return os.path.abspath(workspace_subdir)

    @staticmethod
    def _default_graph() -> Dict[str, Any]:
        now = MathClaimGraphTool._now_iso()
        return {
            "version": 1,
            "created_at": now,
            "updated_at": now,
            "claims": [],
        }

    def _load_graph(self, graph_path: str) -> Dict[str, Any]:
        if not os.path.exists(graph_path):
            return self._default_graph()

        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "claims" not in data or not isinstance(data["claims"], list):
            raise ValueError("invalid claim graph format: missing 'claims' list")
        return data

    def _save_graph(self, graph_path: str, graph: Dict[str, Any]) -> None:
        graph["updated_at"] = self._now_iso()
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)

    def _ensure_lemma_library(self, workspace_dir: str) -> str:
        path = os.path.join(workspace_dir, "lemma_library.md")
        if os.path.exists(path):
            return path

        default_text = """# Standard Lemma Library (Math Fast Path)

Use this file to avoid re-deriving standard results.

Policy:
- Tier 0 (primitive): use inline without claim nodes (e.g., triangle inequality, Cauchy-Schwarz).
- Tier 1 (standard ML-theory): prefer library-backed claim nodes with `origin:library`.
- Tier 2 (specialized known): require explicit conditions and source notes.
- Tier 3 (novel): full proof workflow required.

Suggested entries:
- L_smooth_descent: Descent lemma under L-smoothness.
- L_convex_first_order: First-order condition for convex functions.
- L_trace_frobenius_duality: trace(A^T B) <= ||A||_F ||B||_F.
- L_operator_nuclear_duality: <A,B> <= ||A||_2 ||B||_*.

For each library-backed claim you use, ensure claim notes include:
- exact condition set
- source pointer (paper/textbook/internal note)
- any scope caveats
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(default_text)
        return path

    @staticmethod
    def _find_claim(graph: Dict[str, Any], claim_id: str) -> Optional[Dict[str, Any]]:
        for claim in graph.get("claims", []):
            if claim.get("id") == claim_id:
                return claim
        return None

    @staticmethod
    def _parse_json_list(raw: Optional[str]) -> List[str]:
        if raw is None:
            return []
        text = raw.strip()
        if not text:
            return []
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError("expected a JSON array")
        return [str(x) for x in value]

    def _validate_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        claims = graph.get("claims", [])
        ids = {str(c.get("id")) for c in claims if c.get("id")}
        missing_dependencies: Dict[str, List[str]] = {}
        invalid_statuses: Dict[str, str] = {}

        for claim in claims:
            cid = str(claim.get("id", ""))
            status = str(claim.get("status", "proposed"))
            if status not in self._VALID_STATUSES:
                invalid_statuses[cid] = status
            deps = [str(d) for d in claim.get("depends_on", [])]
            missing = [d for d in deps if d not in ids]
            if missing:
                missing_dependencies[cid] = missing

        adjacency: Dict[str, List[str]] = {
            str(claim.get("id")): [str(d) for d in claim.get("depends_on", []) if str(claim.get("id")) != str(d)]
            for claim in claims
            if claim.get("id")
        }

        cycles = self._find_cycles(adjacency)
        is_valid = not missing_dependencies and not invalid_statuses and not cycles

        return {
            "is_valid": is_valid,
            "missing_dependencies": missing_dependencies,
            "invalid_statuses": invalid_statuses,
            "cycles": cycles,
        }

    @staticmethod
    def _find_cycles(adjacency: Dict[str, List[str]]) -> List[List[str]]:
        visited: Dict[str, int] = {}
        stack: List[str] = []
        cycles: List[List[str]] = []

        def dfs(node: str) -> None:
            visited[node] = 1
            stack.append(node)
            for nxt in adjacency.get(node, []):
                if nxt not in adjacency:
                    continue
                if visited.get(nxt, 0) == 0:
                    dfs(nxt)
                elif visited.get(nxt) == 1:
                    if nxt in stack:
                        i = stack.index(nxt)
                        cycles.append(stack[i:] + [nxt])
            stack.pop()
            visited[node] = 2

        for n in adjacency:
            if visited.get(n, 0) == 0:
                dfs(n)

        dedup: List[List[str]] = []
        seen = set()
        for c in cycles:
            key = tuple(c)
            if key not in seen:
                seen.add(key)
                dedup.append(c)
        return dedup

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _error(message: str) -> str:
        return json.dumps({"success": False, "error": message}, indent=2)
