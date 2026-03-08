"""
MathProofWorkspaceTool - manage proof artifacts for claim-based theorem workflows.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict


class MathProofWorkspaceToolInput(BaseModel):
    action: str = Field(description="Action name")
    claim_id: Optional[str] = Field(default=None, description="Claim id for proof/check operations")
    content: Optional[str] = Field(default=None, description="Proof or note content")
    check_payload_json: Optional[str] = Field(default=None, description="JSON object payload for append_check")
    workspace_subdir: Optional[str] = Field(default=None, description="Workspace subdir root (default: math_workspace)")


class MathProofWorkspaceTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "math_proof_workspace_tool"
    description: str = """
    Manage math proof artifacts under math_workspace/.

    Files managed:
    - proofs/<claim_id>.md
    - checks/<claim_id>.jsonl

    Actions:
    - init
    - create_template
    - write_proof
    - append_proof
    - read_proof
    - list_proofs
    - append_check
    - list_checks
    """
    args_schema: Type[BaseModel] = MathProofWorkspaceToolInput
    working_dir: Optional[str] = None
    _CLAIM_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

    def __init__(self, working_dir: Optional[str] = None, **kwargs: Any):
        super().__init__(working_dir=os.path.abspath(working_dir) if working_dir else None, **kwargs)

    def _run(
        self,
        action: str,
        claim_id: Optional[str] = None,
        content: Optional[str] = None,
        check_payload_json: Optional[str] = None,
        workspace_subdir: Optional[str] = "math_workspace",
    ) -> str:
        try:
            action = (action or "").strip()
            if not action:
                return self._error("'action' is required")

            workspace_dir = self._resolve_workspace_dir(workspace_subdir or "math_workspace")
            proofs_dir = os.path.join(workspace_dir, "proofs")
            checks_dir = os.path.join(workspace_dir, "checks")
            os.makedirs(proofs_dir, exist_ok=True)
            os.makedirs(checks_dir, exist_ok=True)

            if action == "init":
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "workspace_dir": workspace_dir,
                        "proofs_dir": proofs_dir,
                        "checks_dir": checks_dir,
                    },
                    indent=2,
                )

            if action == "list_proofs":
                files = sorted([f for f in os.listdir(proofs_dir) if f.endswith(".md")])
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "proof_files": files,
                        "count": len(files),
                    },
                    indent=2,
                )

            if not claim_id:
                return self._error("'claim_id' is required for this action")
            if not self._CLAIM_ID_RE.match(str(claim_id)):
                return self._error(
                    f"invalid claim_id '{claim_id}'. Allowed pattern: [A-Za-z0-9_.-]+ "
                    "(prevents proof/check filename collisions)."
                )
            safe_id = self._safe_id(claim_id)
            proof_path = os.path.join(proofs_dir, f"{safe_id}.md")
            checks_path = os.path.join(checks_dir, f"{safe_id}.jsonl")

            if action == "create_template":
                template = self._build_template(workspace_dir, claim_id)
                with open(proof_path, "w", encoding="utf-8") as f:
                    f.write(template)
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "claim_id": claim_id,
                        "proof_path": proof_path,
                    },
                    indent=2,
                )

            if action == "write_proof":
                if content is None:
                    return self._error("'content' is required for write_proof")
                with open(proof_path, "w", encoding="utf-8") as f:
                    f.write(content)
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "claim_id": claim_id,
                        "proof_path": proof_path,
                        "chars": len(content),
                    },
                    indent=2,
                )

            if action == "append_proof":
                if content is None:
                    return self._error("'content' is required for append_proof")
                with open(proof_path, "a", encoding="utf-8") as f:
                    if os.path.exists(proof_path) and os.path.getsize(proof_path) > 0:
                        f.write("\n")
                    f.write(content)
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "claim_id": claim_id,
                        "proof_path": proof_path,
                        "appended_chars": len(content),
                    },
                    indent=2,
                )

            if action == "read_proof":
                if not os.path.exists(proof_path):
                    return self._error(f"proof file not found: {proof_path}")
                with open(proof_path, "r", encoding="utf-8") as f:
                    text = f.read()
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "claim_id": claim_id,
                        "proof_path": proof_path,
                        "chars": len(text),
                        "content": text,
                    },
                    indent=2,
                )

            if action == "append_check":
                payload: Dict[str, Any]
                if check_payload_json is None:
                    payload = {}
                else:
                    value = json.loads(check_payload_json)
                    if not isinstance(value, dict):
                        return self._error("check_payload_json must be a JSON object")
                    payload = value

                payload.setdefault("claim_id", claim_id)
                payload.setdefault("timestamp", self._now_iso())
                with open(checks_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(payload) + "\n")
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "claim_id": claim_id,
                        "checks_path": checks_path,
                        "payload": payload,
                    },
                    indent=2,
                )

            if action == "list_checks":
                if not os.path.exists(checks_path):
                    return json.dumps(
                        {
                            "success": True,
                            "action": action,
                            "claim_id": claim_id,
                            "checks": [],
                            "count": 0,
                        },
                        indent=2,
                    )
                checks = []
                with open(checks_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            checks.append(json.loads(line))
                        except Exception:
                            checks.append({"raw": line, "parse_error": True})
                return json.dumps(
                    {
                        "success": True,
                        "action": action,
                        "claim_id": claim_id,
                        "checks": checks,
                        "count": len(checks),
                    },
                    indent=2,
                )

            return self._error(f"unknown action '{action}'")

        except Exception as e:
            return self._error(f"math proof workspace tool failed: {e}")


    def _build_template(self, workspace_dir: str, claim_id: str) -> str:
        statement = ""
        assumptions = []
        graph_path = os.path.join(workspace_dir, "claim_graph.json")
        if os.path.exists(graph_path):
            try:
                with open(graph_path, "r", encoding="utf-8") as f:
                    graph = json.load(f)
                for claim in graph.get("claims", []):
                    if str(claim.get("id")) == str(claim_id):
                        statement = str(claim.get("statement", "")).strip()
                        assumptions = [str(x) for x in claim.get("assumptions", [])]
                        break
            except Exception:
                pass

        assumptions_block = "\n".join([f"- {a}" for a in assumptions]) if assumptions else "- [fill assumptions]"
        statement_line = statement or "[fill claim statement]"

        return (
            f"# Proof Draft: {claim_id}\n\n"
            f"## Claim\n{statement_line}\n\n"
            f"## Assumptions\n{assumptions_block}\n\n"
            "## Proof Plan\n"
            "1. [state strategy]\n"
            "2. [reduce to known lemmas]\n"
            "3. [bound/error argument]\n"
            "4. [conclude]\n\n"
            "## Detailed Steps\n"
            "Step 1. [derive]\n"
            "Step 2. [derive]\n"
            "Step 3. [derive]\n\n"
            "## Conclusion\n"
            "Therefore, the claim holds under the stated assumptions.\n"
        )

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
    def _safe_id(claim_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]", "_", str(claim_id))

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _error(message: str) -> str:
        return json.dumps({"success": False, "error": message}, indent=2)
