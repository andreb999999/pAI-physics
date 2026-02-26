"""
MathProofRigorCheckerTool - heuristic rigor checks for proof drafts.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from smolagents import Tool


class MathProofRigorCheckerTool(Tool):
    name = "math_proof_rigor_checker_tool"
    description = """
    Heuristically evaluate rigor/completeness of a proof draft.

    Checks include:
    - required structure (claim, assumptions, detailed steps, conclusion)
    - placeholder/TODO leakage
    - number of explicit proof steps
    - dependency mention coverage from claim_graph
    """

    inputs = {
        "claim_id": {
            "type": "string",
            "description": "Claim id to validate against claim_graph and proof file",
            "nullable": True,
        },
        "proof_text": {
            "type": "string",
            "description": "Proof text to check (if omitted, loads from proofs/<claim_id>.md)",
            "nullable": True,
        },
        "check_level": {
            "type": "string",
            "description": "basic or strict (default: strict)",
            "nullable": True,
        },
        "workspace_subdir": {
            "type": "string",
            "description": "Workspace subdir root (default: math_workspace)",
            "nullable": True,
        },
    }

    output_type = "string"

    def __init__(self, working_dir: Optional[str] = None):
        super().__init__()
        self.working_dir = os.path.abspath(working_dir) if working_dir else None

    def forward(
        self,
        claim_id: Optional[str] = None,
        proof_text: Optional[str] = None,
        check_level: str = "strict",
        workspace_subdir: Optional[str] = "math_workspace",
    ) -> str:
        try:
            check_level = (check_level or "strict").strip().lower()
            if check_level not in {"basic", "strict"}:
                return self._error("check_level must be 'basic' or 'strict'")

            workspace_dir = self._resolve_workspace_dir(workspace_subdir or "math_workspace")
            claim_graph = self._read_claim_graph(workspace_dir)
            claim = self._get_claim(claim_graph, claim_id) if claim_id else None

            text = (proof_text or "").strip()
            if not text and claim_id:
                proof_path = os.path.join(workspace_dir, "proofs", f"{self._safe_id(claim_id)}.md")
                if os.path.exists(proof_path):
                    with open(proof_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()

            if not text:
                return json.dumps(
                    {
                        "success": False,
                        "claim_id": claim_id,
                        "verdict": "fail",
                        "score": 0.0,
                        "critical_issues": ["proof text is empty"],
                    },
                    indent=2,
                )

            checks = self._run_checks(text, claim)
            score = self._score_checks(checks)
            threshold = 0.55 if check_level == "basic" else 0.75
            has_critical = len(checks["critical_issues"]) > 0
            verdict = "pass" if (score >= threshold and not has_critical) else "fail"

            return json.dumps(
                {
                    "success": True,
                    "claim_id": claim_id,
                    "check_level": check_level,
                    "threshold": threshold,
                    "score": round(score, 4),
                    "verdict": verdict,
                    **checks,
                },
                indent=2,
            )

        except Exception as e:
            return self._error(f"proof rigor checker failed: {e}")

    def _run_checks(self, text: str, claim: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        low = text.lower()
        sections = {
            "claim": bool(re.search(r"^##\s*claim\b", text, flags=re.IGNORECASE | re.MULTILINE)),
            "assumptions": bool(re.search(r"^##\s*assumptions\b", text, flags=re.IGNORECASE | re.MULTILINE)),
            "steps": bool(re.search(r"^##\s*(detailed\s+steps|proof\s+steps)\b", text, flags=re.IGNORECASE | re.MULTILINE)),
            "conclusion": bool(re.search(r"^##\s*conclusion\b", text, flags=re.IGNORECASE | re.MULTILINE)),
        }

        step_count = len(
            re.findall(
                r"^(?:\s*(?:step\s*\d+\.|\d+\.|-\s*step\s*\d+))",
                text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        )
        placeholder_hits = re.findall(r"\[(?:fill|todo|tbd)[^\]]*\]", low)

        dependency_mentions: Dict[str, bool] = {}
        if claim:
            for dep in claim.get("depends_on", []):
                dep = str(dep)
                dependency_mentions[dep] = dep.lower() in low

        assumption_mentions: Dict[str, bool] = {}
        if claim:
            for assumption in claim.get("assumptions", []):
                a = str(assumption).strip()
                if not a:
                    continue
                key = a[:80]
                assumption_mentions[key] = a.lower() in low

        critical_issues: List[str] = []
        warnings: List[str] = []

        for name, present in sections.items():
            if not present:
                warnings.append(f"missing section: {name}")

        if step_count < 2:
            critical_issues.append("proof has fewer than two explicit steps")
        elif step_count < 4:
            warnings.append("proof has low step granularity (<4 steps)")

        if placeholder_hits:
            critical_issues.append("proof still contains placeholders/TODOs")

        if dependency_mentions and not all(dependency_mentions.values()):
            missing = [k for k, v in dependency_mentions.items() if not v]
            warnings.append("dependency references not mentioned: " + ", ".join(missing))

        if assumption_mentions and not all(assumption_mentions.values()):
            missing = [k for k, v in assumption_mentions.items() if not v]
            warnings.append("assumptions not explicitly referenced: " + ", ".join(missing))

        return {
            "section_presence": sections,
            "step_count": step_count,
            "placeholder_hits": placeholder_hits,
            "dependency_mentions": dependency_mentions,
            "assumption_mentions": assumption_mentions,
            "critical_issues": critical_issues,
            "warnings": warnings,
        }

    @staticmethod
    def _score_checks(checks: Dict[str, Any]) -> float:
        score = 0.0
        sections = checks.get("section_presence", {})
        section_ratio = sum(1 for v in sections.values() if v) / max(1, len(sections))
        score += 0.4 * section_ratio

        steps = int(checks.get("step_count", 0))
        score += 0.3 * min(1.0, steps / 6.0)

        dep = checks.get("dependency_mentions", {})
        if dep:
            dep_ratio = sum(1 for v in dep.values() if v) / len(dep)
        else:
            dep_ratio = 1.0
        score += 0.15 * dep_ratio

        asm = checks.get("assumption_mentions", {})
        if asm:
            asm_ratio = sum(1 for v in asm.values() if v) / len(asm)
        else:
            asm_ratio = 1.0
        score += 0.15 * asm_ratio

        if checks.get("placeholder_hits"):
            score -= 0.25
        if checks.get("critical_issues"):
            score -= 0.2

        return max(0.0, min(1.0, score))

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
    def _read_claim_graph(workspace_dir: str) -> Dict[str, Any]:
        path = os.path.join(workspace_dir, "claim_graph.json")
        if not os.path.exists(path):
            return {"claims": []}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _get_claim(graph: Dict[str, Any], claim_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not claim_id:
            return None
        for claim in graph.get("claims", []):
            if str(claim.get("id")) == str(claim_id):
                return claim
        return None

    @staticmethod
    def _error(message: str) -> str:
        return json.dumps({"success": False, "error": message}, indent=2)
