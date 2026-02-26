"""
MathNumericalClaimVerifierTool - randomized numeric checks for scalar claim equalities.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from smolagents import Tool


class MathNumericalClaimVerifierTool(Tool):
    name = "math_numerical_claim_verifier_tool"
    description = """
    Numerically test scalar equality claims by random substitution.

    Provide lhs_expression and rhs_expression plus variable ranges as JSON.
    Example variable_ranges_json:
      {"m": [32, 1024], "n": [32, 1024], "eps": [1e-6, 1e-2]}

    This tool is a fast falsification/sanity check (not a formal proof).
    """

    inputs = {
        "lhs_expression": {
            "type": "string",
            "description": "Left-hand scalar expression (Python/math syntax)",
        },
        "rhs_expression": {
            "type": "string",
            "description": "Right-hand scalar expression",
        },
        "variable_ranges_json": {
            "type": "string",
            "description": "JSON object mapping variable to [min,max] or constant",
            "nullable": True,
        },
        "num_trials": {
            "type": "integer",
            "description": "Number of random trials (default: 64)",
            "nullable": True,
        },
        "tolerance": {
            "type": "number",
            "description": "Absolute/relative tolerance (default: 1e-6)",
            "nullable": True,
        },
        "claim_id": {
            "type": "string",
            "description": "Optional claim id for logging report",
            "nullable": True,
        },
        "save_report": {
            "type": "boolean",
            "description": "If true and claim_id provided, append report to checks/<claim_id>.jsonl",
            "nullable": True,
        },
        "workspace_subdir": {
            "type": "string",
            "description": "Workspace subdir root (default: math_workspace)",
            "nullable": True,
        },
    }

    output_type = "string"

    _ALLOWED_FUNCS = {
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "pi": math.pi,
        "e": math.e,
    }

    _RESERVED = set(_ALLOWED_FUNCS.keys()) | {
        "and",
        "or",
        "not",
        "if",
        "else",
        "for",
        "in",
        "True",
        "False",
        "None",
    }

    def __init__(self, working_dir: Optional[str] = None):
        super().__init__()
        self.working_dir = os.path.abspath(working_dir) if working_dir else None

    def forward(
        self,
        lhs_expression: str,
        rhs_expression: str,
        variable_ranges_json: Optional[str] = None,
        num_trials: Optional[int] = 64,
        tolerance: Optional[float] = 1e-6,
        claim_id: Optional[str] = None,
        save_report: Optional[bool] = True,
        workspace_subdir: Optional[str] = "math_workspace",
    ) -> str:
        try:
            lhs = (lhs_expression or "").strip()
            rhs = (rhs_expression or "").strip()
            if not lhs or not rhs:
                return self._error("lhs_expression and rhs_expression are required")

            n_trials = max(1, int(num_trials or 64))
            tol = float(tolerance if tolerance is not None else 1e-6)
            if tol < 0:
                tol = 1e-6

            ranges = self._parse_ranges(variable_ranges_json)
            vars_in_expr = self._extract_variables(lhs + " " + rhs)

            assignments_seen = []
            failures = []
            eval_errors = []
            max_abs_err = 0.0
            max_rel_err = 0.0

            for i in range(n_trials):
                assignment = self._sample_assignment(vars_in_expr, ranges)
                assignments_seen.append(assignment)
                try:
                    lv = float(self._safe_eval(lhs, assignment))
                    rv = float(self._safe_eval(rhs, assignment))
                except Exception as e:
                    eval_errors.append({"trial": i, "assignment": assignment, "error": str(e)})
                    continue

                if not math.isfinite(lv) or not math.isfinite(rv):
                    eval_errors.append(
                        {
                            "trial": i,
                            "assignment": assignment,
                            "error": f"non-finite values lhs={lv}, rhs={rv}",
                        }
                    )
                    continue

                abs_err = abs(lv - rv)
                denom = max(1.0, abs(lv), abs(rv))
                rel_err = abs_err / denom
                max_abs_err = max(max_abs_err, abs_err)
                max_rel_err = max(max_rel_err, rel_err)

                if abs_err > tol and rel_err > tol:
                    failures.append(
                        {
                            "trial": i,
                            "assignment": assignment,
                            "lhs": lv,
                            "rhs": rv,
                            "abs_err": abs_err,
                            "rel_err": rel_err,
                        }
                    )

            checked_trials = n_trials - len(eval_errors)
            verdict = len(failures) == 0 and checked_trials > 0

            result = {
                "success": True,
                "claim_id": claim_id,
                "lhs_expression": lhs,
                "rhs_expression": rhs,
                "num_trials": n_trials,
                "checked_trials": checked_trials,
                "tolerance": tol,
                "verdict": "pass" if verdict else "fail",
                "max_abs_err": max_abs_err,
                "max_rel_err": max_rel_err,
                "failure_count": len(failures),
                "eval_error_count": len(eval_errors),
                "failures": failures[:10],
                "eval_errors": eval_errors[:10],
            }

            if claim_id and bool(save_report):
                self._append_report(workspace_subdir or "math_workspace", claim_id, result)

            return json.dumps(result, indent=2)

        except Exception as e:
            return self._error(f"numerical claim verifier failed: {e}")

    @staticmethod
    def _extract_variables(text: str) -> List[str]:
        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text)
        vars_found = []
        for t in tokens:
            if t in MathNumericalClaimVerifierTool._RESERVED:
                continue
            if t not in vars_found:
                vars_found.append(t)
        return vars_found

    @staticmethod
    def _parse_ranges(raw: Optional[str]) -> Dict[str, Any]:
        if raw is None or not raw.strip():
            return {}
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("variable_ranges_json must be a JSON object")
        return value

    @staticmethod
    def _sample_assignment(variables: List[str], ranges: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for var in variables:
            spec = ranges.get(var, [0.1, 10.0])
            if isinstance(spec, list) and len(spec) == 2:
                lo = float(spec[0])
                hi = float(spec[1])
                if lo == hi:
                    out[var] = lo
                else:
                    if lo > hi:
                        lo, hi = hi, lo
                    out[var] = random.uniform(lo, hi)
            else:
                out[var] = float(spec)
        return out

    def _safe_eval(self, expr: str, assignment: Dict[str, float]) -> float:
        names = dict(self._ALLOWED_FUNCS)
        names.update(assignment)
        code = compile(expr, "<math_expr>", "eval")
        return eval(code, {"__builtins__": {}}, names)  # noqa: S307 - constrained env

    def _append_report(self, workspace_subdir: str, claim_id: str, payload: Dict[str, Any]) -> None:
        workspace_dir = self._resolve_workspace_dir(workspace_subdir)
        checks_dir = os.path.join(workspace_dir, "checks")
        os.makedirs(checks_dir, exist_ok=True)
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(claim_id))
        checks_path = os.path.join(checks_dir, f"{safe_id}.jsonl")

        report = {
            "kind": "numeric_check",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with open(checks_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(report) + "\n")

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
    def _error(message: str) -> str:
        return json.dumps({"success": False, "error": message}, indent=2)
