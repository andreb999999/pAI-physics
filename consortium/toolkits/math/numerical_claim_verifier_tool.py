"""
MathNumericalClaimVerifierTool - numeric checks for scalar/matrix/rate/bound claims.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, ConfigDict

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency fallback
    np = None


class MathNumericalClaimVerifierToolInput(BaseModel):
    lhs_expression: str = Field(description="Left-hand scalar expression (Python/math syntax). Required for expression mode.")
    rhs_expression: str = Field(description="Right-hand scalar expression. Required for expression mode.")
    variable_ranges_json: Optional[str] = Field(default=None, description="JSON object mapping variable to [min,max] or constant")
    num_trials: Optional[int] = Field(default=None, description="Number of random trials (default: 64)")
    tolerance: Optional[float] = Field(default=None, description="Absolute/relative tolerance (default: 1e-6)")
    claim_id: Optional[str] = Field(default=None, description="Optional claim id for logging report")
    save_report: Optional[bool] = Field(default=None, description="If true and claim_id provided, append report to checks/<claim_id>.jsonl")
    workspace_subdir: Optional[str] = Field(default=None, description="Workspace subdir root (default: math_workspace)")
    verification_mode: Optional[str] = Field(default=None, description="expression|matrix|convergence|bound (default: expression)")
    matrix_lhs_json: Optional[str] = Field(default=None, description="JSON-encoded matrix/tensor for matrix mode")
    matrix_rhs_json: Optional[str] = Field(default=None, description="Optional JSON-encoded matrix/tensor target for matrix mode")
    matrix_norm: Optional[str] = Field(default=None, description="Norm type for matrix mode: spectral|fro|nuclear|inf|max")
    convergence_values_json: Optional[str] = Field(default=None, description="JSON array of nonnegative values over iterations for convergence mode")
    expected_rate: Optional[str] = Field(default=None, description="Expected asymptotic rate, e.g. O(1/T), O(1/sqrt(T)), O(log(T)/T)")
    bound_observed_json: Optional[str] = Field(default=None, description="JSON array of observed values for bound mode")
    bound_reference_json: Optional[str] = Field(default=None, description="JSON array (or single-value array) of reference bounds for bound mode")


class MathNumericalClaimVerifierTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "math_numerical_claim_verifier_tool"
    description: str = """
    Numerically test claims in multiple modes:
    - expression: scalar equalities/inequalities by random substitution
    - matrix: matrix/tensor norm-based checks
    - convergence: estimate empirical convergence-rate slope
    - bound: verify observed values stay below reference bounds

    This tool is a falsification/sanity check, not formal proof.
    """
    args_schema: Type[BaseModel] = MathNumericalClaimVerifierToolInput
    working_dir: Optional[str] = None

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
    _CLAIM_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

    def __init__(self, working_dir: Optional[str] = None, **kwargs: Any):
        super().__init__(working_dir=os.path.abspath(working_dir) if working_dir else None, **kwargs)

    def _run(
        self,
        lhs_expression: str,
        rhs_expression: str,
        variable_ranges_json: Optional[str] = None,
        num_trials: Optional[int] = 64,
        tolerance: Optional[float] = 1e-6,
        claim_id: Optional[str] = None,
        save_report: Optional[bool] = True,
        workspace_subdir: Optional[str] = "math_workspace",
        verification_mode: Optional[str] = "expression",
        matrix_lhs_json: Optional[str] = None,
        matrix_rhs_json: Optional[str] = None,
        matrix_norm: Optional[str] = "spectral",
        convergence_values_json: Optional[str] = None,
        expected_rate: Optional[str] = None,
        bound_observed_json: Optional[str] = None,
        bound_reference_json: Optional[str] = None,
    ) -> str:
        try:
            mode = (verification_mode or "expression").strip().lower()
            tol = float(tolerance if tolerance is not None else 1e-6)
            if tol < 0:
                tol = 1e-6

            if mode == "expression":
                result = self._verify_expression(
                    lhs_expression=lhs_expression,
                    rhs_expression=rhs_expression,
                    variable_ranges_json=variable_ranges_json,
                    num_trials=num_trials,
                    tolerance=tol,
                    claim_id=claim_id,
                )
            elif mode == "matrix":
                result = self._verify_matrix(
                    matrix_lhs_json=matrix_lhs_json,
                    matrix_rhs_json=matrix_rhs_json,
                    rhs_expression=rhs_expression,
                    matrix_norm=(matrix_norm or "spectral"),
                    tolerance=tol,
                    claim_id=claim_id,
                )
            elif mode == "convergence":
                result = self._verify_convergence(
                    convergence_values_json=convergence_values_json,
                    expected_rate=expected_rate,
                    tolerance=tol,
                    claim_id=claim_id,
                )
            elif mode == "bound":
                result = self._verify_bound(
                    observed_json=bound_observed_json,
                    reference_json=bound_reference_json,
                    tolerance=tol,
                    claim_id=claim_id,
                )
            else:
                return self._error(
                    "verification_mode must be one of: expression, matrix, convergence, bound"
                )

            result["verification_mode"] = mode

            if claim_id and bool(save_report):
                if not self._CLAIM_ID_RE.match(str(claim_id)):
                    return self._error(
                        f"invalid claim_id '{claim_id}'. Allowed pattern: [A-Za-z0-9_.-]+ "
                        "(prevents checks filename collisions)."
                    )
                self._append_report(workspace_subdir or "math_workspace", claim_id, result)

            return json.dumps(result, indent=2)
        except Exception as e:
            return self._error(f"numerical claim verifier failed: {e}")


    def _verify_expression(
        self,
        lhs_expression: str,
        rhs_expression: str,
        variable_ranges_json: Optional[str],
        num_trials: Optional[int],
        tolerance: float,
        claim_id: Optional[str],
    ) -> Dict[str, Any]:
        lhs = (lhs_expression or "").strip()
        rhs = (rhs_expression or "").strip()
        if not lhs or not rhs:
            return {
                "success": False,
                "claim_id": claim_id,
                "verdict": "fail",
                "error": "lhs_expression and rhs_expression are required for expression mode",
            }

        n_trials = max(1, int(num_trials or 64))
        ranges = self._parse_ranges(variable_ranges_json)
        vars_in_expr = self._extract_variables(lhs + " " + rhs)

        failures = []
        eval_errors = []
        max_abs_err = 0.0
        max_rel_err = 0.0

        for i in range(n_trials):
            assignment = self._sample_assignment(vars_in_expr, ranges)
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

            if abs_err > tolerance and rel_err > tolerance:
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

        return {
            "success": True,
            "claim_id": claim_id,
            "lhs_expression": lhs,
            "rhs_expression": rhs,
            "num_trials": n_trials,
            "checked_trials": checked_trials,
            "tolerance": tolerance,
            "verdict": "pass" if verdict else "fail",
            "max_abs_err": max_abs_err,
            "max_rel_err": max_rel_err,
            "failure_count": len(failures),
            "eval_error_count": len(eval_errors),
            "failures": failures[:10],
            "eval_errors": eval_errors[:10],
        }

    def _verify_matrix(
        self,
        matrix_lhs_json: Optional[str],
        matrix_rhs_json: Optional[str],
        rhs_expression: str,
        matrix_norm: str,
        tolerance: float,
        claim_id: Optional[str],
    ) -> Dict[str, Any]:
        if np is None:
            return {
                "success": False,
                "claim_id": claim_id,
                "verdict": "fail",
                "error": "numpy is required for matrix mode",
            }
        if not matrix_lhs_json:
            return {
                "success": False,
                "claim_id": claim_id,
                "verdict": "fail",
                "error": "matrix_lhs_json is required for matrix mode",
            }

        lhs = np.array(json.loads(matrix_lhs_json), dtype=float)
        if lhs.size == 0:
            return {
                "success": False,
                "claim_id": claim_id,
                "verdict": "fail",
                "error": "matrix_lhs_json produced empty array",
            }

        norm_name = (matrix_norm or "spectral").strip().lower()
        if matrix_rhs_json:
            rhs = np.array(json.loads(matrix_rhs_json), dtype=float)
            if rhs.shape != lhs.shape:
                return {
                    "success": False,
                    "claim_id": claim_id,
                    "verdict": "fail",
                    "error": f"shape mismatch lhs={lhs.shape}, rhs={rhs.shape}",
                }
            diff = lhs - rhs
            diff_norm = self._matrix_norm(diff, norm_name)
            verdict = diff_norm <= tolerance
            return {
                "success": True,
                "claim_id": claim_id,
                "verdict": "pass" if verdict else "fail",
                "matrix_norm": norm_name,
                "lhs_shape": list(lhs.shape),
                "rhs_shape": list(rhs.shape),
                "difference_norm": diff_norm,
                "tolerance": tolerance,
            }

        if not rhs_expression:
            return {
                "success": False,
                "claim_id": claim_id,
                "verdict": "fail",
                "error": "Provide either matrix_rhs_json or rhs_expression (scalar bound) for matrix mode",
            }

        try:
            bound = float(rhs_expression)
        except Exception:
            return {
                "success": False,
                "claim_id": claim_id,
                "verdict": "fail",
                "error": "rhs_expression must be a numeric scalar bound in matrix mode without matrix_rhs_json",
            }

        lhs_norm = self._matrix_norm(lhs, norm_name)
        verdict = lhs_norm <= bound + tolerance
        return {
            "success": True,
            "claim_id": claim_id,
            "verdict": "pass" if verdict else "fail",
            "matrix_norm": norm_name,
            "lhs_shape": list(lhs.shape),
            "lhs_norm": lhs_norm,
            "bound": bound,
            "tolerance": tolerance,
            "violation": max(0.0, lhs_norm - bound),
        }

    def _verify_convergence(
        self,
        convergence_values_json: Optional[str],
        expected_rate: Optional[str],
        tolerance: float,
        claim_id: Optional[str],
    ) -> Dict[str, Any]:
        if np is None:
            return {
                "success": False,
                "claim_id": claim_id,
                "verdict": "fail",
                "error": "numpy is required for convergence mode",
            }
        if not convergence_values_json:
            return {
                "success": False,
                "claim_id": claim_id,
                "verdict": "fail",
                "error": "convergence_values_json is required for convergence mode",
            }

        values = np.array(json.loads(convergence_values_json), dtype=float).reshape(-1)
        if values.size < 5:
            return {
                "success": False,
                "claim_id": claim_id,
                "verdict": "fail",
                "error": "convergence_values_json must contain at least 5 values",
            }
        if np.any(~np.isfinite(values)):
            return {
                "success": False,
                "claim_id": claim_id,
                "verdict": "fail",
                "error": "convergence_values_json contains non-finite values",
            }

        ts = np.arange(1, values.size + 1, dtype=float)
        positive = np.maximum(np.abs(values), 1e-12)
        x = np.log(ts)
        y = np.log(positive)
        slope = float(np.polyfit(x, y, 1)[0])

        monotone_violations = int(np.sum(values[1:] > values[:-1] + tolerance))
        verdict = monotone_violations <= max(1, int(0.05 * (values.size - 1)))

        target_slope = self._expected_rate_slope(expected_rate)
        slope_pass = None
        if target_slope is not None:
            slope_pass = slope <= (target_slope + 0.15)
            verdict = verdict and slope_pass

        return {
            "success": True,
            "claim_id": claim_id,
            "verdict": "pass" if verdict else "fail",
            "series_length": int(values.size),
            "estimated_loglog_slope": slope,
            "expected_rate": expected_rate or "",
            "target_slope": target_slope,
            "slope_pass": slope_pass,
            "monotone_violations": monotone_violations,
            "initial_value": float(values[0]),
            "final_value": float(values[-1]),
            "improvement_ratio": float((values[0] + 1e-12) / (values[-1] + 1e-12)),
        }

    def _verify_bound(
        self,
        observed_json: Optional[str],
        reference_json: Optional[str],
        tolerance: float,
        claim_id: Optional[str],
    ) -> Dict[str, Any]:
        if observed_json is None or reference_json is None:
            return {
                "success": False,
                "claim_id": claim_id,
                "verdict": "fail",
                "error": "bound_observed_json and bound_reference_json are required for bound mode",
            }

        observed = [float(x) for x in json.loads(observed_json)]
        reference = [float(x) for x in json.loads(reference_json)]
        if not observed:
            return {
                "success": False,
                "claim_id": claim_id,
                "verdict": "fail",
                "error": "observed list is empty",
            }
        if len(reference) == 1:
            reference = reference * len(observed)
        if len(observed) != len(reference):
            return {
                "success": False,
                "claim_id": claim_id,
                "verdict": "fail",
                "error": "observed and reference lengths mismatch",
            }

        violations = []
        max_violation = 0.0
        for idx, (o, b) in enumerate(zip(observed, reference)):
            v = o - b
            if v > tolerance:
                violations.append({"index": idx, "observed": o, "bound": b, "violation": v})
                max_violation = max(max_violation, v)

        verdict = len(violations) == 0
        return {
            "success": True,
            "claim_id": claim_id,
            "verdict": "pass" if verdict else "fail",
            "tolerance": tolerance,
            "num_points": len(observed),
            "violation_count": len(violations),
            "max_violation": max_violation,
            "violations": violations[:20],
        }

    @staticmethod
    def _expected_rate_slope(expected_rate: Optional[str]) -> Optional[float]:
        if not expected_rate:
            return None
        rate = expected_rate.replace(" ", "").lower()
        if "o(1/t)" in rate or "o(1/n)" in rate:
            return -1.0
        if "o(1/sqrt(t))" in rate or "o(1/sqrt(n))" in rate:
            return -0.5
        if "o(log(t)/t)" in rate or "o(log(n)/n)" in rate:
            return -1.0
        return None

    @staticmethod
    def _matrix_norm(arr: "np.ndarray", norm_name: str) -> float:
        mode = (norm_name or "spectral").lower()
        if mode in {"spectral", "op", "operator", "2"}:
            return float(np.linalg.norm(arr, 2))
        if mode in {"fro", "frob", "frobenius"}:
            return float(np.linalg.norm(arr, "fro"))
        if mode in {"nuclear", "nuc"}:
            return float(np.linalg.norm(arr, "nuc"))
        if mode in {"inf", "infinity"}:
            return float(np.linalg.norm(arr, np.inf))
        if mode in {"max"}:
            return float(np.max(np.abs(arr)))
        raise ValueError(f"unsupported matrix_norm: {norm_name}")

    @staticmethod
    def _extract_variables(text: str) -> List[str]:
        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text)
        vars_found = []
        for token in tokens:
            if token in MathNumericalClaimVerifierTool._RESERVED:
                continue
            if token not in vars_found:
                vars_found.append(token)
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
