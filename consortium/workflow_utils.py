"""
Shared helpers for graph routing, validation, and handoff context.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from datetime import datetime, timezone

from .supervision import (
    validate_claim_traceability,
    validate_cross_track_consistency,
    validate_math_acceptance,
    validate_paper_quality,
    validate_result_artifacts,
    validate_review_verdict,
    save_cross_track_report,
)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def safe_int_env(name: str, default: int) -> int:
    """Read an environment variable as a non-negative integer."""
    try:
        value = int(os.getenv(name, str(default)))
        return value if value >= 0 else default
    except Exception:
        return default


def safe_float_env(name: str, default: float) -> float:
    """Read an environment variable as a non-negative float."""
    try:
        value = float(os.getenv(name, str(default)))
        return value if value >= 0 else default
    except Exception:
        return default


def read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def build_context_message(state: dict) -> str:
    """Format agent outputs and validation state for manager prompts."""
    lines = [f"Task: {state.get('task', '')}\n"]

    agent_outputs = state.get("agent_outputs", {})
    if agent_outputs:
        lines.append("=== Previous agent outputs ===")
        for agent_name, output in agent_outputs.items():
            lines.append(f"\n--- {agent_name} ---\n{output}")
        lines.append("")

    interrupt = state.get("interrupt_instruction")
    if interrupt:
        lines.append(f"=== LIVE STEERING INSTRUCTION ===\n{interrupt}\n")

    validation = state.get("validation_results", {})
    if validation:
        lines.append("=== Validation results ===")
        for gate, result in validation.items():
            status = "PASS" if result.get("is_valid") else "FAIL"
            errors = "; ".join(result.get("errors", []))
            lines.append(f"  {gate}: {status}" + (f" — {errors}" if errors else ""))
        lines.append("")

    iteration = safe_int(state.get("iteration_count", 0), 0)
    max_steps = state.get("manager_max_steps", 50)
    lines.append(f"Iteration: {iteration}/{max_steps}")

    return "\n".join(lines)


def followup_decision_requires_loop(workspace_dir: str) -> tuple[bool, str]:
    """Return (required, reason) from followup_decision.json."""
    path = os.path.join(workspace_dir, "paper_workspace", "followup_decision.json")
    payload = read_json(path)
    if not payload:
        return False, "No followup_decision.json found."

    decision = str(payload.get("decision", "")).strip().lower()
    if decision == "followup_required":
        reason = payload.get("evidence_summary") or payload.get("blocking_issues") or []
        reason_text = str(reason[:3]) if isinstance(reason, list) else str(reason)
        return True, f"results_analysis requested follow-up: {reason_text}"

    followup_needed = payload.get("followup_needed")
    if isinstance(followup_needed, bool) and followup_needed:
        return True, "results_analysis set followup_needed=true"

    return False, "followup_not_required"


def build_required_artifacts(state: dict) -> list[str]:
    pipeline_mode = str(state.get("pipeline_mode", "default")).strip().lower()
    required = ["final_paper.tex"]
    if pipeline_mode == "full_research":
        required.extend([
            "paper_workspace/track_decomposition.json",
            "paper_workspace/literature_review.pdf",
            "paper_workspace/research_plan.pdf",
            "paper_workspace/results_assessment.pdf",
            "paper_workspace/followup_decision.json",
        ])
    if state.get("require_experiment_plan", False):
        required.append("experiments_to_run_later.md")
    if state.get("require_pdf", False):
        required.append("final_paper.pdf")
    if state.get("enforce_editorial_artifacts", False):
        required.extend([
            "paper_workspace/author_style_guide.md",
            "paper_workspace/intro_skeleton.tex",
            "paper_workspace/style_macros.tex",
            "paper_workspace/reader_contract.json",
            "paper_workspace/editorial_contract.md",
            "paper_workspace/theorem_map.json",
            "paper_workspace/revision_log.md",
            "paper_workspace/copyedit_report.tex",
            "paper_workspace/review_report.tex",
            "paper_workspace/review_verdict.json",
        ])
        if state.get("math_enabled", False):
            required.append("paper_workspace/claim_traceability.json")
    return required


def run_validation_gates(state: dict) -> dict:
    """
    Run all quality gates. Returns validation_results and gate_passed.
    """
    workspace = state.get("workspace_dir") or "."
    results: dict[str, dict] = {}
    all_valid = True

    enforce_paper = state.get("enforce_paper_artifacts", False)
    pipeline_mode = str(state.get("pipeline_mode", "default")).strip().lower()
    should_enforce = enforce_paper or (pipeline_mode == "full_research")

    if should_enforce:
        required = build_required_artifacts(state)
        summary = validate_result_artifacts(
            result="",
            workspace_dir=workspace,
            required_artifacts=required,
        )
        ok = not summary.get("missing_required_artifacts")
        results["artifact_gate"] = {
            "is_valid": ok,
            "errors": [
                "Missing: " + ", ".join(summary.get("missing_required_artifacts", []))
            ] if not ok else [],
        }
        all_valid = all_valid and ok

    if state.get("math_enabled", False):
        math_summary = validate_math_acceptance(workspace_dir=workspace)
        if math_summary.get("graph_present"):
            ok = math_summary.get("is_valid", True)
            results["math_acceptance"] = {
                "is_valid": ok,
                "errors": math_summary.get("errors", []),
            }
            all_valid = all_valid and ok

        if state.get("enforce_editorial_artifacts", False):
            traceability = validate_claim_traceability(workspace_dir=workspace)
            ok = traceability.get("is_valid", True)
            results["claim_traceability"] = {
                "is_valid": ok,
                "errors": traceability.get("errors", []),
            }
            all_valid = all_valid and ok

    if state.get("enforce_editorial_artifacts", False):
        min_score = state.get("min_review_score", 8)
        review = validate_review_verdict(workspace_dir=workspace, min_review_score=min_score)
        ok = review.get("is_valid", True)
        results["review_verdict"] = {
            "is_valid": ok,
            "errors": review.get("errors", []),
        }
        all_valid = all_valid and ok

        quality = validate_paper_quality(workspace_dir=workspace)
        ok = quality.get("is_valid", True)
        results["paper_quality"] = {
            "is_valid": ok,
            "errors": quality.get("errors", []),
        }
        all_valid = all_valid and ok

    return {"validation_results": results, "gate_passed": all_valid}


def choose_validation_retry_stage(
    validation_results: dict,
    stages: list[str],
) -> tuple[int, str]:
    """Pick a deterministic retry stage when final validation fails."""
    if not stages:
        return 0, "Validation failed; restarting from first stage."

    if "math_acceptance" in validation_results and "math_prover_agent" in stages:
        idx = stages.index("math_prover_agent")
        return idx, "Validation failed on math acceptance; rerouting to math prover."

    if "claim_traceability" in validation_results and "writeup_agent" in stages:
        idx = stages.index("writeup_agent")
        return idx, "Validation failed on claim traceability; rerouting to writeup."

    if "review_verdict" in validation_results and "writeup_agent" in stages:
        idx = stages.index("writeup_agent")
        return idx, "Validation failed on review verdict; rerouting to writeup."

    if "paper_quality" in validation_results and "writeup_agent" in stages:
        idx = stages.index("writeup_agent")
        return idx, "Validation failed on paper quality; rerouting to writeup."

    if "artifact_gate" in validation_results and "resource_preparation_agent" in stages:
        idx = stages.index("resource_preparation_agent")
        return idx, "Validation failed on missing artifacts; rerouting to resource preparation."

    if "writeup_agent" in stages:
        idx = stages.index("writeup_agent")
        return idx, "Validation failed; rerouting to writeup."

    return 0, "Validation failed; restarting from first stage."


# ---------------------------------------------------------------------------
# Rebuttal fix classification (Phase 1: data-gathering)
# ---------------------------------------------------------------------------

_EXPERIMENT_KEYWORDS = frozenset({
    "experiment", "rerun", "ablation", "baseline", "evaluation",
    "metric", "benchmark", "dataset", "training", "hyperparameter",
    "reproduce", "reproducibility", "run",
})

_THEORY_KEYWORDS = frozenset({
    "proof", "theorem", "lemma", "mathematical", "formal",
    "derivation", "bound", "convergence", "assumption", "rigor",
})


def run_intermediate_validation(
    state: dict,
    checkpoint_name: str,
) -> dict:
    """Run lightweight validation gates at intermediate pipeline points.

    Unlike the final ``run_validation_gates`` which blocks on failure, these
    checkpoints are *advisory*: results accumulate in
    ``state["intermediate_validation_log"]`` and surface as warnings in
    milestone reports.  They never block the pipeline.

    Supported checkpoints:
        post_theory   — run ``validate_math_acceptance`` if claims exist
        post_experiment — check experiment artifact existence
        post_merge    — run cross-track consistency check
        post_analysis — early ``validate_claim_traceability``

    Returns a dict suitable for merging into state:
        ``{"intermediate_validation_log": [...existing..., new_entry]}``
    """
    workspace = state.get("workspace_dir") or "."
    results: dict[str, dict] = {}

    if checkpoint_name == "post_theory":
        if state.get("math_enabled", False):
            math_summary = validate_math_acceptance(workspace_dir=workspace)
            if math_summary.get("graph_present"):
                results["math_acceptance"] = {
                    "is_valid": math_summary.get("is_valid", True),
                    "errors": math_summary.get("errors", []),
                }

    elif checkpoint_name == "post_experiment":
        # Check for key experiment artifacts
        exp_artifacts = [
            "paper_workspace/experiment_results.json",
            "paper_workspace/experiment_design.json",
        ]
        missing = [
            a for a in exp_artifacts
            if not os.path.exists(os.path.join(workspace, a))
        ]
        results["experiment_artifacts"] = {
            "is_valid": len(missing) == 0,
            "errors": [f"Missing: {', '.join(missing)}"] if missing else [],
        }

    elif checkpoint_name == "post_merge":
        # Cross-track consistency (only if both tracks ran)
        theory_ran = state.get("theory_track_status") in ("completed", "in_progress")
        experiment_ran = state.get("experiment_track_status") in ("completed", "in_progress")
        if theory_ran and experiment_ran:
            consistency = validate_cross_track_consistency(workspace_dir=workspace)
            save_cross_track_report(workspace, consistency)
            results["cross_track_consistency"] = {
                "is_valid": consistency["is_valid"],
                "errors": [
                    f"{i['severity']}: {i['description']}"
                    for i in consistency.get("inconsistencies", [])
                    if i.get("severity") in ("critical", "major")
                ],
            }

    elif checkpoint_name == "post_analysis":
        if state.get("math_enabled", False) and state.get("enforce_editorial_artifacts", False):
            traceability = validate_claim_traceability(workspace_dir=workspace)
            results["claim_traceability"] = {
                "is_valid": traceability.get("is_valid", True),
                "errors": traceability.get("errors", []),
            }

    # Build log entry
    entry = {
        "checkpoint": checkpoint_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }

    prev_log = list(state.get("intermediate_validation_log") or [])
    prev_log.append(entry)

    # Print summary
    for gate, result in results.items():
        status = "PASS" if result.get("is_valid") else "WARN"
        errors = result.get("errors", [])
        msg = f"[intermediate:{checkpoint_name}] {gate}: {status}"
        if errors:
            msg += f" ({'; '.join(errors[:2])})"
        print(msg)

    return {"intermediate_validation_log": prev_log}


def classify_review_fixes(workspace_dir: str) -> str:
    """Classify the type of fixes needed based on review_verdict.json.

    Returns one of: ``"experiment"``, ``"theory"``, ``"writeup"``.
    Priority: experiment > theory > writeup (most expensive fix type wins).
    """
    path = os.path.join(workspace_dir, "paper_workspace", "review_verdict.json")
    if not os.path.exists(path):
        return "writeup"

    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception:
        return "writeup"

    must_fix = payload.get("must_fix_actions", [])
    if not isinstance(must_fix, list):
        return "writeup"

    needs_experiment = False
    needs_theory = False

    for fix in must_fix:
        fix_type = str(fix.get("fix_type", "")).strip().lower()
        action = str(fix.get("action", "")).lower()
        target_files = [str(f).lower() for f in fix.get("target_files", [])]

        # Check explicit fix_type field first
        if fix_type == "experiment":
            needs_experiment = True
        elif fix_type == "theory":
            needs_theory = True
        else:
            # Heuristic classification from action text and target files
            combined = action + " " + " ".join(target_files)
            if any(kw in combined for kw in _EXPERIMENT_KEYWORDS):
                needs_experiment = True
            if any(kw in combined for kw in _THEORY_KEYWORDS):
                needs_theory = True

    if needs_experiment:
        return "experiment"
    if needs_theory:
        return "theory"
    return "writeup"
