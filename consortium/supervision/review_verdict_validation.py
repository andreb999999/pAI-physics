"""
Validation helpers for strict reviewer verdict gating.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def validate_review_verdict(workspace_dir: str, min_review_score: int = 8) -> Dict[str, Any]:
    """
    Validate reviewer verdict for strict paper-quality runs.

    Expected file:
      paper_workspace/review_verdict.json

    Required gates:
    - overall_score >= min_review_score
    - ai_voice_risk != high
    - hard_blockers is empty
    - intro_compliance has:
      - has_questions == true
      - has_takeaways == true
      - questions_answered == true
      - takeaways_supported == true
    """
    path = os.path.join(workspace_dir, "paper_workspace", "review_verdict.json")
    errors: List[str] = []

    if not os.path.exists(path):
        return {
            "present": False,
            "is_valid": False,
            "errors": ["missing paper_workspace/review_verdict.json"],
            "overall_score": None,
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            verdict = json.load(f)
    except Exception as e:
        return {
            "present": True,
            "is_valid": False,
            "errors": [f"failed to parse review_verdict.json: {e}"],
            "overall_score": None,
        }

    if not isinstance(verdict, dict):
        return {
            "present": True,
            "is_valid": False,
            "errors": ["review_verdict.json must be a JSON object"],
            "overall_score": None,
        }

    score_raw = verdict.get("overall_score")
    try:
        score = int(score_raw)
    except Exception:
        score = None
        errors.append(f"invalid overall_score: {score_raw!r}")

    if score is not None and score < int(min_review_score):
        errors.append(
            f"overall_score below threshold: {score} < {int(min_review_score)}"
        )

    ai_voice_risk = str(verdict.get("ai_voice_risk", "")).strip().lower()
    if ai_voice_risk == "high":
        errors.append("ai_voice_risk is high")

    hard_blockers = verdict.get("hard_blockers", [])
    if not isinstance(hard_blockers, list):
        errors.append("hard_blockers must be a list")
    elif len(hard_blockers) > 0:
        errors.append(f"hard_blockers present: {len(hard_blockers)}")

    intro = verdict.get("intro_compliance", {})
    if not isinstance(intro, dict):
        errors.append("intro_compliance must be an object")
        intro = {}

    required_intro_flags = [
        "has_questions",
        "has_takeaways",
        "questions_answered",
        "takeaways_supported",
    ]
    for key in required_intro_flags:
        if not _to_bool(intro.get(key)):
            errors.append(f"intro_compliance failed: {key} != true")

    # Validate must_fix_actions structure for downstream routing
    must_fix = verdict.get("must_fix_actions", [])
    if must_fix is not None and not isinstance(must_fix, list):
        errors.append("must_fix_actions must be a list")
    elif isinstance(must_fix, list):
        for i, fix in enumerate(must_fix):
            if not isinstance(fix, dict):
                errors.append(f"must_fix_actions[{i}] must be an object")
            elif "fix_type" not in fix:
                errors.append(f"must_fix_actions[{i}] missing fix_type field")

    return {
        "present": True,
        "is_valid": len(errors) == 0,
        "errors": errors,
        "overall_score": score,
        "ai_voice_risk": ai_voice_risk,
        "hard_blockers_count": len(hard_blockers) if isinstance(hard_blockers, list) else None,
    }

