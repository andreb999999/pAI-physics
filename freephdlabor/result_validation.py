"""
Final-result artifact validation helpers.

These helpers are used to:
- prevent reporting artifacts that do not exist
- compute run completeness against required deliverables
"""

from __future__ import annotations

import ast
import json
import os
from typing import Any, Dict, List, Optional


def _as_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return []


def parse_result_payload(result: Any) -> Optional[Dict[str, Any]]:
    """
    Parse a final agent result into a dictionary when possible.

    Supports:
    - dict objects
    - JSON strings
    - Python-literal dict strings
    """
    if isinstance(result, dict):
        return dict(result)
    if not isinstance(result, str):
        return None

    text = result.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return None


def artifact_exists(artifact: str, workspace_dir: str) -> bool:
    """
    Check whether an artifact path exists.
    Relative paths are resolved under workspace_dir.
    """
    if not artifact:
        return False

    if os.path.isabs(artifact):
        return os.path.exists(artifact)

    return os.path.exists(os.path.join(workspace_dir, artifact))


def validate_result_artifacts(
    result: Any,
    workspace_dir: str,
    required_artifacts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate final-result artifacts against filesystem state.
    """
    payload = parse_result_payload(result)
    artifacts = _as_list(payload.get("artifacts")) if payload else []
    required = required_artifacts or []

    existing_artifacts = [a for a in artifacts if artifact_exists(a, workspace_dir)]
    missing_artifacts = [a for a in artifacts if not artifact_exists(a, workspace_dir)]
    missing_required_artifacts = [
        a for a in required if not artifact_exists(a, workspace_dir)
    ]

    return {
        "payload": payload,
        "artifacts": artifacts,
        "existing_artifacts": existing_artifacts,
        "missing_artifacts": missing_artifacts,
        "required_artifacts": required,
        "missing_required_artifacts": missing_required_artifacts,
        "is_complete": len(missing_required_artifacts) == 0,
    }


def sanitize_result_payload(
    result: Any,
    workspace_dir: str,
    required_artifacts: Optional[List[str]] = None,
) -> Any:
    """
    Return a sanitized result with truthful artifact reporting when parseable.
    """
    original_payload = parse_result_payload(result)
    summary = validate_result_artifacts(
        result=result,
        workspace_dir=workspace_dir,
        required_artifacts=required_artifacts,
    )
    payload = summary["payload"]
    if payload is None:
        return result
    payload = dict(payload)

    if "artifacts" in payload:
        payload["artifacts"] = summary["existing_artifacts"]
        if summary["missing_artifacts"]:
            payload["missing_artifacts"] = summary["missing_artifacts"]

    if summary["required_artifacts"]:
        payload["status"] = "complete" if summary["is_complete"] else "incomplete"
        if summary["missing_required_artifacts"]:
            payload["missing_required_artifacts"] = summary["missing_required_artifacts"]

    # Preserve original string answers when no semantic change was needed.
    if isinstance(result, str) and payload == (original_payload or {}):
        return result

    return payload
