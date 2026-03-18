"""
ManagerAgent — deterministic stage router for LangGraph.

This manager no longer lets the LLM choose the next specialist. Instead, the
next stage is selected from a fixed pipeline order (`pipeline_stages`) and the
LLM is only used to craft a high-quality task description for that stage.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, List, Optional

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from ..prompts.manager_instructions import get_manager_system_prompt
from ..supervision import (
    validate_review_verdict,
)
from ..toolkits.filesystem.file_editing.file_editing_tools import (
    CreateFileWithContent, DeleteFileOrFolder, ListDir, ModifyFile, SearchKeyword, SeeFile,
)
from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool
from ..workflow_utils import (
    build_context_message as _shared_build_context_message,
    build_required_artifacts as _shared_build_required_artifacts,
    choose_validation_retry_stage as _shared_choose_validation_retry_stage,
    followup_decision_requires_loop as _shared_followup_decision_requires_loop,
    read_json as _shared_read_json,
    run_validation_gates as _shared_run_validation_gates,
    safe_int as _shared_safe_int,
)

try:
    from ..toolkits.search.text_inspector.text_inspector_tool import TextInspectorTool
    _TEXT_INSPECTOR_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _TEXT_INSPECTOR_AVAILABLE = False

try:
    from ..toolkits.math.claim_graph_tool import MathClaimGraphTool
    _MATH_TOOLS_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _MATH_TOOLS_AVAILABLE = False


_AGENT_TASK_RE = re.compile(r"AGENT_TASK\s*:\s*(.*)", re.DOTALL | re.IGNORECASE)


def _safe_int(value: Any, default: int) -> int:
    return _shared_safe_int(value, default)


def _extract_agent_task(text: str, next_agent: str) -> str:
    """Parse AGENT_TASK block from manager response."""
    if not isinstance(text, str):
        text = str(text or "")
    m = _AGENT_TASK_RE.search(text)
    if m:
        task = m.group(1).strip()
        if task:
            return task
    stripped = text.strip()
    if stripped:
        return stripped
    return (
        f"Continue with `{next_agent}`. Inspect current workspace artifacts, "
        f"execute the stage thoroughly, and produce all required outputs."
    )


def _build_context_message(state: dict) -> str:
    return _shared_build_context_message(state)


def _read_json(path: str) -> Optional[dict]:
    return _shared_read_json(path)


def _followup_decision_requires_loop(workspace_dir: str) -> tuple[bool, str]:
    return _shared_followup_decision_requires_loop(workspace_dir)


def _apply_followup_loop_if_needed(
    state: dict,
    stages: list[str],
    stage_index: int,
) -> tuple[int, int, Optional[str]]:
    """
    If results analysis requested follow-up, jump back to experimentation stage.
    """
    if stage_index <= 0 or not stages:
        return stage_index, _safe_int(state.get("followup_iteration", 0), 0), None

    last_stage = stages[min(stage_index - 1, len(stages) - 1)]
    if last_stage != "results_analysis_agent":
        return stage_index, _safe_int(state.get("followup_iteration", 0), 0), None

    workspace = state.get("workspace_dir") or "."
    required, reason = _followup_decision_requires_loop(workspace)
    if not required:
        return stage_index, _safe_int(state.get("followup_iteration", 0), 0), None

    followup_iter = _safe_int(state.get("followup_iteration", 0), 0)
    followup_max = max(0, _safe_int(state.get("followup_max_iterations", 0), 0))
    if followup_iter >= followup_max:
        return (
            stage_index,
            followup_iter,
            f"Follow-up requested but loop limit reached ({followup_iter}/{followup_max}).",
        )

    if "experimentation_agent" not in stages:
        return stage_index, followup_iter, "Follow-up requested but experimentation stage is unavailable."

    exp_idx = stages.index("experimentation_agent")
    return (
        exp_idx,
        followup_iter + 1,
        f"Looping back to experimentation for follow-up ({followup_iter + 1}/{followup_max}): {reason}",
    )


def _apply_reviewer_loop_if_needed(
    state: dict,
    stages: list[str],
    stage_index: int,
) -> tuple[int, Optional[str]]:
    """
    If reviewer score is below threshold in editorial mode, route back to writeup.
    """
    if stage_index <= 0 or not stages:
        return stage_index, None

    last_stage = stages[min(stage_index - 1, len(stages) - 1)]
    if last_stage != "reviewer_agent":
        return stage_index, None

    if not state.get("enforce_editorial_artifacts", False):
        return stage_index, None

    workspace = state.get("workspace_dir") or "."
    min_score = _safe_int(state.get("min_review_score", 8), 8)
    rv = validate_review_verdict(workspace_dir=workspace, min_review_score=min_score)
    score = rv.get("overall_score")
    if score is None:
        return stage_index, None

    if score < min_score and "writeup_agent" in stages:
        writeup_idx = stages.index("writeup_agent")
        return (
            writeup_idx,
            f"Reviewer score {score} is below threshold {min_score}; rerouting to writeup loop.",
        )

    return stage_index, None


def _choose_validation_retry_stage(
    validation_results: dict,
    stages: list[str],
) -> tuple[int, str]:
    return _shared_choose_validation_retry_stage(validation_results, stages)


# ---------------------------------------------------------------------------
# Validation gate
# ---------------------------------------------------------------------------

def _run_validation_gates(state: dict) -> dict:
    return _shared_run_validation_gates(state)


def _build_required_artifacts(state: dict) -> list[str]:
    return _shared_build_required_artifacts(state)


# ---------------------------------------------------------------------------
# Tool builder
# ---------------------------------------------------------------------------

def get_tools(workspace_dir: Optional[str], model_id: str, enable_math_agents: bool = False) -> list:
    tools = [VLMDocumentAnalysisTool(model=model_id, working_dir=workspace_dir)]

    if enable_math_agents and _MATH_TOOLS_AVAILABLE:
        tools.append(MathClaimGraphTool(
            working_dir=workspace_dir,
            allow_accepted_transition=True,
        ))

    enable_text_inspector = os.getenv(
        "CONSORTIUM_ENABLE_MANAGER_TEXT_INSPECTOR", "1"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if enable_text_inspector and _TEXT_INSPECTOR_AVAILABLE:
        try:
            tools.append(TextInspectorTool(model=model_id, working_dir=workspace_dir))
        except Exception as e:
            print(f"TextInspectorTool disabled: {e}")

    if workspace_dir:
        tools += [
            SeeFile(working_dir=workspace_dir),
            CreateFileWithContent(working_dir=workspace_dir),
            ModifyFile(working_dir=workspace_dir),
            ListDir(working_dir=workspace_dir),
            SearchKeyword(working_dir=workspace_dir),
            DeleteFileOrFolder(working_dir=workspace_dir),
        ]
    return tools


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def build_node(
    model: Any,
    workspace_dir: Optional[str],
    pipeline_mode: str = "full_research",
    pipeline_stages: Optional[List[str]] = None,
    enable_math_agents: bool = False,
    followup_max_iterations: int = 3,
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    from ..toolkits.model_utils import get_raw_model
    model_id = get_raw_model(model)

    tools = get_tools(workspace_dir, model_id, enable_math_agents=enable_math_agents)
    system_prompt = get_manager_system_prompt(
        tools=tools,
        managed_agents=None,
    )
    from .base_agent import _unwrap_model
    # Budget is now recorded automatically by the monkey-patched litellm.completion()
    react_agent = create_react_agent(
        model=_unwrap_model(model),
        tools=tools,
        prompt=system_prompt,
    )

    if workspace_dir:
        os.makedirs(os.path.join(workspace_dir, "inter_agent_messages"), exist_ok=True)

    default_stages = list(pipeline_stages or [])

    def _invoke_for_handoff_task(state: dict, next_agent: str, notes: list[str]) -> str:
        context = _build_context_message(state)
        notes_block = "\n".join(f"- {n}" for n in notes) if notes else "- none"
        prompt = (
            f"{context}\n\n"
            "You are preparing the handoff for the next fixed pipeline stage.\n"
            f"NEXT_STAGE_AGENT: {next_agent}\n"
            f"SYSTEM_NOTES:\n{notes_block}\n\n"
            "Return ONLY this format:\n"
            "AGENT_TASK:\n"
            f"<detailed task for {next_agent}>\n"
        )
        result = react_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        last_msg = result["messages"][-1] if result.get("messages") else None
        last_content = last_msg.content if last_msg and hasattr(last_msg, "content") else ""
        return _extract_agent_task(last_content, next_agent)

    def manager_node(state: dict) -> dict:
        stages = state.get("pipeline_stages")
        if not isinstance(stages, list) or not stages:
            stages = list(default_stages)

        if not stages:
            return {
                "finished": True,
                "current_agent": "FINISH",
                "agent_task": "No pipeline stages configured.",
            }

        stage_index = _safe_int(state.get("pipeline_stage_index", 0), 0)
        followup_iteration = _safe_int(state.get("followup_iteration", 0), 0)
        notes: list[str] = []

        stage_index, followup_iteration, followup_note = _apply_followup_loop_if_needed(
            state, stages, stage_index
        )
        if followup_note:
            notes.append(followup_note)

        stage_index, reviewer_note = _apply_reviewer_loop_if_needed(
            state, stages, stage_index
        )
        if reviewer_note:
            notes.append(reviewer_note)

        new_iteration = _safe_int(state.get("iteration_count", 0), 0) + 1

        # All deterministic stages completed: run final validations.
        if stage_index >= len(stages):
            val = _run_validation_gates(state)
            if not val["gate_passed"]:
                error_msg = "VALIDATION FAILED:\n" + "\n".join(
                    f"  {gate}: {'; '.join(r['errors'])}"
                    for gate, r in val["validation_results"].items()
                    if not r["is_valid"]
                )
                retry_idx, retry_note = _choose_validation_retry_stage(
                    val["validation_results"], stages
                )
                notes.append(error_msg)
                notes.append(retry_note)
                next_agent = stages[retry_idx]
                agent_task = _invoke_for_handoff_task(state, next_agent, notes)
                return {
                    "validation_results": val["validation_results"],
                    "current_agent": next_agent,
                    "agent_task": agent_task,
                    "finished": False,
                    "iteration_count": new_iteration,
                    "pipeline_stages": stages,
                    "pipeline_stage_index": retry_idx + 1,
                    "followup_iteration": followup_iteration,
                    "interrupt_instruction": None,
                }
            return {
                "validation_results": val["validation_results"],
                "current_agent": "FINISH",
                "agent_task": "All deterministic stages completed and validation gates passed.",
                "finished": True,
                "iteration_count": new_iteration,
                "pipeline_stages": stages,
                "pipeline_stage_index": stage_index,
                "followup_iteration": followup_iteration,
            }

        next_agent = stages[stage_index]
        agent_task = _invoke_for_handoff_task(state, next_agent, notes)

        return {
            "current_agent": next_agent,
            "agent_task": agent_task,
            "finished": False,
            "iteration_count": new_iteration,
            "pipeline_stages": stages,
            "pipeline_stage_index": stage_index + 1,
            "followup_iteration": followup_iteration,
            # Clear interrupt once processed
            "interrupt_instruction": None,
        }

    manager_node.__name__ = "manager"
    return manager_node
