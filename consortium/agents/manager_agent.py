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
    validate_claim_traceability,
    validate_math_acceptance,
    validate_paper_quality,
    validate_result_artifacts,
    validate_review_verdict,
)
from ..toolkits.filesystem.file_editing.file_editing_tools import (
    CreateFileWithContent, DeleteFileOrFolder, ListDir, ModifyFile, SearchKeyword, SeeFile,
)
from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool

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
    try:
        return int(value)
    except Exception:
        return default


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
    """Format agent outputs and iteration info for the manager prompt."""
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

    iteration = _safe_int(state.get("iteration_count", 0), 0)
    max_steps = state.get("manager_max_steps", 50)
    lines.append(f"Iteration: {iteration}/{max_steps}")

    return "\n".join(lines)


def _read_json(path: str) -> Optional[dict]:
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


def _followup_decision_requires_loop(workspace_dir: str) -> tuple[bool, str]:
    """Return (required, reason) from followup_decision.json."""
    path = os.path.join(workspace_dir, "paper_workspace", "followup_decision.json")
    payload = _read_json(path)
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
# Validation gate
# ---------------------------------------------------------------------------

def _run_validation_gates(state: dict) -> dict:
    """
    Run all quality gates.  Returns a ``validation_results`` dict mapping
    gate name -> {is_valid, errors}.  Also returns ``gate_passed`` bool.
    """
    workspace = state.get("workspace_dir") or "."
    results: dict[str, dict] = {}
    all_valid = True

    enforce_paper = state.get("enforce_paper_artifacts", False)
    pipeline_mode = str(state.get("pipeline_mode", "default")).strip().lower()
    should_enforce = enforce_paper or (pipeline_mode == "full_research")

    if should_enforce:
        required = _build_required_artifacts(state)
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
            tr = validate_claim_traceability(workspace_dir=workspace)
            ok = tr.get("is_valid", True)
            results["claim_traceability"] = {
                "is_valid": ok,
                "errors": tr.get("errors", []),
            }
            all_valid = all_valid and ok

    if state.get("enforce_editorial_artifacts", False):
        min_score = state.get("min_review_score", 8)
        rv = validate_review_verdict(workspace_dir=workspace, min_review_score=min_score)
        ok = rv.get("is_valid", True)
        results["review_verdict"] = {
            "is_valid": ok,
            "errors": rv.get("errors", []),
        }
        all_valid = all_valid and ok

        pq = validate_paper_quality(workspace_dir=workspace)
        ok = pq.get("is_valid", True)
        results["paper_quality"] = {
            "is_valid": ok,
            "errors": pq.get("errors", []),
        }
        all_valid = all_valid and ok

    return {"validation_results": results, "gate_passed": all_valid}


def _build_required_artifacts(state: dict) -> list[str]:
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
            "paper_workspace/copyedit_report.md",
            "paper_workspace/review_report.md",
            "paper_workspace/review_verdict.json",
        ])
        if state.get("math_enabled", False):
            required.append("paper_workspace/claim_traceability.json")
    return required


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
