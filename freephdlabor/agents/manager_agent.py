"""
ManagerAgent — LangGraph router node module.

Replaces the smolagents CodeAgent + managed_agents pattern with a LangGraph
router node that uses an LLM to decide which specialist to invoke next.

Routing protocol
----------------
The manager LLM ends every response with a JSON block:

    ROUTING_DECISION:
    {"next_agent": "<agent_name>", "agent_task": "<task for that agent>"}

or, to finish:

    ROUTING_DECISION:
    {"next_agent": "FINISH", "summary": "<completion summary>"}

The node wrapper parses this block, sets ``current_agent`` / ``finished`` in
state, and the graph's conditional edges route accordingly.
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


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

_ROUTING_RE = re.compile(
    r"ROUTING_DECISION\s*:\s*(\{.*?\})",
    re.DOTALL | re.IGNORECASE,
)


def _parse_routing_decision(text: str) -> dict:
    """Extract the ROUTING_DECISION JSON from the manager's last message."""
    m = _ROUTING_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Fallback: look for any JSON object with next_agent key
    for match in re.finditer(r'\{[^{}]*"next_agent"[^{}]*\}', text, re.DOTALL):
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
    # Default: no routing found, treat as finish
    return {"next_agent": "FINISH", "summary": text}


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

    iteration = state.get("iteration_count", 0)
    max_steps = state.get("manager_max_steps", 50)
    lines.append(f"Iteration: {iteration}/{max_steps}")

    return "\n".join(lines)


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
        "FREEPHDLABOR_ENABLE_MANAGER_TEXT_INSPECTOR", "1"
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
    pipeline_mode: str = "default",
    available_agents: Optional[List[str]] = None,
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
        pipeline_mode=pipeline_mode,
        followup_max_iterations=followup_max_iterations,
    )

    # Append routing protocol to system prompt
    agent_list = ", ".join(available_agents or [])
    routing_addendum = f"""

ROUTING OUTPUT (mandatory — append to every response)
------------------------------------------------------
Always end your response with exactly this block (no extra text after it):

ROUTING_DECISION:
{{"next_agent": "<one of: {agent_list}, FINISH>", "agent_task": "<full task description for that agent>"}}

When finishing:
ROUTING_DECISION:
{{"next_agent": "FINISH", "summary": "<brief completion summary>"}}
"""
    full_prompt = system_prompt + routing_addendum

    react_agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=full_prompt,
    )

    if workspace_dir:
        os.makedirs(os.path.join(workspace_dir, "inter_agent_messages"), exist_ok=True)

    def manager_node(state: dict) -> dict:
        context = _build_context_message(state)

        result = react_agent.invoke({
            "messages": [HumanMessage(content=context)],
        })

        last_msg = result["messages"][-1] if result.get("messages") else None
        last_content = last_msg.content if last_msg and hasattr(last_msg, "content") else ""

        routing = _parse_routing_decision(last_content)
        next_agent = routing.get("next_agent", "FINISH")
        agent_task = routing.get("agent_task", routing.get("summary", ""))

        new_iteration = state.get("iteration_count", 0) + 1

        # If manager wants to finish, run validation gates first
        if next_agent == "FINISH":
            val = _run_validation_gates(state)
            if not val["gate_passed"]:
                # Validation failed — inject error into state and re-run manager
                error_msg = "VALIDATION FAILED:\n" + "\n".join(
                    f"  {gate}: {'; '.join(r['errors'])}"
                    for gate, r in val["validation_results"].items()
                    if not r["is_valid"]
                )
                return {
                    "validation_results": val["validation_results"],
                    "current_agent": None,
                    "agent_task": error_msg,
                    "finished": False,
                    "iteration_count": new_iteration,
                    # inject a synthetic message so manager sees the failure
                    "messages": [HumanMessage(content=f"[SYSTEM] {error_msg}")],
                }
            return {
                "validation_results": val["validation_results"],
                "current_agent": "FINISH",
                "agent_task": agent_task,
                "finished": True,
                "iteration_count": new_iteration,
            }

        return {
            "current_agent": next_agent,
            "agent_task": agent_task,
            "finished": False,
            "iteration_count": new_iteration,
            # Clear interrupt once processed
            "interrupt_instruction": None,
        }

    manager_node.__name__ = "manager"
    return manager_node
