"""
MathRigorousVerifierAgent — LangGraph node module.

Audits proof rigor and symbolic completeness.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.math_rigorous_verifier_instructions import get_math_rigorous_verifier_system_prompt
from ..toolkits.math.claim_graph_tool import MathClaimGraphTool
from ..toolkits.math.proof_workspace_tool import MathProofWorkspaceTool
from ..toolkits.math.proof_rigor_checker_tool import MathProofRigorCheckerTool
from ..toolkits.filesystem.file_editing.file_editing_tools import (
    CreateFileWithContent, DeleteFileOrFolder, ListDir, ModifyFile, SearchKeyword, SeeFile,
)


def get_tools(workspace_dir: Optional[str]) -> list:
    tools = [
        MathClaimGraphTool(working_dir=workspace_dir, allow_accepted_transition=False),
        MathProofWorkspaceTool(working_dir=workspace_dir),
        MathProofRigorCheckerTool(working_dir=workspace_dir),
    ]
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


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    tools = get_tools(workspace_dir)
    system_prompt = get_math_rigorous_verifier_system_prompt(tools=tools, managed_agents=None)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="math_rigorous_verifier_agent",
        workspace_dir=workspace_dir,
    )
