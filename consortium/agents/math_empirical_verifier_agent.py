"""
MathEmpiricalVerifierAgent — LangGraph node module.

Performs numeric sanity checks for claims.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.math_empirical_verifier_instructions import get_math_empirical_verifier_system_prompt
from ..toolkits.math.claim_graph_tool import MathClaimGraphTool
from ..toolkits.math.proof_workspace_tool import MathProofWorkspaceTool
from ..toolkits.math.numerical_claim_verifier_tool import MathNumericalClaimVerifierTool
from ..toolkits.filesystem.file_editing.file_editing_tools import (
    CreateFileWithContent, DeleteFileOrFolder, ListDir, ModifyFile, SearchKeyword, SeeFile,
)


def get_tools(workspace_dir: Optional[str]) -> list:
    from . import tool_registry as _reg
    tools = [
        _reg.get_or_create(MathClaimGraphTool, working_dir=workspace_dir, allow_accepted_transition=False),
        _reg.get_or_create(MathProofWorkspaceTool, working_dir=workspace_dir),
        _reg.get_or_create(MathNumericalClaimVerifierTool, working_dir=workspace_dir),
    ]
    if workspace_dir:
        tools += [
            _reg.get_or_create(SeeFile, working_dir=workspace_dir),
            _reg.get_or_create(CreateFileWithContent, working_dir=workspace_dir),
            _reg.get_or_create(ModifyFile, working_dir=workspace_dir),
            _reg.get_or_create(ListDir, working_dir=workspace_dir),
            _reg.get_or_create(SearchKeyword, working_dir=workspace_dir),
            _reg.get_or_create(DeleteFileOrFolder, working_dir=workspace_dir),
        ]
    return tools


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    tools = get_tools(workspace_dir)
    system_prompt = get_math_empirical_verifier_system_prompt(tools=tools, managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "math_empirical_verifier_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="math_empirical_verifier_agent",
        workspace_dir=workspace_dir,
    )
