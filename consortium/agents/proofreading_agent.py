"""
ProofreadingAgent — LangGraph node module.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..agents.stage_write_tools import (
    RestrictedCreateFileWithContent,
    RestrictedModifyFile,
)
from ..prompts.proofreading_instructions import get_proofreading_system_prompt
from ..toolkits.filesystem.file_editing.file_editing_tools import (
    ListDir,
    SearchKeyword,
    SeeFile,
)
from ..toolkits.writeup.latex_compiler_tool import LaTeXCompilerTool
from ..toolkits.writeup.latex_content_verification_tool import LaTeXContentVerificationTool
from ..toolkits.writeup.latex_reflection_tool import LaTeXReflectionTool
from ..toolkits.writeup.latex_syntax_checker_tool import LaTeXSyntaxCheckerTool
from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool
from ..toolkits.code_execution_tool import PythonCodeExecutionTool


def get_tools(
    workspace_dir: Optional[str],
    model_id: str,
    authorized_imports: Optional[List[str]] = None,
) -> list:
    from . import tool_registry as _reg
    allowed_modify_paths = [
        "paper_workspace/abstract.tex",
        "paper_workspace/introduction.tex",
        "paper_workspace/methods.tex",
        "paper_workspace/results.tex",
        "paper_workspace/discussion.tex",
        "paper_workspace/conclusion.tex",
        "paper_workspace/final_paper.tex",
        "paper_workspace/references.bib",
        "paper_workspace/copyedit_report.tex",
        "paper_workspace/revision_log.md",
    ]
    allowed_create_paths = [
        "paper_workspace/copyedit_report.tex",
    ]
    tools = [
        _reg.get_or_create(LaTeXCompilerTool, working_dir=workspace_dir, model=model_id),
        _reg.get_or_create(LaTeXSyntaxCheckerTool, working_dir=workspace_dir),
        _reg.get_or_create(LaTeXContentVerificationTool, working_dir=workspace_dir),
        _reg.get_or_create(LaTeXReflectionTool, model=model_id, working_dir=workspace_dir),
        _reg.get_or_create(VLMDocumentAnalysisTool, working_dir=workspace_dir, model=model_id),
        _reg.get_or_create(SeeFile, working_dir=workspace_dir),
        _reg.get_or_create(
            RestrictedModifyFile,
            working_dir=workspace_dir,
            allowed_write_prefixes=allowed_modify_paths,
        ),
        _reg.get_or_create(ListDir, working_dir=workspace_dir),
        _reg.get_or_create(SearchKeyword, working_dir=workspace_dir),
        _reg.get_or_create(
            RestrictedCreateFileWithContent,
            working_dir=workspace_dir,
            allowed_write_prefixes=allowed_create_paths,
        ),
    ]
    if workspace_dir:
        tools.append(_reg.get_or_create(PythonCodeExecutionTool, workspace_dir=workspace_dir, authorized_imports=authorized_imports or []))
    return tools


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    from ..toolkits.model_utils import get_raw_model
    model_id = get_raw_model(model)
    tools = get_tools(workspace_dir, model_id, authorized_imports=authorized_imports)
    system_prompt = get_proofreading_system_prompt(tools=tools, managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "proofreading_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="proofreading_agent",
        workspace_dir=workspace_dir,
    )
