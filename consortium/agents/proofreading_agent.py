"""
ProofreadingAgent — LangGraph node module.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.proofreading_instructions import get_proofreading_system_prompt
from ..toolkits.filesystem.file_editing.file_editing_tools import (
    CreateFileWithContent,
    DeleteFileOrFolder,
    ListDir,
    ModifyFile,
    SearchKeyword,
    SeeFile,
)
from ..toolkits.writeup.latex_compiler_tool import LaTeXCompilerTool
from ..toolkits.writeup.latex_generator_tool import LaTeXGeneratorTool
from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool
from ..toolkits.code_execution_tool import PythonCodeExecutionTool


def get_tools(
    workspace_dir: Optional[str],
    model_id: str,
    authorized_imports: Optional[List[str]] = None,
) -> list:
    tools = [
        LaTeXGeneratorTool(model=model_id, working_dir=workspace_dir),
        LaTeXCompilerTool(working_dir=workspace_dir, model=model_id),
        VLMDocumentAnalysisTool(working_dir=workspace_dir, model=model_id),
        SeeFile(working_dir=workspace_dir),
        ModifyFile(working_dir=workspace_dir),
        ListDir(working_dir=workspace_dir),
        SearchKeyword(working_dir=workspace_dir),
        CreateFileWithContent(working_dir=workspace_dir),
        DeleteFileOrFolder(working_dir=workspace_dir),
    ]
    if workspace_dir:
        tools.append(PythonCodeExecutionTool(workspace_dir=workspace_dir, authorized_imports=authorized_imports or []))
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
