"""
BrainstormAgent — LangGraph node module.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.brainstorm_instructions import get_brainstorm_system_prompt
from ..toolkits.filesystem.file_editing.file_editing_tools import (
    CreateFileWithContent, DeleteFileOrFolder, ListDir, ModifyFile, SearchKeyword, SeeFile,
)
from ..toolkits.search.deep_research.openrouter_deep_research_tool import OpenRouterDeepResearchTool
from ..toolkits.search.fetch_arxiv_papers.fetch_arxiv_papers_tools import FetchArxivPapersTool
from ..toolkits.writeup.latex_compiler_tool import LaTeXCompilerTool
from ..toolkits.writeup.latex_generator_tool import LaTeXGeneratorTool
from ..toolkits.writeup.latex_syntax_checker_tool import LaTeXSyntaxCheckerTool
from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool
from ..toolkits.code_execution_tool import PythonCodeExecutionTool

try:
    from ..toolkits.search.deep_research.deep_research_tool import DeepResearchNoveltyScanTool
except (ImportError, ModuleNotFoundError):
    DeepResearchNoveltyScanTool = None


def get_tools(workspace_dir: Optional[str], model_id: str) -> list:
    from . import tool_registry as _reg

    tools = [
        _reg.get_or_create(OpenRouterDeepResearchTool),
        _reg.get_or_create(FetchArxivPapersTool, working_dir=workspace_dir),
    ]
    if DeepResearchNoveltyScanTool is not None:
        tools.append(_reg.get_or_create(DeepResearchNoveltyScanTool, model_name=model_id))
    tools += [
        _reg.get_or_create(VLMDocumentAnalysisTool, model=model_id, working_dir=workspace_dir),
        _reg.get_or_create(LaTeXGeneratorTool, model=model_id, working_dir=workspace_dir),
        _reg.get_or_create(LaTeXCompilerTool, model=model_id, working_dir=workspace_dir),
        _reg.get_or_create(LaTeXSyntaxCheckerTool, working_dir=workspace_dir),
    ]
    if workspace_dir:
        tools += [
            _reg.get_or_create(SeeFile, working_dir=workspace_dir),
            _reg.get_or_create(CreateFileWithContent, working_dir=workspace_dir),
            _reg.get_or_create(ModifyFile, working_dir=workspace_dir),
            _reg.get_or_create(ListDir, working_dir=workspace_dir),
            _reg.get_or_create(SearchKeyword, working_dir=workspace_dir),
            _reg.get_or_create(DeleteFileOrFolder, working_dir=workspace_dir),
            _reg.get_or_create(PythonCodeExecutionTool, workspace_dir=workspace_dir, authorized_imports=[]),
        ]
    return tools


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    from ..toolkits.model_utils import get_raw_model
    model_id = get_raw_model(model)
    tools = get_tools(workspace_dir, model_id)
    system_prompt = get_brainstorm_system_prompt(tools=tools, managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "brainstorm_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="brainstorm_agent",
        workspace_dir=workspace_dir,
    )
