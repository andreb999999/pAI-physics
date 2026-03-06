"""
IdeationAgent — LangGraph node module.

Exports:
    get_tools(workspace_dir, model_id)  -> list[BaseTool]
    build_node(model, workspace_dir, authorized_imports, **cfg)  -> callable
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.ideation_instructions import get_ideation_system_prompt
from ..toolkits.filesystem.file_editing.file_editing_tools import (
    CreateFileWithContent, DeleteFileOrFolder, ListDir, ModifyFile, SearchKeyword, SeeFile,
)
from ..toolkits.ideation.generate_idea_tool import GenerateIdeaTool
from ..toolkits.ideation.refine_idea_tool import RefineIdeaTool
from ..toolkits.search.fetch_arxiv_papers.fetch_arxiv_papers_tools import FetchArxivPapersTool
from ..toolkits.writeup.latex_compiler_tool import LaTeXCompilerTool
from ..toolkits.writeup.latex_generator_tool import LaTeXGeneratorTool
from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool
from ..toolkits.code_execution_tool import PythonCodeExecutionTool

try:
    from ..toolkits.search.open_deep_search.ods_tool import OpenDeepSearchTool
except (ImportError, ModuleNotFoundError):
    OpenDeepSearchTool = None


def get_tools(workspace_dir: Optional[str], model_id: str) -> list:
    tools = [
        FetchArxivPapersTool(working_dir=workspace_dir),
        GenerateIdeaTool(model=model_id),
        RefineIdeaTool(model=model_id),
        VLMDocumentAnalysisTool(model=model_id, working_dir=workspace_dir),
        LaTeXGeneratorTool(model=model_id, working_dir=workspace_dir),
        LaTeXCompilerTool(working_dir=workspace_dir, model=model_id),
    ]
    if OpenDeepSearchTool is not None:
        tools.insert(0, OpenDeepSearchTool(model_name=model_id))
    else:
        print("⚠️ OpenDeepSearchTool disabled for IdeationAgent: crawl4ai not installed.")
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
    from ..toolkits.model_utils import get_raw_model
    model_id = get_raw_model(model)
    tools = get_tools(workspace_dir, model_id)
    system_prompt = get_ideation_system_prompt(tools=tools, managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "ideation_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="ideation_agent",
        workspace_dir=workspace_dir,
    )
