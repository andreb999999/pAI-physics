"""
ResearchPlanWriteupAgent — LangGraph node module.

Reads research_goals.json and track_decomposition.json produced by
formalize_goals_agent and renders them into research_plan.tex + .pdf.
Separated from formalize_goals_agent so LaTeX failures cannot block
goal formalization.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.research_plan_writeup_instructions import get_research_plan_writeup_system_prompt
from ..toolkits.writeup.latex_generator_tool import LaTeXGeneratorTool
from ..toolkits.writeup.latex_reflection_tool import LaTeXReflectionTool
from ..toolkits.writeup.latex_syntax_checker_tool import LaTeXSyntaxCheckerTool
from ..toolkits.writeup.latex_compiler_tool import LaTeXCompilerTool
from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool
from ..toolkits.writeup.citation_search_tool import CitationSearchTool
from ..toolkits.filesystem.file_editing.file_editing_tools import (
    CreateFileWithContent, DeleteFileOrFolder, ListDir, ModifyFile, SearchKeyword, SeeFile,
)


def get_tools(workspace_dir: Optional[str], model_id: str) -> list:
    tools = [
        LaTeXGeneratorTool(model=model_id, working_dir=workspace_dir),
        LaTeXReflectionTool(model=model_id, working_dir=workspace_dir),
        LaTeXSyntaxCheckerTool(working_dir=workspace_dir),
        LaTeXCompilerTool(model=model_id, working_dir=workspace_dir),
        VLMDocumentAnalysisTool(model=model_id, working_dir=workspace_dir),
        CitationSearchTool(),
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
    from ..toolkits.model_utils import get_raw_model
    model_id = get_raw_model(model)
    tools = get_tools(workspace_dir, model_id)
    system_prompt = get_research_plan_writeup_system_prompt(tools=tools, managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "research_plan_writeup_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="research_plan_writeup_agent",
        workspace_dir=workspace_dir,
    )
