"""
ResearchPlannerAgent — LangGraph node module.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from ..prompts.research_planner_instructions import get_research_planner_system_prompt
from ..toolkits.filesystem.file_editing.file_editing_tools import (
    CreateFileWithContent, DeleteFileOrFolder, ListDir, ModifyFile, SearchKeyword, SeeFile,
)
from ..toolkits.ideation.paper_search_tool import PaperSearchTool
from ..toolkits.writeup.citation_search_tool import CitationSearchTool
from ..toolkits.writeup.latex_compiler_tool import LaTeXCompilerTool
from ..toolkits.writeup.latex_generator_tool import LaTeXGeneratorTool
from ..toolkits.writeup.latex_syntax_checker_tool import LaTeXSyntaxCheckerTool
from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool
from ..toolkits.code_execution_tool import PythonCodeExecutionTool
from ..workflow_utils import read_json


def get_tools(workspace_dir: Optional[str], model_id: str) -> list:
    tools = [
        PaperSearchTool(),
        CitationSearchTool(),
        VLMDocumentAnalysisTool(model=model_id, working_dir=workspace_dir),
        LaTeXGeneratorTool(model=model_id, working_dir=workspace_dir),
        LaTeXCompilerTool(model=model_id, working_dir=workspace_dir),
        LaTeXSyntaxCheckerTool(working_dir=workspace_dir),
    ]
    if workspace_dir:
        tools += [
            SeeFile(working_dir=workspace_dir),
            CreateFileWithContent(working_dir=workspace_dir),
            ModifyFile(working_dir=workspace_dir),
            ListDir(working_dir=workspace_dir),
            SearchKeyword(working_dir=workspace_dir),
            DeleteFileOrFolder(working_dir=workspace_dir),
            PythonCodeExecutionTool(workspace_dir=workspace_dir, authorized_imports=[]),
        ]
    return tools


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    from ..toolkits.model_utils import get_raw_model
    from .base_agent import _unwrap_model

    model_id = get_raw_model(model)
    tools = get_tools(workspace_dir, model_id)
    system_prompt = get_research_planner_system_prompt(tools=tools, managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models:
        from ..counsel import create_counsel_node
        inner_node = create_counsel_node(system_prompt, tools, "research_planner_agent", workspace_dir, counsel_models)

        def counsel_with_track_decomposition(state: dict) -> dict:
            result = inner_node(state)
            # Read track_decomposition.json that sandbox agents may have created
            # (merged back to main workspace by counsel artifact promotion).
            if workspace_dir:
                td = read_json(
                    os.path.join(workspace_dir, "paper_workspace", "track_decomposition.json")
                )
                if td is not None:
                    result["track_decomposition"] = td
            return result

        counsel_with_track_decomposition.__name__ = "research_planner_agent"
        return counsel_with_track_decomposition

    # Budget is now recorded automatically by the monkey-patched litellm.completion()
    react_agent = create_react_agent(
        model=_unwrap_model(model),
        tools=tools,
        prompt=system_prompt,
    )

    def node_fn(state: dict) -> dict:
        task = state.get("agent_task") or state.get("task", "")
        result = react_agent.invoke(
            {"messages": [HumanMessage(content=task)]},
        )
        last_msg = result["messages"][-1] if result.get("messages") else None
        output = last_msg.content if last_msg and hasattr(last_msg, "content") else str(last_msg)

        track_decomposition = None
        if workspace_dir:
            track_decomposition = read_json(
                os.path.join(workspace_dir, "paper_workspace", "track_decomposition.json")
            )

        return {
            "agent_outputs": {
                **state.get("agent_outputs", {}),
                "research_planner_agent": output,
            },
            "track_decomposition": track_decomposition,
            "agent_task": None,
        }

    node_fn.__name__ = "research_planner_agent"
    return node_fn
