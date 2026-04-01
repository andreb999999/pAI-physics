"""
ExperimentDesignAgent — LangGraph node module.

Transforms empirical questions into executable experiment specs.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.experiment_design_instructions import get_experiment_design_system_prompt
from ..toolkits.code_execution_tool import PythonCodeExecutionTool
from ..toolkits.filesystem.file_editing.file_editing_tools import (
    CreateFileWithContent, DeleteFileOrFolder, ListDir, ModifyFile, SearchKeyword, SeeFile,
)
from ..toolkits.search.deep_research.openrouter_deep_research_tool import OpenRouterDeepResearchTool
from ..toolkits.writeup.citation_search_tool import CitationSearchTool


def get_tools(
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
) -> list:
    from . import tool_registry as _reg
    tools = [
        _reg.get_or_create(OpenRouterDeepResearchTool),
        _reg.get_or_create(CitationSearchTool),
    ]
    if workspace_dir:
        tools += [
            _reg.get_or_create(SeeFile, working_dir=workspace_dir),
            _reg.get_or_create(CreateFileWithContent, working_dir=workspace_dir),
            _reg.get_or_create(ModifyFile, working_dir=workspace_dir),
            _reg.get_or_create(ListDir, working_dir=workspace_dir),
            _reg.get_or_create(SearchKeyword, working_dir=workspace_dir),
            _reg.get_or_create(DeleteFileOrFolder, working_dir=workspace_dir),
            _reg.get_or_create(PythonCodeExecutionTool,
                workspace_dir=workspace_dir,
                authorized_imports=authorized_imports or [],
            ),
        ]
    return tools


def build_node(
    model: Any,
    workspace_dir: Optional[str],
    authorized_imports: Optional[List[str]] = None,
    **cfg: Any,
) -> Callable:
    tools = get_tools(workspace_dir, authorized_imports=authorized_imports)
    system_prompt = get_experiment_design_system_prompt(tools=tools, managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "experiment_design_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="experiment_design_agent",
        workspace_dir=workspace_dir,
    )
