"""
ExperimentationAgent — LangGraph node module.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.experimentation_instructions import get_experimentation_system_prompt
from ..toolkits.experimentation.idea_standardization_tool import IdeaStandardizationTool
from ..toolkits.experimentation.run_experiment_tool import RunExperimentTool
from ..toolkits.filesystem.file_editing.file_editing_tools import (
    CreateFileWithContent, DeleteFileOrFolder, ListDir, ModifyFile, SearchKeyword, SeeFile,
)
from ..toolkits.writeup.latex_compiler_tool import LaTeXCompilerTool
from ..toolkits.writeup.latex_generator_tool import LaTeXGeneratorTool
from ..toolkits.code_execution_tool import PythonCodeExecutionTool


def get_tools(workspace_dir: Optional[str], model_id: str) -> list:
    tools = [
        IdeaStandardizationTool(model=model_id),
        RunExperimentTool(workspace_dir=workspace_dir),
        LaTeXGeneratorTool(model=model_id, working_dir=workspace_dir),
        LaTeXCompilerTool(working_dir=workspace_dir, model=model_id),
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
    model_id = get_raw_model(model)
    tools = get_tools(workspace_dir, model_id)
    system_prompt = get_experimentation_system_prompt(tools=tools, managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "experimentation_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="experimentation_agent",
        workspace_dir=workspace_dir,
    )
