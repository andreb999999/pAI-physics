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
    from . import tool_registry as _reg
    tools = [
        _reg.get_or_create(IdeaStandardizationTool, model=model_id),
        _reg.get_or_create(RunExperimentTool, workspace_dir=workspace_dir),
        _reg.get_or_create(LaTeXGeneratorTool, model=model_id, working_dir=workspace_dir),
        _reg.get_or_create(LaTeXCompilerTool, working_dir=workspace_dir, model=model_id),
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
    from ..toolkits.model_utils import get_raw_model
    model_id = get_raw_model(model)
    tools = get_tools(workspace_dir, model_id)
    if workspace_dir:
        from . import tool_registry as _reg
        tools.append(
            _reg.get_or_create(PythonCodeExecutionTool,
                workspace_dir=workspace_dir,
                authorized_imports=authorized_imports or [],
            )
        )
    system_prompt = get_experimentation_system_prompt(tools=tools, managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "experimentation_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="experimentation_agent",
        workspace_dir=workspace_dir,
    )
