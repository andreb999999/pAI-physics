"""
ExperimentTranscriptionAgent — LangGraph node module.

Converts verified experiment artifacts to publication-quality LaTeX.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..agents.base_agent import create_specialist_agent
from ..prompts.experiment_transcription_instructions import get_experiment_transcription_system_prompt
from ..toolkits.filesystem.file_editing.file_editing_tools import (
    CreateFileWithContent, DeleteFileOrFolder, ListDir, ModifyFile, SearchKeyword, SeeFile,
)
from ..toolkits.writeup.latex_compiler_tool import LaTeXCompilerTool
from ..toolkits.writeup.latex_generator_tool import LaTeXGeneratorTool
from ..toolkits.writeup.latex_reflection_tool import LaTeXReflectionTool
from ..toolkits.writeup.latex_syntax_checker_tool import LaTeXSyntaxCheckerTool
from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool


def get_tools(workspace_dir: Optional[str], model_id: str) -> list:
    from . import tool_registry as _reg
    tools = [
        _reg.get_or_create(LaTeXGeneratorTool, model=model_id, working_dir=workspace_dir),
        _reg.get_or_create(LaTeXReflectionTool, model=model_id, working_dir=workspace_dir),
        _reg.get_or_create(LaTeXCompilerTool, model=model_id, working_dir=workspace_dir),
        _reg.get_or_create(LaTeXSyntaxCheckerTool, working_dir=workspace_dir),
        _reg.get_or_create(VLMDocumentAnalysisTool, model=model_id, working_dir=workspace_dir),
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
    system_prompt = get_experiment_transcription_system_prompt(tools=tools, managed_agents=None)
    counsel_models = cfg.get("counsel_models")
    if counsel_models is not None:
        from ..counsel import create_counsel_node
        return create_counsel_node(system_prompt, tools, "experiment_transcription_agent", workspace_dir, counsel_models)
    return create_specialist_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        agent_name="experiment_transcription_agent",
        workspace_dir=workspace_dir,
    )
