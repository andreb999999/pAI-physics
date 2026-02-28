"""
ResearchPlannerAgent implementation using smolagents framework.
"""

import os
from typing import Optional

from .base_research_agent import BaseResearchAgent
from ..toolkits.paper_search_tool import PaperSearchTool
from ..toolkits.writeup.citation_search_tool import CitationSearchTool
from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool
from ..toolkits.writeup.latex_generator_tool import LaTeXGeneratorTool
from ..toolkits.writeup.latex_compiler_tool import LaTeXCompilerTool
from ..toolkits.writeup.latex_syntax_checker_tool import LaTeXSyntaxCheckerTool
from ..toolkits.general_tools.file_editing.file_editing_tools import (
    SeeFile,
    CreateFileWithContent,
    ModifyFile,
    ListDir,
    SearchKeyword,
    DeleteFileOrFolder,
)
from ..prompts.research_planner_instructions import get_research_planner_system_prompt


class ResearchPlannerAgent(BaseResearchAgent):
    """
    Agent specialized in creating literature-grounded research plans.
    """

    def __init__(self, model, workspace_dir: Optional[str] = None, **kwargs):
        if workspace_dir:
            workspace_dir = os.path.abspath(workspace_dir)
            self.agent_folder = os.path.join(workspace_dir, "research_planner_agent")

        from ..toolkits.model_utils import get_raw_model

        raw_model = get_raw_model(model)
        tools = [
            PaperSearchTool(),
            CitationSearchTool(),
            VLMDocumentAnalysisTool(model=raw_model, working_dir=workspace_dir),
            LaTeXGeneratorTool(model=raw_model, working_dir=workspace_dir),
            LaTeXCompilerTool(model=raw_model, working_dir=workspace_dir),
            LaTeXSyntaxCheckerTool(working_dir=workspace_dir),
        ]

        if workspace_dir:
            tools.extend(
                [
                    SeeFile(working_dir=workspace_dir),
                    CreateFileWithContent(working_dir=workspace_dir),
                    ModifyFile(working_dir=workspace_dir),
                    ListDir(working_dir=workspace_dir),
                    SearchKeyword(working_dir=workspace_dir),
                    DeleteFileOrFolder(working_dir=workspace_dir),
                ]
            )

        system_prompt = get_research_planner_system_prompt(
            tools=tools,
            managed_agents=None,
        )

        super().__init__(
            model=model,
            agent_name="research_planner_agent",
            workspace_dir=workspace_dir,
            tools=tools,
            **kwargs,
        )
        self.prompt_templates["system_prompt"] = system_prompt
        self.resume_memory()
