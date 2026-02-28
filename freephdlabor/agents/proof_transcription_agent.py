"""
ProofTranscriptionAgent: converts math proof artifacts to publication-quality LaTeX.
"""

import os
from typing import Optional

from .base_research_agent import BaseResearchAgent
from ..toolkits.math.claim_graph_tool import MathClaimGraphTool
from ..toolkits.math.proof_workspace_tool import MathProofWorkspaceTool
from ..toolkits.writeup.latex_generator_tool import LaTeXGeneratorTool
from ..toolkits.writeup.latex_reflection_tool import LaTeXReflectionTool
from ..toolkits.writeup.latex_syntax_checker_tool import LaTeXSyntaxCheckerTool
from ..toolkits.writeup.latex_compiler_tool import LaTeXCompilerTool
from ..toolkits.general_tools.file_editing.file_editing_tools import (
    SeeFile,
    CreateFileWithContent,
    ModifyFile,
    ListDir,
    SearchKeyword,
    DeleteFileOrFolder,
)
from ..prompts.proof_transcription_instructions import (
    get_proof_transcription_system_prompt,
)


class ProofTranscriptionAgent(BaseResearchAgent):
    def __init__(self, model, workspace_dir: Optional[str] = None, **kwargs):
        if workspace_dir:
            workspace_dir = os.path.abspath(workspace_dir)
            self.agent_folder = os.path.join(workspace_dir, "proof_transcription_agent")

        from ..toolkits.model_utils import get_raw_model

        raw_model = get_raw_model(model)
        tools = [
            MathClaimGraphTool(working_dir=workspace_dir),
            MathProofWorkspaceTool(working_dir=workspace_dir),
            LaTeXGeneratorTool(model=raw_model, working_dir=workspace_dir),
            LaTeXReflectionTool(model=raw_model, working_dir=workspace_dir),
            LaTeXSyntaxCheckerTool(working_dir=workspace_dir),
            LaTeXCompilerTool(model=raw_model, working_dir=workspace_dir),
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

        system_prompt = get_proof_transcription_system_prompt(
            tools=tools, managed_agents=None
        )

        super().__init__(
            model=model,
            agent_name="proof_transcription_agent",
            workspace_dir=workspace_dir,
            tools=tools,
            **kwargs,
        )
        self.prompt_templates["system_prompt"] = system_prompt
        self.resume_memory()
