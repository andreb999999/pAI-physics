"""MathProposerAgent: builds structured mathematical claim graphs."""

import os
from typing import Optional

from .base_research_agent import BaseResearchAgent
from ..prompts.math_proposer_instructions import get_math_proposer_system_prompt
from ..toolkits.math.claim_graph_tool import MathClaimGraphTool
from ..toolkits.math.proof_workspace_tool import MathProofWorkspaceTool
from ..toolkits.general_tools.file_editing.file_editing_tools import (
    SeeFile,
    CreateFileWithContent,
    ModifyFile,
    ListDir,
    SearchKeyword,
    DeleteFileOrFolder,
)


class MathProposerAgent(BaseResearchAgent):
    def __init__(self, model, workspace_dir: Optional[str] = None, **kwargs):
        if workspace_dir:
            workspace_dir = os.path.abspath(workspace_dir)

        tools = [
            MathClaimGraphTool(
                working_dir=workspace_dir,
                allow_accepted_transition=False,
            ),
            MathProofWorkspaceTool(working_dir=workspace_dir),
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

        system_prompt = get_math_proposer_system_prompt(tools=tools, managed_agents=None)

        super().__init__(
            model=model,
            agent_name="math_proposer_agent",
            workspace_dir=workspace_dir,
            tools=tools,
            **kwargs,
        )
        self.prompt_templates["system_prompt"] = system_prompt
        self.resume_memory()
