"""
MathLiteratureAgent: mines literature for reusable lemma infrastructure.
"""

import os
from typing import Optional

from .base_research_agent import BaseResearchAgent
from ..toolkits.paper_search_tool import PaperSearchTool
from ..toolkits.general_tools.fetch_arxiv_papers.fetch_arxiv_papers_tools import (
    FetchArxivPapersTool,
)

# Optional dependency guard to keep runs working without crawl4ai
try:
    from ..toolkits.general_tools.open_deep_search.ods_tool import OpenDeepSearchTool
except (ImportError, ModuleNotFoundError):  # pragma: no cover - optional dependency fallback
    OpenDeepSearchTool = None

from ..toolkits.writeup.citation_search_tool import CitationSearchTool
from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool
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
from ..prompts.math_literature_instructions import get_math_literature_system_prompt


class MathLiteratureAgent(BaseResearchAgent):
    def __init__(self, model, workspace_dir: Optional[str] = None, **kwargs):
        if workspace_dir:
            workspace_dir = os.path.abspath(workspace_dir)
            self.agent_folder = os.path.join(workspace_dir, "math_literature_agent")

        from ..toolkits.model_utils import get_raw_model

        raw_model = get_raw_model(model)
        tools = [
            PaperSearchTool(),
            FetchArxivPapersTool(working_dir=workspace_dir),
            CitationSearchTool(),
            VLMDocumentAnalysisTool(model=raw_model, working_dir=workspace_dir),
            MathClaimGraphTool(working_dir=workspace_dir),
            MathProofWorkspaceTool(working_dir=workspace_dir),
        ]

        if OpenDeepSearchTool is not None:
            tools.insert(2, OpenDeepSearchTool(model_name=model.model_id))
        else:
            print(
                "⚠️ OpenDeepSearchTool disabled for MathLiteratureAgent: "
                "optional dependency 'crawl4ai' is not installed."
            )

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

        system_prompt = get_math_literature_system_prompt(tools=tools, managed_agents=None)

        super().__init__(
            model=model,
            agent_name="math_literature_agent",
            workspace_dir=workspace_dir,
            tools=tools,
            **kwargs,
        )
        self.prompt_templates["system_prompt"] = system_prompt
        self.resume_memory()
