"""
ManagerAgent implementation using smolagents framework.
Orchestrates IdeationAgent and other agents in the multi-agent system.
"""

import os
from typing import List
from .base_research_agent import BaseResearchAgent
from ..result_validation import validate_result_artifacts
from ..math_acceptance_validation import validate_math_acceptance

from .reviewer_agent import ReviewerAgent
from .ideation_agent import IdeationAgent
from .experimentation_agent import ExperimentationAgent
from .resource_preparation_agent import ResourcePreparationAgent
from .writeup_agent import WriteupAgent
from .math_proposer_agent import MathProposerAgent
from .math_prover_agent import MathProverAgent
from .math_rigorous_verifier_agent import MathRigorousVerifierAgent
from .math_empirical_verifier_agent import MathEmpiricalVerifierAgent
from ..toolkits.general_tools.file_editing.file_editing_tools import (
    SeeFile,
    CreateFileWithContent,
    ModifyFile,
    ListDir,
    SearchKeyword,
    DeleteFileOrFolder,
)
from ..toolkits.general_tools.text_inspector.text_inspector_tool import TextInspectorTool
from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool
from ..prompts.manager_instructions import get_manager_system_prompt


class ManagerAgent(BaseResearchAgent):
    """
    A manager agent that orchestrates other agents in the AI scientist system.
    This agent decides which specialist agent to use to accomplish a task.

    Workflow Intelligence:
    - Automatically delegates idea generation to IdeationAgent
    - Delegates experiment execution to ExperimentationAgent
    - Delegates paper writing to WriteupAgent
    - Handles failures gracefully without generating synthetic results
    - Manages workspace coordination between agents
    """

    def __init__(
        self, model, interpreter, workspace_dir=None, managed_agents=None, **kwargs
    ):
        """
        Initialize the ManagerAgent.

        Args:
            model: The LLM model to use for the agent.
            interpreter: Code interpreter for the agent.
            workspace_dir: Optional workspace directory for agent coordination.
            managed_agents: Optional list of pre-initialized agents to manage.
                          If None, will create default agents internally.
            **kwargs: Additional arguments passed to BaseResearchAgent.
        """
        # Store the interpreter for later use (BaseResearchAgent will handle workspace executor)
        self.interpreter = interpreter
        self.require_pdf = bool(kwargs.pop("require_pdf", False))
        self.enforce_paper_artifacts = bool(kwargs.pop("enforce_paper_artifacts", False))
        self.require_experiment_plan = bool(kwargs.pop("require_experiment_plan", False))
        self.enable_math_agents = bool(kwargs.pop("enable_math_agents", False))
        existing_final_answer_checks = kwargs.pop("final_answer_checks", [])

        # Create inter-agent messages folder (specific to ManagerAgent)
        if workspace_dir:
            os.makedirs(
                os.path.join(workspace_dir, "inter_agent_messages"), exist_ok=True
            )

        # Use provided managed agents or create them if not provided
        if managed_agents is not None:
            # Use pre-initialized agents (recommended approach)
            self.managed_agents = managed_agents
        else:
            # Fallback: Create agents internally (legacy behavior)
            # Essential imports for tool-centric agents (shared across all agents)
            essential_imports = kwargs.get("additional_authorized_imports", [])

            # Create managed agents for delegation - they will initialize their own file editing tools
            # Note: Managed agents will get their own logging wrappers in their constructors
            from ..prompts.ideation_instructions import get_ideation_system_prompt
            ideation_agent = IdeationAgent(
                model=model,  # Pass original model, they'll wrap it themselves
                workspace_dir=workspace_dir,
                name="ideation_agent",
                description=f"""A specialist agent for generating, refining, and evaluating research ideas.

--- SYSTEM INSTRUCTIONS ---
{get_ideation_system_prompt()}
--- END SYSTEM INSTRUCTIONS ---""",
                additional_authorized_imports=essential_imports,
            )

            from ..prompts.experimentation_instructions import get_experimentation_system_prompt
            experimentation_agent = ExperimentationAgent(
                model=model,  # Pass original model, they'll wrap it themselves
                workspace_dir=workspace_dir,
                name="experimentation_agent",
                description=f"""A specialist agent for running experiments and analyzing results using RunExperimentTool.

--- SYSTEM INSTRUCTIONS ---
{get_experimentation_system_prompt()}
--- END SYSTEM INSTRUCTIONS ---""",
                additional_authorized_imports=essential_imports,
            )

            from ..prompts.writeup_instructions import get_writeup_system_prompt
            writeup_agent = WriteupAgent(
                model=model,  # Pass original model, they'll wrap it themselves
                workspace_dir=workspace_dir,
                name="writeup_agent",
                description=f"""A specialist agent for academic paper writing that works with pre-organized resources from ResourcePreparationAgent.

--- SYSTEM INSTRUCTIONS ---
{get_writeup_system_prompt(tools=[], managed_agents=None)}
--- END SYSTEM INSTRUCTIONS ---""",
                additional_authorized_imports=essential_imports,
            )

            from ..prompts.resource_preparation_instructions import get_resource_preparation_system_prompt
            resource_preparation_agent = ResourcePreparationAgent(
                model=model,  # Pass original model, they'll wrap it themselves
                workspace_dir=workspace_dir,
                name="resource_preparation_agent",
                description=f"""A comprehensive resource organization agent that prepares complete experimental documentation for WriteupAgent.

Key Functions: Locates experiment results folders, creates paper_workspace/ workspace, links experiment data using symlinks/copies, generates complete file structure analysis with descriptions of EVERY file found, creates comprehensive bibliography based on full experimental understanding.

Key Tools: ExperimentLinkerTool, CitationSearchTool, VLMDocumentAnalysisTool, file editing tools.

Approach: Comprehensive documentation of all experimental artifacts without selectivity. Creates complete file tree structure, reads actual content of every file (VLM for images), and provides complete resource inventory. WriteupAgent can then selectively choose what to use from the comprehensive documentation.

--- SYSTEM INSTRUCTIONS ---
{get_resource_preparation_system_prompt()}
--- END SYSTEM INSTRUCTIONS ---""",
                additional_authorized_imports=essential_imports,
            )

            from ..prompts.reviewer_instructions import get_reviewer_system_prompt
            reviewer_agent = ReviewerAgent(
                model=model,  # Pass original model, they'll wrap it themselves
                workspace_dir=workspace_dir,
                name="reviewer_agent",
                description=f"""A specialist agent for peer-reviewing AI research paper.

--- SYSTEM INSTRUCTIONS ---
{get_reviewer_system_prompt()}
--- END SYSTEM INSTRUCTIONS ---""",
                additional_authorized_imports=essential_imports,
            )

            self.managed_agents = [ideation_agent, experimentation_agent, resource_preparation_agent, writeup_agent, reviewer_agent]

            if self.enable_math_agents:
                math_proposer_agent = MathProposerAgent(
                    model=model,
                    workspace_dir=workspace_dir,
                    name="math_proposer_agent",
                    description="A specialist agent for constructing mathematical claim graphs (claims, assumptions, dependencies).",
                    additional_authorized_imports=essential_imports,
                )
                math_prover_agent = MathProverAgent(
                    model=model,
                    workspace_dir=workspace_dir,
                    name="math_prover_agent",
                    description="A specialist agent for writing structured proof drafts tied to claim graph items.",
                    additional_authorized_imports=essential_imports,
                )
                math_rigorous_verifier_agent = MathRigorousVerifierAgent(
                    model=model,
                    workspace_dir=workspace_dir,
                    name="math_rigorous_verifier_agent",
                    description="A specialist agent for auditing proof rigor and symbolic completeness.",
                    additional_authorized_imports=essential_imports,
                )
                math_empirical_verifier_agent = MathEmpiricalVerifierAgent(
                    model=model,
                    workspace_dir=workspace_dir,
                    name="math_empirical_verifier_agent",
                    description="A specialist agent for numeric sanity checks and counterexample search on math claims.",
                    additional_authorized_imports=essential_imports,
                )
                self.managed_agents.extend(
                    [
                        math_proposer_agent,
                        math_prover_agent,
                        math_rigorous_verifier_agent,
                        math_empirical_verifier_agent,
                    ]
                )

        # Build dynamic agent list for prompt
        available_agents = [agent.name for agent in self.managed_agents]

        # Initialize file editing tools for ManagerAgent
        from ..toolkits.model_utils import get_raw_model
        raw_model = get_raw_model(model)
        file_editing_tools = []
        enable_manager_text_inspector = os.getenv(
            "FREEPHDLABOR_ENABLE_MANAGER_TEXT_INSPECTOR", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Keep at least one robust PDF/doc analysis tool available in manager.
        document_tools = [VLMDocumentAnalysisTool(model=raw_model, working_dir=workspace_dir)]
        if enable_manager_text_inspector:
            # Always available in manager for PDF/long-doc ingestion when deps exist.
            try:
                document_tools.append(
                    TextInspectorTool(model=raw_model, working_dir=workspace_dir)
                )
            except Exception as e:
                print(
                    "⚠️ TextInspectorTool disabled: optional document dependencies "
                    f"are missing or misconfigured ({e})."
                )

        if workspace_dir:
            file_editing_tools = [
                SeeFile(working_dir=workspace_dir),
                CreateFileWithContent(working_dir=workspace_dir),
                ModifyFile(working_dir=workspace_dir),
                ListDir(working_dir=workspace_dir),
                SearchKeyword(working_dir=workspace_dir),
                DeleteFileOrFolder(working_dir=workspace_dir),
            ]

        tools: List = [*document_tools, *file_editing_tools]

        # Generate complete system prompt using template
        system_prompt = get_manager_system_prompt(
            tools=tools, managed_agents=self.managed_agents
        )

        super().__init__(
            model=model,  # Pass original model, BaseResearchAgent will handle logging
            agent_name="manager_agent",
            workspace_dir=workspace_dir,
            tools=tools,
            managed_agents=self.managed_agents,
            final_answer_checks=[
                *existing_final_answer_checks,
                self._validate_manager_success_criteria,
            ],
            **kwargs
        )

        # Set system prompt after initialization (correct smolagents pattern)
        self.prompt_templates["system_prompt"] = system_prompt

    # The run method is inherited from the parent CodeAgent and is what should be called
    # to execute a task. It uses the LLM to reason and create a plan.
    
        # Resume memory if possible
        self.resume_memory()

    def _paper_required_artifacts(self) -> list[str]:
        required = ["final_paper.tex"]
        if self.require_experiment_plan:
            required.append("experiments_to_run_later.md")
        if self.require_pdf:
            required.append("final_paper.pdf")
        return required

    def _validate_manager_success_criteria(self, final_answer, memory, agent=None):
        """
        Ensure manager does not terminate with false artifact claims
        or invalid theorem-acceptance state.
        """
        workspace = self.workspace_dir or "."

        if self.enforce_paper_artifacts:
            summary = validate_result_artifacts(
                result=final_answer,
                workspace_dir=workspace,
                required_artifacts=self._paper_required_artifacts(),
            )

            if summary["missing_required_artifacts"]:
                raise ValueError(
                    "TERMINATION_BLOCKED: Missing required paper artifacts: "
                    + ", ".join(summary["missing_required_artifacts"])
                )

            # If the model reports artifacts, force truthful reporting.
            if summary["artifacts"] and summary["missing_artifacts"]:
                raise ValueError(
                    "TERMINATION_BLOCKED: Final answer lists artifacts that do not exist: "
                    + ", ".join(summary["missing_artifacts"])
                    + ". Create them or remove them from the artifacts list."
                )

        if self.enable_math_agents:
            math_summary = validate_math_acceptance(workspace_dir=workspace)
            if math_summary["graph_present"] and not math_summary["is_valid"]:
                raise ValueError(
                    "TERMINATION_BLOCKED: Math claim acceptance audit failed: "
                    + "; ".join(math_summary["errors"])
                )

        return True
