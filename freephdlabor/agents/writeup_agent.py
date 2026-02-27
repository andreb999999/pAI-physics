"""
WriteupAgent implementation using smolagents framework.
Minimal implementation focused on paper writing via writeup tools.
Designed to be managed by ManagerAgent for delegation-based workflow.
"""

import os
import json
import time
from typing import Optional
from .base_research_agent import BaseResearchAgent
from smolagents.memory import ActionStep, MemoryStep

# SPECIALIZED WRITEUP AGENT - LaTeX-focused tools with citation support
# Tools MIGRATED to ResourcePreparationAgent:
# - ExperimentDataOrganizerTool ❌ (prep agent handles all organization)
# - TrainingAnalysisPlotTool ❌ (prep agent handles all plotting)
# - ComparisonPlotTool ❌ (prep agent handles all plotting)
# - StatisticalAnalysisPlotTool ❌ (prep agent handles all plotting)
# - MultiPanelCompositionTool ❌ (prep agent handles all plotting)
# - PlotEnhancementTool ❌ (prep agent handles plot enhancement)

# CORE LaTeX WORKFLOW TOOLS:
from ..toolkits.writeup.latex_generator_tool import LaTeXGeneratorTool
from ..toolkits.writeup.latex_reflection_tool import LaTeXReflectionTool
from ..toolkits.writeup.latex_compiler_tool import LaTeXCompilerTool
from ..toolkits.writeup.latex_content_verification_tool import LaTeXContentVerificationTool
from ..toolkits.writeup.latex_syntax_checker_tool import LaTeXSyntaxCheckerTool
from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool  # PDF validation only
from ..toolkits.general_tools.file_editing.file_editing_tools import (
    SeeFile, CreateFileWithContent, ModifyFile, ListDir, SearchKeyword, DeleteFileOrFolder
)
from ..prompts.writeup_instructions import get_writeup_system_prompt



class WriteupAgent(BaseResearchAgent):
    """
    SPECIALIZED WriteupAgent - Focused on LaTeX writing and compilation with citation support.

    SPECIALIZATION CHANGES:
    - STREAMLINED tools (reduced decision paralysis)
    - EXPECTS pre-organized resources from ResourcePreparationAgent
    - FOCUSES on LaTeX content creation, compilation, and quality validation
    - Citations automatically handled by LaTeXCompilerTool (no manual management needed)

    Design Philosophy:
    - Streamlined LaTeX-focused workflow with citation support
    - Uses pre-validated resources from paper_workspace/
    - Can search for and validate citations dynamically
    - Designed to work AFTER ResourcePreparationAgent has prepared everything
    - Managed by ManagerAgent for delegation-based workflow

    Specialized Agent Process:
    1. READ resource_inventory.md to understand available pre-organized resources
    2. GENERATE LaTeX content using organized figures and data (use [cite: description] placeholders)
    3. ITERATIVELY improve content quality using reflection tools
    4. COMPILE to PDF (LaTeXCompilerTool automatically resolves citations)
    5. VALIDATE success criteria and report completion
    """
    
    def __init__(self, model, workspace_dir: Optional[str] = None, **kwargs):
        """
        Initialize the WriteupAgent.
        
        Args:
            model: The LLM model to use for the agent
            workspace_dir: Directory for workspace operations
            **kwargs: Additional arguments passed to BaseResearchAgent
        """
        # Convert workspace_dir to absolute path immediately to prevent nested directory issues
        if workspace_dir:
            workspace_dir = os.path.abspath(workspace_dir)
            
        # Legacy compatibility: set agent_folder for any code that might reference it
        if workspace_dir:
            self.agent_folder = os.path.join(workspace_dir, "writeup_agent")
        
        # Initialize tools - comprehensive set for academic writing (workspace-aware)
        # NOTE: Tools get raw model for efficiency, agents use LoggingLiteLLMModel for decision tracking
        from ..toolkits.model_utils import get_raw_model
        raw_model = get_raw_model(model)
        
        # STREAMLINED TOOLS - LaTeX workflow with citation support
        # Focus on LaTeX writing, compilation, and citation management
        tools = [
            # CORE LaTeX WORKFLOW (ESSENTIAL - 6 tools)
            LaTeXGeneratorTool(model=raw_model, working_dir=workspace_dir),      # THE CONTENT CREATION BRAIN
            LaTeXReflectionTool(model=raw_model, working_dir=workspace_dir),     # THE QUALITY GUARDIAN
            LaTeXCompilerTool(model=raw_model, working_dir=workspace_dir),       # PDF compilation with BibTeX support
            LaTeXContentVerificationTool(working_dir=workspace_dir),             # Success criteria verification
            LaTeXSyntaxCheckerTool(working_dir=workspace_dir),                   # Document structure validation

            # PDF VALIDATION (1 tool)
            VLMDocumentAnalysisTool(model=raw_model, working_dir=workspace_dir), # PDF validation and analysis

            # NOTE: Citation resolution handled automatically by LaTeXCompilerTool during compilation
            # NOTE: Plotting tools handled by ResourcePreparationAgent (see prep agent for all plotting)
            # NOTE: Data organization handled by ResourcePreparationAgent (see resource_inventory.md)
        ]
        
        # Add file editing tools if workspace_dir is provided
        if workspace_dir:
            file_editing_tools = [
                SeeFile(working_dir=workspace_dir),
                CreateFileWithContent(working_dir=workspace_dir),
                ModifyFile(working_dir=workspace_dir),
                ListDir(working_dir=workspace_dir),
                SearchKeyword(working_dir=workspace_dir),
                DeleteFileOrFolder(working_dir=workspace_dir),
            ]
            tools.extend(file_editing_tools)  # Adds 6 file management tools

        # FINAL TOOL COUNT: 7 core tools + 1 citation tool + 6 file tools = 14 total tools
        # Focused on LaTeX workflow with proper citation support
        
        # Generate complete system prompt using template
        system_prompt = get_writeup_system_prompt(
            tools=tools,
            managed_agents=None  # WriteupAgent typically doesn't manage other agents
        )
        
        # Context management is now automatically handled by BaseResearchAgent
        # Model-specific thresholds are calculated automatically
        # Can still override with max_context_tokens for backward compatibility
        max_context_tokens = kwargs.pop('max_context_tokens', None)
        if max_context_tokens:
            kwargs['token_threshold'] = max_context_tokens
        
        # Combine additional_authorized_imports
        default_imports = ['json', 'os', 'subprocess', 'tempfile', 'shutil', 'pathlib', 'glob', 'numpy', 'numpy.random', 'matplotlib', 'matplotlib.pyplot', 'pandas', 'seaborn', 'scipy', 'scipy.stats', 'sklearn']
        passed_imports = kwargs.pop('additional_authorized_imports', [])
        combined_imports = list(set(default_imports + passed_imports))
        
        # Create success criteria validation function for final_answer_checks
        def validate_success_criteria(final_answer, memory, agent=None):
            """Validate success criteria before allowing termination."""
            return self._validate_writeup_success_criteria(
                final_answer, memory, agent=agent
            )
        
        # Initialize BaseResearchAgent with specialized tools and system prompt
        # Context management now automatically integrated via BaseResearchAgent
        super().__init__(
            model=model,  # Pass original model, BaseResearchAgent will handle logging
            agent_name="writeup_agent",
            workspace_dir=workspace_dir,
            tools=tools,
            additional_authorized_imports=combined_imports,
            max_steps=150,  # Increased from default 20 to allow comprehensive paper writing
            final_answer_checks=[validate_success_criteria],  # CRITICAL: Enable quality gate validation
            **kwargs
        )

        # Set system prompt after initialization (correct smolagents pattern)
        self.prompt_templates["system_prompt"] = system_prompt

        # Resume memory if possible
        self.resume_memory()

    def _validate_writeup_success_criteria(self, final_answer, memory, agent=None):
        """
        Validate success criteria before allowing termination.
        
        Args:
            final_answer: The proposed final answer from the agent
            memory: The agent's memory containing conversation history
            
        Returns:
            bool: True if success criteria are met, False otherwise
            
        This method blocks termination if success criteria aren't met by raising
        an exception with detailed feedback to help the agent continue working.
        """
        print("\n🔍 MANDATORY SUCCESS CRITERIA VERIFICATION")
        print("=" * 60)
        
        # Import here to avoid circular imports
        from ..toolkits.writeup.latex_content_verification_tool import LaTeXContentVerificationTool
        
        workspace_root = os.path.abspath(self.workspace_dir or ".")

        # Resolve final artifacts from either workspace root or paper_workspace/.
        tex_path = self._resolve_paper_artifact_path("final_paper.tex")
        pdf_path = self._resolve_paper_artifact_path("final_paper.pdf")
        
        if tex_path is None:
            print("❌ TERMINATION BLOCKED: final_paper.tex does not exist")
            print("\n📋 REQUIRED ACTIONS:")
            print("1. Create individual sections using LaTeXGeneratorTool")
            print("2. Apply iterative reflection to improve each section") 
            print("3. Create final_paper.tex using LaTeXGeneratorTool with section_type='main_document'")
            print("4. Ensure final_paper.tex uses \\input{} for all sections")
            print("5. Compile final_paper.tex to PDF")
            raise ValueError("TERMINATION_BLOCKED: Missing final_paper.tex. Please create the complete LaTeX document first.")
        
        # Check if final_paper.pdf exists and is valid
        if pdf_path is None:
            print("❌ TERMINATION BLOCKED: final_paper.pdf does not exist")
            print("\n📋 CRITICAL FAILURE: LaTeX compilation failed or was not attempted")
            print("1. You MUST successfully compile final_paper.tex to PDF using LaTeXCompilerTool")
            print("2. If compilation fails with errors, you MUST fix the LaTeX errors")
            print("3. If compilation times out, check for infinite loops or simplify the document")
            print("4. NEVER declare task complete without a valid final_paper.pdf")
            print("\n⚠️  Task Status: FAILED - No PDF produced")
            raise ValueError("TERMINATION_BLOCKED: Missing final_paper.pdf. LaTeX compilation failed. The task cannot be completed without a valid PDF.")
        
        # Verify PDF is valid using VLMDocumentAnalysisTool
        print("🔍 Validating PDF integrity...")
        from ..toolkits.writeup.vlm_document_analysis_tool import VLMDocumentAnalysisTool
        import json
        
        try:
            # Use VLM tool to validate PDF (it will check if PDF can be opened and read)
            vlm_tool = VLMDocumentAnalysisTool(model=self.model, working_dir=self.workspace_dir)
            pdf_rel_path = os.path.relpath(pdf_path, workspace_root)
            validation_result_str = vlm_tool.forward(
                file_paths=pdf_rel_path,
                analysis_focus="pdf_validation"
            )
            
            # Parse validation result
            validation_result = json.loads(validation_result_str)
            
            # Check for errors in PDF processing
            if validation_result.get("error"):
                print(f"❌ TERMINATION BLOCKED: PDF validation failed - {validation_result.get('error')}")
                print("\n📋 CRITICAL FAILURE: PDF is corrupted or unreadable")
                print("1. The PDF file exists but cannot be properly read")
                print("2. Check LaTeX compilation logs for errors")
                print("3. Ensure LaTeX compilation completed successfully")
                print("4. Try recompiling with simplified content if timeout occurs")
                print("\n⚠️  Task Status: FAILED - Corrupted PDF")
                raise ValueError(f"TERMINATION_BLOCKED: PDF validation failed - {validation_result.get('error')}. Fix compilation and regenerate.")
            
            critical_issues = self._extract_pdf_critical_issues(validation_result)
            if critical_issues:
                print("❌ TERMINATION BLOCKED: PDF has critical issues")
                print("\n📋 CRITICAL ISSUES FOUND IN PDF:")
                for issue in critical_issues:
                    print(f"  • {issue}")
                print("\n⚠️  Task Status: FAILED - PDF not publication ready")
                issue_summary = ", ".join(critical_issues[:3])
                raise ValueError(f"TERMINATION_BLOCKED: PDF has critical issues: {issue_summary}. Fix these issues and regenerate.")

        except ValueError:
            raise
        except Exception as e:
            print(f"❌ TERMINATION BLOCKED: PDF validation error - {str(e)}")
            print("\n📋 CRITICAL FAILURE: Unable to validate PDF")
            print("1. PDF may be corrupted or incomplete")
            print("2. Ensure LaTeX compilation completed without errors")
            print("\n⚠️  Task Status: FAILED - PDF validation error")
            raise ValueError(f"TERMINATION_BLOCKED: PDF validation failed with error: {str(e)}. Regenerate the PDF.")
        
        # Run mandatory verification
        print("🔧 Running LaTeXContentVerificationTool...")
        verification_tool = LaTeXContentVerificationTool(working_dir=self.workspace_dir)
        tex_rel_path = os.path.relpath(tex_path, workspace_root)
        verification_result_str = verification_tool.forward(tex_rel_path)
        
        try:
            import json
            verification_result = json.loads(verification_result_str)
        except:
            print("⚠️  Warning: Could not parse verification result")
            verification_result = {"overall_assessment": {"all_criteria_met": False}}
        
        all_criteria_met = verification_result.get("overall_assessment", {}).get("all_criteria_met", False)
        
        if all_criteria_met:
            print("✅ SUCCESS CRITERIA VERIFIED - Termination allowed")
            print("=" * 60)
            # Return True to allow termination
            return True
        else:
            print("❌ TERMINATION BLOCKED: Success criteria not met")
            print("=" * 60)
            
            # Extract specific failure details for actionable feedback
            criteria_breakdown = verification_result.get("overall_assessment", {}).get("criteria_breakdown", {})
            recommendations = verification_result.get("recommendations", [])
            section_analysis = verification_result.get("section_analysis", {})
            content_quality = verification_result.get("content_quality", {})
            
            # Provide detailed feedback
            print("\n📋 CRITICAL ISSUES TO FIX:")
            
            # File existence issues
            file_checks = verification_result.get("file_checks", {})
            if not file_checks.get("pdf_exists", True):
                print("• Missing final_paper.pdf - Compile LaTeX to PDF")
            if not file_checks.get("bib_exists", True):
                print("• Missing references.bib - Create bibliography file")
            
            # Section completeness issues
            print("\n📝 MISSING OR INADEQUATE SECTIONS:")
            for section, analysis in section_analysis.items():
                if isinstance(analysis, dict) and not analysis.get("found", True):
                    print(f"• Missing {section} section")
                elif isinstance(analysis, dict) and not analysis.get("has_substantial_content", True):
                    print(f"• {section} section needs more content ({analysis.get('content_chars', 0)} chars)")
            
            # Content quality issues
            if not (content_quality.get("has_figures", False) or content_quality.get("has_tables", False)):
                print("• Missing visual evidence (add at least one figure or table)")
            if not content_quality.get("has_citations", True):
                print(f"• Missing citations (found: {content_quality.get('citation_count', 0)})")
            
            # Length requirement
            total_chars = section_analysis.get("total_content_chars", 0)
            if total_chars < 15000:
                print(f"• Content too short: {total_chars} chars (need: >15,000 chars)")
            
            # Specific recommendations
            if recommendations:
                print("\n🔧 RECOMMENDED ACTIONS:")
                for i, rec in enumerate(recommendations[:10], 1):  # Limit to top 10
                    print(f"{i}. {rec}")
            
            print("\n⚠️  You must fix these issues before the task can be completed.")
            print("Use your available tools to address each issue, then try to complete again.")
            print("=" * 60)
            
            # Raise an exception to block termination with detailed feedback
            feedback_message = f"""TERMINATION BLOCKED: Success criteria verification failed.

CRITICAL ISSUES FOUND:
- Files missing: {not file_checks.get('pdf_exists', True) or not file_checks.get('bib_exists', True)}
- Sections incomplete: {sum(1 for s, a in section_analysis.items() if isinstance(a, dict) and (not a.get('found', True) or not a.get('has_substantial_content', True)))} sections need work
- Content length: {total_chars}/15000+ characters required
- Visual evidence present: {content_quality.get('has_figures', False) or content_quality.get('has_tables', False)}
- Citations missing: {content_quality.get('citation_count', 0)} found

Please fix these issues using your available tools and try again."""
            
            # Raise exception to prevent termination and force agent to continue
            raise ValueError(feedback_message)

    def _resolve_paper_artifact_path(self, filename: str) -> Optional[str]:
        """Find artifact in root workspace or paper_workspace."""
        workspace_root = os.path.abspath(self.workspace_dir or ".")
        candidates = [
            os.path.join(workspace_root, filename),
            os.path.join(workspace_root, "paper_workspace", filename),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    def _extract_pdf_critical_issues(self, validation_result: dict) -> list[str]:
        """
        Extract critical issues from either VLM schema:
        - structured: {"pdf_validation": {...}}
        - comprehensive: {"publication_issues": [...]}
        """
        issues: list[str] = []

        # Preferred structured schema from _extract_pdf_validation_results
        pdf_validation = validation_result.get("pdf_validation")
        if isinstance(pdf_validation, dict):
            issues.extend(pdf_validation.get("layout_issues", []) or [])
            issues.extend(pdf_validation.get("missing_citations", []) or [])
            issues.extend(pdf_validation.get("missing_figures", []) or [])
            issues.extend(pdf_validation.get("structural_problems", []) or [])
        else:
            # Fallback schema from comprehensive PDF analysis
            publication_issues = validation_result.get("publication_issues", [])
            if isinstance(publication_issues, list):
                issues.extend(publication_issues)

            # Legacy top-level buckets, if present
            issues.extend(validation_result.get("layout_issues", []) or [])
            issues.extend(validation_result.get("missing_citations", []) or [])
            issues.extend(validation_result.get("missing_figures", []) or [])
            issues.extend(validation_result.get("structural_problems", []) or [])

        # Remove non-blocking heuristic warnings that frequently produce false failures.
        filtered_issues = []
        for issue in issues:
            issue_text = str(issue).strip()
            if not issue_text:
                continue
            if "no images found in pdf" in issue_text.lower():
                continue
            filtered_issues.append(issue_text)

        return filtered_issues
