"""
Prompt templates for freephdlabor package.

Organized by function rather than agent for maximum reusability.
"""

# Available instruction modules
from freephdlabor.prompts.ideation_instructions import IDEATION_INSTRUCTIONS
from freephdlabor.prompts.experimentation_instructions import EXPERIMENTATION_INSTRUCTIONS
from freephdlabor.prompts.manager_instructions import MANAGER_INSTRUCTIONS
from freephdlabor.prompts.writeup_instructions import WRITEUP_INSTRUCTIONS
from freephdlabor.prompts.literature_review_instructions import LITERATURE_REVIEW_INSTRUCTIONS
from freephdlabor.prompts.research_planner_instructions import RESEARCH_PLANNER_INSTRUCTIONS
from freephdlabor.prompts.results_analysis_instructions import RESULTS_ANALYSIS_INSTRUCTIONS
from freephdlabor.prompts.math_literature_instructions import MATH_LITERATURE_INSTRUCTIONS
from freephdlabor.prompts.proof_transcription_instructions import PROOF_TRANSCRIPTION_INSTRUCTIONS

# Workspace management functions
from freephdlabor.prompts.workspace_management import WORKSPACE_GUIDANCE

__all__ = [
    'IDEATION_INSTRUCTIONS',
    'EXPERIMENTATION_INSTRUCTIONS',
    'MANAGER_INSTRUCTIONS', 
    'WRITEUP_INSTRUCTIONS',
    'LITERATURE_REVIEW_INSTRUCTIONS',
    'RESEARCH_PLANNER_INSTRUCTIONS',
    'RESULTS_ANALYSIS_INSTRUCTIONS',
    'MATH_LITERATURE_INSTRUCTIONS',
    'PROOF_TRANSCRIPTION_INSTRUCTIONS',
    'WORKSPACE_GUIDANCE'
]