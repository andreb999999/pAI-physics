"""
Prompt templates for consortium package.
"""

# Re-export active top-level prompt modules.
from .experiment_design_instructions import EXPERIMENT_DESIGN_INSTRUCTIONS
from .experiment_literature_instructions import EXPERIMENT_LITERATURE_INSTRUCTIONS
from .experiment_transcription_instructions import EXPERIMENT_TRANSCRIPTION_INSTRUCTIONS
from .experiment_verification_instructions import EXPERIMENT_VERIFICATION_INSTRUCTIONS
from .experimentation_instructions import EXPERIMENTATION_INSTRUCTIONS
from .ideation_instructions import IDEATION_INSTRUCTIONS
from .literature_review_instructions import LITERATURE_REVIEW_INSTRUCTIONS
from .manager_instructions import MANAGER_INSTRUCTIONS
from .math_literature_instructions import MATH_LITERATURE_INSTRUCTIONS
from .proof_transcription_instructions import PROOF_TRANSCRIPTION_INSTRUCTIONS
from .research_planner_instructions import RESEARCH_PLANNER_INSTRUCTIONS
from .results_analysis_instructions import RESULTS_ANALYSIS_INSTRUCTIONS
from .workspace_management import WORKSPACE_GUIDANCE
from .writeup_instructions import WRITEUP_INSTRUCTIONS

# v2 pipeline prompt modules
from .persona_instructions import (
    PRACTICAL_COMPASS_PERSONA,
    RIGOR_AND_NOVELTY_PERSONA,
    NARRATIVE_ARCHITECT_PERSONA,
    PERSONA_SYNTHESIS_PROMPT,
)
from .duality_check_instructions import (
    DUALITY_CHECK_A_PROMPT,
    DUALITY_CHECK_B_PROMPT,
)
from .brainstorm_instructions import BRAINSTORM_INSTRUCTIONS
from .formalize_goals_instructions import FORMALIZE_GOALS_INSTRUCTIONS
from .formalize_results_instructions import FORMALIZE_RESULTS_INSTRUCTIONS

__all__ = [
    "IDEATION_INSTRUCTIONS",
    "EXPERIMENTATION_INSTRUCTIONS",
    "EXPERIMENT_DESIGN_INSTRUCTIONS",
    "EXPERIMENT_LITERATURE_INSTRUCTIONS",
    "EXPERIMENT_TRANSCRIPTION_INSTRUCTIONS",
    "EXPERIMENT_VERIFICATION_INSTRUCTIONS",
    "MANAGER_INSTRUCTIONS",
    "WRITEUP_INSTRUCTIONS",
    "LITERATURE_REVIEW_INSTRUCTIONS",
    "RESEARCH_PLANNER_INSTRUCTIONS",
    "RESULTS_ANALYSIS_INSTRUCTIONS",
    "MATH_LITERATURE_INSTRUCTIONS",
    "PROOF_TRANSCRIPTION_INSTRUCTIONS",
    "WORKSPACE_GUIDANCE",
    # v2 pipeline
    "PRACTICAL_COMPASS_PERSONA",
    "RIGOR_AND_NOVELTY_PERSONA",
    "NARRATIVE_ARCHITECT_PERSONA",
    "PERSONA_SYNTHESIS_PROMPT",
    "DUALITY_CHECK_A_PROMPT",
    "DUALITY_CHECK_B_PROMPT",
    "BRAINSTORM_INSTRUCTIONS",
    "FORMALIZE_GOALS_INSTRUCTIONS",
    "FORMALIZE_RESULTS_INSTRUCTIONS",
]