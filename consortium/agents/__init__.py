"""
Research agent node modules — LangGraph migration.

Each module exposes build_node(model, workspace_dir, **cfg) -> Callable.
"""

from .base_agent import create_specialist_agent
from .experiment_design_agent import build_node as build_experiment_design_node
from .experiment_literature_agent import build_node as build_experiment_literature_node
from .experiment_transcription_agent import build_node as build_experiment_transcription_node
from .experiment_verification_agent import build_node as build_experiment_verification_node
from .ideation_agent import build_node as build_ideation_node
from .literature_review_agent import build_node as build_literature_review_node
from .research_planner_agent import build_node as build_research_planner_node
from .results_analysis_agent import build_node as build_results_analysis_node
from .experimentation_agent import build_node as build_experimentation_node
from .resource_preparation_agent import build_node as build_resource_preparation_node
from .writeup_agent import build_node as build_writeup_node
from .proofreading_agent import build_node as build_proofreading_node
from .reviewer_agent import build_node as build_reviewer_node
from .math_literature_agent import build_node as build_math_literature_node
from .math_proposer_agent import build_node as build_math_proposer_node
from .math_prover_agent import build_node as build_math_prover_node
from .math_rigorous_verifier_agent import build_node as build_math_rigorous_verifier_node
from .math_empirical_verifier_agent import build_node as build_math_empirical_verifier_node
from .proof_transcription_agent import build_node as build_proof_transcription_node
from .manager_agent import build_node as build_manager_node
from .track_merge_node import build_node as build_track_merge_node

__all__ = [
    "create_specialist_agent",
    "build_experiment_design_node",
    "build_experiment_literature_node",
    "build_experiment_transcription_node",
    "build_experiment_verification_node",
    "build_ideation_node",
    "build_literature_review_node",
    "build_research_planner_node",
    "build_results_analysis_node",
    "build_experimentation_node",
    "build_resource_preparation_node",
    "build_writeup_node",
    "build_proofreading_node",
    "build_reviewer_node",
    "build_math_literature_node",
    "build_math_proposer_node",
    "build_math_prover_node",
    "build_math_rigorous_verifier_node",
    "build_math_empirical_verifier_node",
    "build_proof_transcription_node",
    "build_manager_node",
    "build_track_merge_node",
]
