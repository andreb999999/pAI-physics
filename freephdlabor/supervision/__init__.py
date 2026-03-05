"""
Supervision module for the Smolagents Research System.

This module provides hierarchical supervision to prevent hallucination and
ensure research quality through multiple validation strategies, and all
quality-gate validation logic for paper, math, and editorial artifacts.
"""

from .supervision_manager import AgentSupervisionManager, SupervisionLevel
from .validation_strategies import (
    ValidationStrategy,
    OutputValidationStrategy,
    AuthenticityCheckingStrategy,
    HallucinationDetectionStrategy,
)
from .result_validation import (
    sanitize_result_payload,
    validate_result_artifacts,
    artifact_exists,
    parse_result_payload,
)
from .review_verdict_validation import validate_review_verdict
from .math_acceptance_validation import validate_math_acceptance
from .paper_traceability_validation import validate_claim_traceability
from .paper_quality_validation import validate_paper_quality

__all__ = [
    "AgentSupervisionManager",
    "SupervisionLevel",
    "ValidationStrategy",
    "OutputValidationStrategy",
    "AuthenticityCheckingStrategy",
    "HallucinationDetectionStrategy",
    "sanitize_result_payload",
    "validate_result_artifacts",
    "artifact_exists",
    "parse_result_payload",
    "validate_review_verdict",
    "validate_math_acceptance",
    "validate_claim_traceability",
    "validate_paper_quality",
]