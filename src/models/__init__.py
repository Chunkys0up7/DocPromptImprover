"""
Models package for evaluation-only document extraction framework.
"""

from .evaluation_models import (
    ExtractionStatus,
    FieldEvaluationResult,
    DocumentEvaluationInput,
    DocumentEvaluationResult,
    EvaluationStatistics,
    FailurePattern,
    OptimizationRecommendation,
    EvaluationConfig
)

__all__ = [
    "ExtractionStatus",
    "FieldEvaluationResult", 
    "DocumentEvaluationInput",
    "DocumentEvaluationResult",
    "EvaluationStatistics",
    "FailurePattern",
    "OptimizationRecommendation",
    "EvaluationConfig"
] 