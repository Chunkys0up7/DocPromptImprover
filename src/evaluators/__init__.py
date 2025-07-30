"""
Evaluators package for evaluation-only document extraction framework.
"""

from .evaluation_signatures import (
    FieldEvaluationSignature,
    DocumentAggregationSignature,
    FailurePatternAnalysisSignature,
    PromptOptimizationSignature,
    ConfidenceCalibrationSignature,
    ErrorCategorizationSignature,
    PerformanceTrendAnalysisSignature,
    QualityAssessmentSignature,
    OptimizationFeedbackSignature,
    get_evaluation_signature,
    list_available_evaluation_signatures,
    EVALUATION_SIGNATURE_REGISTRY
)

__all__ = [
    "FieldEvaluationSignature",
    "DocumentAggregationSignature", 
    "FailurePatternAnalysisSignature",
    "PromptOptimizationSignature",
    "ConfidenceCalibrationSignature",
    "ErrorCategorizationSignature",
    "PerformanceTrendAnalysisSignature",
    "QualityAssessmentSignature",
    "OptimizationFeedbackSignature",
    "get_evaluation_signature",
    "list_available_evaluation_signatures",
    "EVALUATION_SIGNATURE_REGISTRY"
] 