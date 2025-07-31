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

from .feedback_models import (
    FeedbackStatus,
    FeedbackReason,
    FieldFeedback,
    UserFeedbackRecord,
    FeedbackAggregation,
    FeedbackTrend,
    FeedbackAlert,
    FeedbackOptimizationRecommendation,
    FeedbackStatistics
)

__all__ = [
    "ExtractionStatus",
    "FieldEvaluationResult", 
    "DocumentEvaluationInput",
    "DocumentEvaluationResult",
    "EvaluationStatistics",
    "FailurePattern",
    "OptimizationRecommendation",
    "EvaluationConfig",
    "FeedbackStatus",
    "FeedbackReason",
    "FieldFeedback",
    "UserFeedbackRecord",
    "FeedbackAggregation",
    "FeedbackTrend",
    "FeedbackAlert",
    "FeedbackOptimizationRecommendation",
    "FeedbackStatistics"
] 