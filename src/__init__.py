"""
Evaluation-Only Framework for Prompt-Driven Document Extraction.

This package provides a lightweight, decoupled evaluation service that assesses
the outputs of existing OCR-plus-prompt pipelines and provides data-driven
feedback for prompt optimization.
"""

__version__ = "1.0.0"
__author__ = "Document Evaluation Team"
__description__ = "Evaluation-only framework for prompt-driven document extraction"

# Core evaluation models
from .models.evaluation_models import (
    ExtractionStatus,
    FieldEvaluationResult,
    DocumentEvaluationInput,
    DocumentEvaluationResult,
    EvaluationStatistics,
    FailurePattern,
    OptimizationRecommendation,
    EvaluationConfig
)

# Evaluation signatures
from .evaluators.evaluation_signatures import (
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

# API service
from .api.evaluation_service import (
    DocumentExtractionEvaluator,
    app,
    evaluator
)

# Utilities
from .utils.config import (
    get_config,
    get_settings,
    validate_config,
    update_config
)

from .utils.logging import (
    get_logger,
    setup_logging,
    DocumentProcessingLogger,
    MetricsLogger
)

__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    "__description__",
    
    # Core models
    "ExtractionStatus",
    "FieldEvaluationResult",
    "DocumentEvaluationInput", 
    "DocumentEvaluationResult",
    "EvaluationStatistics",
    "FailurePattern",
    "OptimizationRecommendation",
    "EvaluationConfig",
    
    # Evaluation signatures
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
    "EVALUATION_SIGNATURE_REGISTRY",
    
    # API service
    "DocumentExtractionEvaluator",
    "app",
    "evaluator",
    
    # Utilities
    "get_config",
    "get_settings", 
    "validate_config",
    "update_config",
    "get_logger",
    "setup_logging",
    "DocumentProcessingLogger",
    "MetricsLogger"
] 