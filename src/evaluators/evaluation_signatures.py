"""
DSPy signatures for evaluation-only document extraction framework.

This module defines the signatures that specify the input-output behavior
for various evaluation tasks, enabling automatic optimization of evaluation metrics.
"""

import dspy
from typing import Dict, Any, Optional, List


class FieldEvaluationSignature(dspy.Signature):
    """
    Evaluate the accuracy of a single field extraction.
    
    This signature defines the task of comparing an extracted field value
    against ground truth and providing a detailed evaluation score.
    """
    
    field_name = dspy.InputField(
        desc="Name of the field being evaluated"
    )
    expected_value = dspy.InputField(
        desc="Ground truth value for the field"
    )
    extracted_value = dspy.InputField(
        desc="Value extracted by the model"
    )
    confidence_score = dspy.InputField(
        desc="Confidence score from the extraction model (0.0-1.0)"
    )
    field_type = dspy.InputField(
        desc="Expected data type of the field (text, number, date, etc.)"
    )
    
    evaluation_score = dspy.OutputField(
        desc="Evaluation score (0.0-1.0) based on accuracy comparison"
    )
    status = dspy.OutputField(
        desc="Evaluation status: success, partial, failed, or missing"
    )
    error_message = dspy.OutputField(
        desc="Error message if extraction failed, otherwise empty"
    )
    evaluation_notes = dspy.OutputField(
        desc="Detailed notes about the evaluation reasoning"
    )


class DocumentAggregationSignature(dspy.Signature):
    """
    Aggregate field-level evaluations into document-level metrics.
    
    This signature combines individual field evaluation results to produce
    overall document accuracy and performance metrics.
    """
    
    field_evaluations = dspy.InputField(
        desc="List of field evaluation results with scores and statuses"
    )
    document_type = dspy.InputField(
        desc="Type of document being evaluated"
    )
    confidence_scores = dspy.InputField(
        desc="Confidence scores for all extracted fields"
    )
    
    overall_accuracy = dspy.OutputField(
        desc="Overall accuracy score (0.0-1.0) for the document"
    )
    confidence_correlation = dspy.OutputField(
        desc="Correlation between confidence scores and accuracy (0.0-1.0)"
    )
    quality_assessment = dspy.OutputField(
        desc="Overall quality assessment of the extraction"
    )
    improvement_suggestions = dspy.OutputField(
        desc="Suggestions for improving extraction quality"
    )


class FailurePatternAnalysisSignature(dspy.Signature):
    """
    Analyze failure patterns across multiple evaluations.
    
    This signature identifies common failure patterns and provides
    insights for prompt optimization.
    """
    
    failed_evaluations = dspy.InputField(
        desc="List of failed field evaluations with error messages"
    )
    document_types = dspy.InputField(
        desc="Types of documents in the evaluation set"
    )
    field_names = dspy.InputField(
        desc="Names of fields that failed extraction"
    )
    
    failure_patterns = dspy.OutputField(
        desc="Identified failure patterns with descriptions"
    )
    pattern_frequency = dspy.OutputField(
        desc="Frequency of each failure pattern"
    )
    root_causes = dspy.OutputField(
        desc="Root causes for the identified patterns"
    )
    optimization_priorities = dspy.OutputField(
        desc="Prioritized list of optimization opportunities"
    )


class PromptOptimizationSignature(dspy.Signature):
    """
    Generate prompt optimization recommendations based on evaluation results.
    
    This signature analyzes evaluation statistics and failure patterns to
    generate specific recommendations for prompt improvement.
    """
    
    evaluation_statistics = dspy.InputField(
        desc="Aggregated statistics from document evaluations"
    )
    failure_patterns = dspy.InputField(
        desc="Identified failure patterns and their frequencies"
    )
    current_prompt = dspy.InputField(
        desc="Current prompt being used for extraction"
    )
    target_improvement = dspy.InputField(
        desc="Target improvement percentage in accuracy"
    )
    
    optimized_prompt = dspy.OutputField(
        desc="Improved version of the prompt"
    )
    improvement_rationale = dspy.OutputField(
        desc="Explanation of the improvements made"
    )
    expected_improvement = dspy.OutputField(
        desc="Expected improvement in accuracy (0.0-1.0)"
    )
    confidence_in_improvement = dspy.OutputField(
        desc="Confidence in the expected improvement (0.0-1.0)"
    )


class ConfidenceCalibrationSignature(dspy.Signature):
    """
    Calibrate confidence scores based on evaluation results.
    
    This signature adjusts confidence scores to better correlate with
    actual accuracy based on historical evaluation data.
    """
    
    field_name = dspy.InputField(
        desc="Name of the field being calibrated"
    )
    raw_confidence_scores = dspy.InputField(
        desc="Raw confidence scores from the model"
    )
    actual_accuracy_scores = dspy.InputField(
        desc="Actual accuracy scores from evaluation"
    )
    historical_data = dspy.InputField(
        desc="Historical confidence vs accuracy data"
    )
    
    calibrated_confidence = dspy.OutputField(
        desc="Calibrated confidence score"
    )
    calibration_factor = dspy.OutputField(
        desc="Factor applied during calibration"
    )
    reliability_indicator = dspy.OutputField(
        desc="Indicator of confidence reliability"
    )


class ErrorCategorizationSignature(dspy.Signature):
    """
    Categorize and classify extraction errors.
    
    This signature analyzes error messages and patterns to categorize
    errors for better understanding and optimization.
    """
    
    error_messages = dspy.InputField(
        desc="List of error messages from failed extractions"
    )
    field_names = dspy.InputField(
        desc="Names of fields that produced errors"
    )
    document_types = dspy.InputField(
        desc="Types of documents with errors"
    )
    
    error_categories = dspy.OutputField(
        desc="Categorized error types and descriptions"
    )
    category_frequency = dspy.OutputField(
        desc="Frequency of each error category"
    )
    severity_assessment = dspy.OutputField(
        desc="Severity assessment for each error category"
    )
    mitigation_strategies = dspy.OutputField(
        desc="Suggested strategies to mitigate each error type"
    )


class PerformanceTrendAnalysisSignature(dspy.Signature):
    """
    Analyze performance trends over time.
    
    This signature analyzes evaluation results over time to identify
    trends, regressions, and improvements in extraction performance.
    """
    
    historical_evaluations = dspy.InputField(
        desc="Historical evaluation results with timestamps"
    )
    prompt_versions = dspy.InputField(
        desc="Prompt versions used over time"
    )
    performance_metrics = dspy.InputField(
        desc="Key performance metrics to analyze"
    )
    
    performance_trends = dspy.OutputField(
        desc="Identified performance trends over time"
    )
    regression_detection = dspy.OutputField(
        desc="Detection of performance regressions"
    )
    improvement_areas = dspy.OutputField(
        desc="Areas showing improvement over time"
    )
    trend_predictions = dspy.OutputField(
        desc="Predictions for future performance trends"
    )


class QualityAssessmentSignature(dspy.Signature):
    """
    Assess overall quality of extraction results.
    
    This signature provides a comprehensive quality assessment
    based on multiple evaluation dimensions.
    """
    
    field_evaluations = dspy.InputField(
        desc="Individual field evaluation results"
    )
    document_metadata = dspy.InputField(
        desc="Metadata about the document being evaluated"
    )
    extraction_context = dspy.InputField(
        desc="Context about the extraction process"
    )
    
    overall_quality_score = dspy.OutputField(
        desc="Overall quality score (0.0-1.0)"
    )
    quality_dimensions = dspy.OutputField(
        desc="Quality scores for different dimensions"
    )
    quality_issues = dspy.OutputField(
        desc="Identified quality issues and concerns"
    )
    quality_recommendations = dspy.OutputField(
        desc="Recommendations for improving quality"
    )


class OptimizationFeedbackSignature(dspy.Signature):
    """
    Generate feedback for optimization based on evaluation results.
    
    This signature analyzes evaluation results to provide actionable
    feedback for prompt optimization.
    """
    
    evaluation_results = dspy.OutputField(
        desc="Results from recent evaluations"
    )
    success_rates = dspy.InputField(
        desc="Success rates for different fields and document types"
    )
    common_errors = dspy.InputField(
        desc="Common errors encountered during evaluation"
    )
    
    optimization_suggestions = dspy.OutputField(
        desc="Suggestions for prompt optimization"
    )
    priority_areas = dspy.OutputField(
        desc="Areas that need immediate attention"
    )
    expected_improvements = dspy.OutputField(
        desc="Expected improvements from suggested changes"
    )


# Signature registry for easy access
EVALUATION_SIGNATURE_REGISTRY = {
    "field_evaluation": FieldEvaluationSignature,
    "document_aggregation": DocumentAggregationSignature,
    "failure_pattern_analysis": FailurePatternAnalysisSignature,
    "prompt_optimization": PromptOptimizationSignature,
    "confidence_calibration": ConfidenceCalibrationSignature,
    "error_categorization": ErrorCategorizationSignature,
    "performance_trend_analysis": PerformanceTrendAnalysisSignature,
    "quality_assessment": QualityAssessmentSignature,
    "optimization_feedback": OptimizationFeedbackSignature,
}


def get_evaluation_signature(signature_name: str):
    """Get an evaluation signature by name from the registry."""
    if signature_name not in EVALUATION_SIGNATURE_REGISTRY:
        raise ValueError(f"Unknown evaluation signature: {signature_name}")
    return EVALUATION_SIGNATURE_REGISTRY[signature_name]


def list_available_evaluation_signatures() -> list:
    """List all available evaluation signatures."""
    return list(EVALUATION_SIGNATURE_REGISTRY.keys()) 