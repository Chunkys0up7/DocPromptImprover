"""
DSPy Signature Definitions for Document Extraction Evaluation

This module defines all DSPy signatures used for AI-powered document evaluation,
pattern analysis, and prompt optimization.
"""

import dspy
from typing import Optional, List, Dict, Any


class FieldEvaluationSignature(dspy.Signature):
    """
    Evaluate a single field extraction with AI-powered analysis.
    
    This signature provides intelligent evaluation of individual field extractions,
    considering context, field type, and extraction quality.
    """
    
    field_name = dspy.InputField(desc="Name of the field being evaluated")
    expected_value = dspy.InputField(desc="Ground truth value for the field")
    extracted_value = dspy.InputField(desc="Value extracted by the model")
    confidence_score = dspy.InputField(desc="Confidence score from the extraction model (0.0-1.0)")
    field_type = dspy.InputField(desc="Expected data type: text, number, date, email, phone")
    
    evaluation_score = dspy.OutputField(desc="Evaluation score between 0.0 and 1.0, where 1.0 is perfect match")
    status = dspy.OutputField(desc="Extraction status: success, partial, failed, or missing")
    error_message = dspy.OutputField(desc="Detailed error message if evaluation failed, or null if successful")
    evaluation_notes = dspy.OutputField(desc="Detailed evaluation notes explaining the score and reasoning")
    confidence_assessment = dspy.OutputField(desc="Assessment of whether the confidence score is appropriate")


class DocumentAggregationSignature(dspy.Signature):
    """
    Aggregate field evaluations into document-level metrics.
    
    This signature combines individual field evaluations to provide
    comprehensive document-level assessment and insights.
    """
    
    field_evaluations = dspy.InputField(desc="JSON string containing list of field evaluation results")
    document_type = dspy.InputField(desc="Type of document: invoice, receipt, form, contract, medical_record, bank_statement")
    confidence_scores = dspy.InputField(desc="JSON string of confidence scores for each field")
    prompt_version = dspy.InputField(desc="Version of the prompt used for extraction")
    
    overall_accuracy = dspy.OutputField(desc="Overall document accuracy score (0.0-1.0)")
    confidence_correlation = dspy.OutputField(desc="Correlation between confidence scores and actual accuracy (-1.0 to 1.0)")
    quality_assessment = dspy.OutputField(desc="Overall quality assessment: excellent, good, fair, poor")
    critical_errors = dspy.OutputField(desc="List of critical errors that significantly impact document quality")
    improvement_suggestions = dspy.OutputField(desc="Specific suggestions for improving extraction quality")


class FailurePatternAnalysisSignature(dspy.Signature):
    """
    Analyze failure patterns across multiple evaluations.
    
    This signature identifies common patterns in extraction failures
    and provides insights for improvement.
    """
    
    evaluation_results = dspy.InputField(desc="JSON string containing multiple document evaluation results")
    document_types = dspy.InputField(desc="List of document types in the evaluation set")
    time_period = dspy.InputField(desc="Time period covered by the evaluations")
    
    common_patterns = dspy.OutputField(desc="List of common failure patterns identified")
    pattern_frequency = dspy.OutputField(desc="Frequency of each pattern as percentage of total failures")
    pattern_severity = dspy.OutputField(desc="Severity assessment for each pattern: high, medium, low")
    root_causes = dspy.OutputField(desc="Root causes for each failure pattern")
    suggested_fixes = dspy.OutputField(desc="Specific fixes or improvements for each pattern")


class ConfidenceCalibrationSignature(dspy.Signature):
    """
    Calibrate confidence scores based on actual accuracy.
    
    This signature analyzes the relationship between confidence scores
    and actual accuracy to improve confidence calibration.
    """
    
    confidence_scores = dspy.InputField(desc="List of confidence scores from extractions")
    actual_accuracy = dspy.InputField(desc="List of actual accuracy scores for the same extractions")
    field_types = dspy.InputField(desc="List of field types corresponding to each score")
    
    calibration_factor = dspy.OutputField(desc="Calibration factor to apply to confidence scores")
    calibrated_scores = dspy.OutputField(desc="List of calibrated confidence scores")
    calibration_quality = dspy.OutputField(desc="Quality of calibration: excellent, good, fair, poor")
    calibration_notes = dspy.OutputField(desc="Notes about the calibration process and recommendations")


class PromptOptimizationSignature(dspy.Signature):
    """
    Generate optimized prompts based on evaluation results.
    
    This signature creates improved prompts by analyzing failure patterns
    and incorporating best practices.
    """
    
    current_prompt = dspy.InputField(desc="Current prompt being used for extraction")
    evaluation_statistics = dspy.InputField(desc="JSON string of evaluation statistics and metrics")
    failure_patterns = dspy.InputField(desc="JSON string of identified failure patterns")
    target_improvement = dspy.InputField(desc="Target improvement percentage (0.0-1.0)")
    document_type = dspy.InputField(desc="Type of document the prompt is used for")
    
    optimized_prompt = dspy.OutputField(desc="Optimized version of the prompt with improvements")
    improvement_rationale = dspy.OutputField(desc="Detailed explanation of improvements made and why")
    expected_improvement = dspy.OutputField(desc="Expected improvement percentage based on changes")
    confidence_in_improvement = dspy.OutputField(desc="Confidence level in the expected improvement (0.0-1.0)")
    implementation_notes = dspy.OutputField(desc="Notes for implementing the optimized prompt")


class ErrorCategorizationSignature(dspy.Signature):
    """
    Categorize and classify extraction errors.
    
    This signature provides intelligent categorization of errors
    to help understand and address specific issues.
    """
    
    error_message = dspy.InputField(desc="Error message or description")
    field_name = dspy.InputField(desc="Name of the field where error occurred")
    field_type = dspy.InputField(desc="Type of field: text, number, date, email, phone")
    document_type = dspy.InputField(desc="Type of document being processed")
    context = dspy.InputField(desc="Additional context about the error")
    
    error_category = dspy.OutputField(desc="Primary error category: format, missing, partial, wrong_type, other")
    error_subcategory = dspy.OutputField(desc="Specific subcategory of the error")
    severity_level = dspy.OutputField(desc="Severity level: critical, high, medium, low")
    suggested_action = dspy.OutputField(desc="Suggested action to fix this type of error")
    prevention_strategy = dspy.OutputField(desc="Strategy to prevent similar errors in the future")


class PerformanceTrendAnalysisSignature(dspy.Signature):
    """
    Analyze performance trends over time.
    
    This signature identifies trends in extraction performance
    and provides insights for continuous improvement.
    """
    
    historical_data = dspy.InputField(desc="JSON string of historical evaluation data over time")
    time_period = dspy.InputField(desc="Time period covered by the data")
    document_types = dspy.InputField(desc="Document types included in the analysis")
    
    performance_trend = dspy.OutputField(desc="Overall performance trend: improving, declining, stable")
    trend_strength = dspy.OutputField(desc="Strength of the trend: strong, moderate, weak")
    key_insights = dspy.OutputField(desc="Key insights about performance changes")
    contributing_factors = dspy.OutputField(desc="Factors contributing to the observed trends")
    recommendations = dspy.OutputField(desc="Recommendations for maintaining or improving performance")


class QualityAssessmentSignature(dspy.Signature):
    """
    Comprehensive quality assessment of extraction results.
    
    This signature provides a holistic quality assessment
    considering multiple factors and dimensions.
    """
    
    field_evaluations = dspy.InputField(desc="JSON string of field evaluation results")
    document_metadata = dspy.InputField(desc="Document metadata including type, complexity, etc.")
    extraction_metadata = dspy.InputField(desc="Extraction process metadata")
    
    overall_quality_score = dspy.OutputField(desc="Overall quality score (0.0-1.0)")
    quality_dimensions = dspy.OutputField(desc="Quality scores for different dimensions: accuracy, completeness, consistency")
    quality_grade = dspy.OutputField(desc="Quality grade: A, B, C, D, F")
    quality_issues = dspy.OutputField(desc="List of quality issues identified")
    improvement_priorities = dspy.OutputField(desc="Prioritized list of improvements needed")


class OptimizationFeedbackSignature(dspy.Signature):
    """
    Generate feedback for optimization processes.
    
    This signature provides detailed feedback on optimization attempts
    and guides future improvement efforts.
    """
    
    optimization_attempt = dspy.InputField(desc="Details of the optimization attempt")
    before_metrics = dspy.InputField(desc="Performance metrics before optimization")
    after_metrics = dspy.InputField(desc="Performance metrics after optimization")
    changes_made = dspy.InputField(desc="Changes made during optimization")
    
    optimization_success = dspy.OutputField(desc="Whether the optimization was successful: yes, partial, no")
    improvement_metrics = dspy.OutputField(desc="Quantified improvements in key metrics")
    lessons_learned = dspy.OutputField(desc="Key lessons learned from this optimization")
    next_steps = dspy.OutputField(desc="Recommended next steps for further optimization")
    risk_assessment = dspy.OutputField(desc="Assessment of risks associated with the changes")


class ContextAwareEvaluationSignature(dspy.Signature):
    """
    Context-aware evaluation considering document context.
    
    This signature evaluates extractions considering the broader
    context of the document and field relationships.
    """
    
    field_name = dspy.InputField(desc="Name of the field being evaluated")
    field_value = dspy.InputField(desc="Extracted value for the field")
    document_context = dspy.InputField(desc="Context information about the document")
    related_fields = dspy.InputField(desc="Values of related fields in the document")
    business_rules = dspy.InputField(desc="Business rules or constraints that apply")
    
    context_score = dspy.OutputField(desc="Context-aware evaluation score (0.0-1.0)")
    context_validation = dspy.OutputField(desc="Validation of the value in context")
    consistency_check = dspy.OutputField(desc="Consistency with related fields")
    business_rule_compliance = dspy.OutputField(desc="Compliance with business rules")
    context_notes = dspy.OutputField(desc="Notes about context considerations")


class MultiLanguageEvaluationSignature(dspy.Signature):
    """
    Evaluate extractions in multiple languages.
    
    This signature handles evaluation of extractions from
    documents in different languages.
    """
    
    field_name = dspy.InputField(desc="Name of the field being evaluated")
    expected_value = dspy.InputField(desc="Ground truth value")
    extracted_value = dspy.InputField(desc="Extracted value")
    language = dspy.InputField(desc="Language of the document")
    field_type = dspy.InputField(desc="Type of field")
    
    language_aware_score = dspy.OutputField(desc="Language-aware evaluation score (0.0-1.0)")
    language_specific_issues = dspy.OutputField(desc="Language-specific issues identified")
    cultural_considerations = dspy.OutputField(desc="Cultural considerations for the language")
    translation_quality = dspy.OutputField(desc="Quality of any translations involved")
    language_notes = dspy.OutputField(desc="Notes about language-specific evaluation")


# Registry of all signatures for easy access
SIGNATURE_REGISTRY = {
    "field_evaluation": FieldEvaluationSignature,
    "document_aggregation": DocumentAggregationSignature,
    "failure_pattern_analysis": FailurePatternAnalysisSignature,
    "confidence_calibration": ConfidenceCalibrationSignature,
    "prompt_optimization": PromptOptimizationSignature,
    "error_categorization": ErrorCategorizationSignature,
    "performance_trend_analysis": PerformanceTrendAnalysisSignature,
    "quality_assessment": QualityAssessmentSignature,
    "optimization_feedback": OptimizationFeedbackSignature,
    "context_aware_evaluation": ContextAwareEvaluationSignature,
    "multi_language_evaluation": MultiLanguageEvaluationSignature,
}


def get_signature(signature_name: str):
    """Get a signature by name."""
    return SIGNATURE_REGISTRY.get(signature_name)


def list_available_signatures():
    """List all available signatures."""
    return list(SIGNATURE_REGISTRY.keys())


def validate_signature_inputs(signature_name: str, inputs: Dict[str, Any]) -> bool:
    """Validate inputs for a specific signature."""
    signature_class = get_signature(signature_name)
    if not signature_class:
        return False
    
    # Check if all required input fields are present
    required_fields = signature_class.input_fields.keys()
    provided_fields = inputs.keys()
    
    return all(field in provided_fields for field in required_fields) 