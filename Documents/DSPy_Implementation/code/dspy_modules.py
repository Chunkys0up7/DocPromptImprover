"""
DSPy Module Implementations for Document Extraction Evaluation

This module implements DSPy modules that use the signatures to provide
AI-powered evaluation, analysis, and optimization capabilities.
"""

import dspy
import json
from typing import Dict, Any, List, Optional
from .dspy_signatures import (
    FieldEvaluationSignature,
    DocumentAggregationSignature,
    FailurePatternAnalysisSignature,
    ConfidenceCalibrationSignature,
    PromptOptimizationSignature,
    ErrorCategorizationSignature,
    PerformanceTrendAnalysisSignature,
    QualityAssessmentSignature,
    OptimizationFeedbackSignature,
    ContextAwareEvaluationSignature,
    MultiLanguageEvaluationSignature
)


class DSPyFieldEvaluator(dspy.Module):
    """
    AI-powered field evaluator using DSPy.
    
    This module provides intelligent evaluation of individual field extractions,
    considering context, field type, and extraction quality.
    """
    
    def __init__(self):
        super().__init__()
        self.field_evaluator = dspy.ChainOfThought(FieldEvaluationSignature)
    
    def forward(self, field_name: str, expected_value: str, extracted_value: str, 
                confidence_score: float, field_type: str) -> Dict[str, Any]:
        """
        Evaluate a single field extraction using AI.
        
        Args:
            field_name: Name of the field being evaluated
            expected_value: Ground truth value
            extracted_value: Value extracted by the model
            confidence_score: Confidence score from extraction
            field_type: Expected data type of the field
            
        Returns:
            Dict containing evaluation results
        """
        
        try:
            result = self.field_evaluator(
                field_name=field_name,
                expected_value=str(expected_value) if expected_value is not None else "",
                extracted_value=str(extracted_value) if extracted_value is not None else "",
                confidence_score=confidence_score,
                field_type=field_type
            )
            
            return {
                "evaluation_score": float(result.evaluation_score),
                "status": result.status.lower(),
                "error_message": result.error_message,
                "evaluation_notes": result.evaluation_notes,
                "confidence_assessment": result.confidence_assessment,
                "method": "ai_powered"
            }
            
        except Exception as e:
            # Fallback to basic evaluation
            return {
                "evaluation_score": 0.0,
                "status": "failed",
                "error_message": f"AI evaluation failed: {str(e)}",
                "evaluation_notes": "Fallback evaluation due to AI error",
                "confidence_assessment": "unable_to_assess",
                "method": "ai_fallback"
            }


class DSPyDocumentAggregator(dspy.Module):
    """
    AI-powered document aggregator using DSPy.
    
    This module combines individual field evaluations to provide
    comprehensive document-level assessment and insights.
    """
    
    def __init__(self):
        super().__init__()
        self.aggregator = dspy.ChainOfThought(DocumentAggregationSignature)
    
    def forward(self, field_evaluations: List[Dict[str, Any]], document_type: str, 
                confidence_scores: Dict[str, float], prompt_version: str = "v1.0") -> Dict[str, Any]:
        """
        Aggregate field evaluations using AI.
        
        Args:
            field_evaluations: List of field evaluation results
            document_type: Type of document being evaluated
            confidence_scores: Confidence scores for each field
            prompt_version: Version of the prompt used
            
        Returns:
            Dict containing aggregated results
        """
        
        try:
            result = self.aggregator(
                field_evaluations=json.dumps(field_evaluations),
                document_type=document_type,
                confidence_scores=json.dumps(confidence_scores),
                prompt_version=prompt_version
            )
            
            return {
                "overall_accuracy": float(result.overall_accuracy),
                "confidence_correlation": float(result.confidence_correlation),
                "quality_assessment": result.quality_assessment,
                "critical_errors": result.critical_errors,
                "improvement_suggestions": result.improvement_suggestions,
                "method": "ai_powered"
            }
            
        except Exception as e:
            # Fallback to statistical aggregation
            return {
                "overall_accuracy": sum(f["evaluation_score"] for f in field_evaluations) / len(field_evaluations) if field_evaluations else 0.0,
                "confidence_correlation": 0.0,
                "quality_assessment": "unable_to_assess",
                "critical_errors": [],
                "improvement_suggestions": [],
                "method": "ai_fallback"
            }


class DSPyFailureAnalyzer(dspy.Module):
    """
    AI-powered failure pattern analyzer using DSPy.
    
    This module identifies common patterns in extraction failures
    and provides insights for improvement.
    """
    
    def __init__(self):
        super().__init__()
        self.pattern_analyzer = dspy.ChainOfThought(FailurePatternAnalysisSignature)
    
    def forward(self, evaluation_results: List[Dict[str, Any]], 
                document_types: List[str], time_period: str = "recent") -> Dict[str, Any]:
        """
        Analyze failure patterns using AI.
        
        Args:
            evaluation_results: List of evaluation results
            document_types: List of document types
            time_period: Time period covered by evaluations
            
        Returns:
            Dict containing pattern analysis results
        """
        
        try:
            result = self.pattern_analyzer(
                evaluation_results=json.dumps(evaluation_results),
                document_types=json.dumps(document_types),
                time_period=time_period
            )
            
            return {
                "common_patterns": result.common_patterns,
                "pattern_frequency": result.pattern_frequency,
                "pattern_severity": result.pattern_severity,
                "root_causes": result.root_causes,
                "suggested_fixes": result.suggested_fixes,
                "method": "ai_powered"
            }
            
        except Exception as e:
            return {
                "common_patterns": [],
                "pattern_frequency": {},
                "pattern_severity": {},
                "root_causes": [],
                "suggested_fixes": [],
                "method": "ai_fallback",
                "error": str(e)
            }


class DSPyConfidenceCalibrator(dspy.Module):
    """
    AI-powered confidence score calibrator using DSPy.
    
    This module analyzes the relationship between confidence scores
    and actual accuracy to improve confidence calibration.
    """
    
    def __init__(self):
        super().__init__()
        self.calibrator = dspy.ChainOfThought(ConfidenceCalibrationSignature)
    
    def forward(self, confidence_scores: List[float], actual_accuracy: List[float], 
                field_types: List[str]) -> Dict[str, Any]:
        """
        Calibrate confidence scores using AI.
        
        Args:
            confidence_scores: List of confidence scores
            actual_accuracy: List of actual accuracy scores
            field_types: List of field types
            
        Returns:
            Dict containing calibration results
        """
        
        try:
            result = self.calibrator(
                confidence_scores=json.dumps(confidence_scores),
                actual_accuracy=json.dumps(actual_accuracy),
                field_types=json.dumps(field_types)
            )
            
            return {
                "calibration_factor": float(result.calibration_factor),
                "calibrated_scores": result.calibrated_scores,
                "calibration_quality": result.calibration_quality,
                "calibration_notes": result.calibration_notes,
                "method": "ai_powered"
            }
            
        except Exception as e:
            return {
                "calibration_factor": 1.0,
                "calibrated_scores": confidence_scores,
                "calibration_quality": "unable_to_assess",
                "calibration_notes": f"Calibration failed: {str(e)}",
                "method": "ai_fallback"
            }


class DSPyPromptOptimizer(dspy.Module):
    """
    AI-powered prompt optimizer using DSPy.
    
    This module creates improved prompts by analyzing failure patterns
    and incorporating best practices.
    """
    
    def __init__(self):
        super().__init__()
        self.optimizer = dspy.ChainOfThought(PromptOptimizationSignature)
    
    def forward(self, current_prompt: str, evaluation_statistics: Dict[str, Any], 
                failure_patterns: List[Dict[str, Any]], target_improvement: float = 0.1,
                document_type: str = "general") -> Dict[str, Any]:
        """
        Generate optimized prompt using AI.
        
        Args:
            current_prompt: Current prompt being used
            evaluation_statistics: Evaluation statistics and metrics
            failure_patterns: Identified failure patterns
            target_improvement: Target improvement percentage
            document_type: Type of document the prompt is used for
            
        Returns:
            Dict containing optimization results
        """
        
        try:
            result = self.optimizer(
                current_prompt=current_prompt,
                evaluation_statistics=json.dumps(evaluation_statistics),
                failure_patterns=json.dumps(failure_patterns),
                target_improvement=target_improvement,
                document_type=document_type
            )
            
            return {
                "optimized_prompt": result.optimized_prompt,
                "improvement_rationale": result.improvement_rationale,
                "expected_improvement": float(result.expected_improvement),
                "confidence_in_improvement": float(result.confidence_in_improvement),
                "implementation_notes": result.implementation_notes,
                "method": "ai_powered"
            }
            
        except Exception as e:
            return {
                "optimized_prompt": current_prompt,
                "improvement_rationale": f"Optimization failed: {str(e)}",
                "expected_improvement": 0.0,
                "confidence_in_improvement": 0.0,
                "implementation_notes": "No changes made due to optimization failure",
                "method": "ai_fallback"
            }


class DSPyErrorCategorizer(dspy.Module):
    """
    AI-powered error categorizer using DSPy.
    
    This module provides intelligent categorization of errors
    to help understand and address specific issues.
    """
    
    def __init__(self):
        super().__init__()
        self.categorizer = dspy.ChainOfThought(ErrorCategorizationSignature)
    
    def forward(self, error_message: str, field_name: str, field_type: str, 
                document_type: str, context: str = "") -> Dict[str, Any]:
        """
        Categorize error using AI.
        
        Args:
            error_message: Error message or description
            field_name: Name of the field where error occurred
            field_type: Type of field
            document_type: Type of document being processed
            context: Additional context about the error
            
        Returns:
            Dict containing error categorization results
        """
        
        try:
            result = self.categorizer(
                error_message=error_message,
                field_name=field_name,
                field_type=field_type,
                document_type=document_type,
                context=context
            )
            
            return {
                "error_category": result.error_category,
                "error_subcategory": result.error_subcategory,
                "severity_level": result.severity_level,
                "suggested_action": result.suggested_action,
                "prevention_strategy": result.prevention_strategy,
                "method": "ai_powered"
            }
            
        except Exception as e:
            return {
                "error_category": "unknown",
                "error_subcategory": "unknown",
                "severity_level": "medium",
                "suggested_action": "Review error manually",
                "prevention_strategy": "Improve error handling",
                "method": "ai_fallback",
                "error": str(e)
            }


class DSPyPerformanceAnalyzer(dspy.Module):
    """
    AI-powered performance trend analyzer using DSPy.
    
    This module identifies trends in extraction performance
    and provides insights for continuous improvement.
    """
    
    def __init__(self):
        super().__init__()
        self.trend_analyzer = dspy.ChainOfThought(PerformanceTrendAnalysisSignature)
    
    def forward(self, historical_data: List[Dict[str, Any]], time_period: str,
                document_types: List[str]) -> Dict[str, Any]:
        """
        Analyze performance trends using AI.
        
        Args:
            historical_data: Historical evaluation data over time
            time_period: Time period covered by the data
            document_types: Document types included in the analysis
            
        Returns:
            Dict containing trend analysis results
        """
        
        try:
            result = self.trend_analyzer(
                historical_data=json.dumps(historical_data),
                time_period=time_period,
                document_types=json.dumps(document_types)
            )
            
            return {
                "performance_trend": result.performance_trend,
                "trend_strength": result.trend_strength,
                "key_insights": result.key_insights,
                "contributing_factors": result.contributing_factors,
                "recommendations": result.recommendations,
                "method": "ai_powered"
            }
            
        except Exception as e:
            return {
                "performance_trend": "unable_to_assess",
                "trend_strength": "unknown",
                "key_insights": [],
                "contributing_factors": [],
                "recommendations": [],
                "method": "ai_fallback",
                "error": str(e)
            }


class DSPyQualityAssessor(dspy.Module):
    """
    AI-powered quality assessor using DSPy.
    
    This module provides a holistic quality assessment
    considering multiple factors and dimensions.
    """
    
    def __init__(self):
        super().__init__()
        self.quality_assessor = dspy.ChainOfThought(QualityAssessmentSignature)
    
    def forward(self, field_evaluations: List[Dict[str, Any]], 
                document_metadata: Dict[str, Any], 
                extraction_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess quality using AI.
        
        Args:
            field_evaluations: Field evaluation results
            document_metadata: Document metadata
            extraction_metadata: Extraction process metadata
            
        Returns:
            Dict containing quality assessment results
        """
        
        try:
            result = self.quality_assessor(
                field_evaluations=json.dumps(field_evaluations),
                document_metadata=json.dumps(document_metadata),
                extraction_metadata=json.dumps(extraction_metadata)
            )
            
            return {
                "overall_quality_score": float(result.overall_quality_score),
                "quality_dimensions": result.quality_dimensions,
                "quality_grade": result.quality_grade,
                "quality_issues": result.quality_issues,
                "improvement_priorities": result.improvement_priorities,
                "method": "ai_powered"
            }
            
        except Exception as e:
            return {
                "overall_quality_score": 0.0,
                "quality_dimensions": {},
                "quality_grade": "F",
                "quality_issues": [],
                "improvement_priorities": [],
                "method": "ai_fallback",
                "error": str(e)
            }


class DSPyContextAwareEvaluator(dspy.Module):
    """
    AI-powered context-aware evaluator using DSPy.
    
    This module evaluates extractions considering the broader
    context of the document and field relationships.
    """
    
    def __init__(self):
        super().__init__()
        self.context_evaluator = dspy.ChainOfThought(ContextAwareEvaluationSignature)
    
    def forward(self, field_name: str, field_value: str, document_context: Dict[str, Any],
                related_fields: Dict[str, Any], business_rules: List[str]) -> Dict[str, Any]:
        """
        Evaluate field in context using AI.
        
        Args:
            field_name: Name of the field being evaluated
            field_value: Extracted value for the field
            document_context: Context information about the document
            related_fields: Values of related fields in the document
            business_rules: Business rules or constraints that apply
            
        Returns:
            Dict containing context-aware evaluation results
        """
        
        try:
            result = self.context_evaluator(
                field_name=field_name,
                field_value=str(field_value),
                document_context=json.dumps(document_context),
                related_fields=json.dumps(related_fields),
                business_rules=json.dumps(business_rules)
            )
            
            return {
                "context_score": float(result.context_score),
                "context_validation": result.context_validation,
                "consistency_check": result.consistency_check,
                "business_rule_compliance": result.business_rule_compliance,
                "context_notes": result.context_notes,
                "method": "ai_powered"
            }
            
        except Exception as e:
            return {
                "context_score": 0.0,
                "context_validation": "unable_to_validate",
                "consistency_check": "unable_to_check",
                "business_rule_compliance": "unable_to_assess",
                "context_notes": f"Context evaluation failed: {str(e)}",
                "method": "ai_fallback"
            }


class DSPyMultiLanguageEvaluator(dspy.Module):
    """
    AI-powered multi-language evaluator using DSPy.
    
    This module handles evaluation of extractions from
    documents in different languages.
    """
    
    def __init__(self):
        super().__init__()
        self.language_evaluator = dspy.ChainOfThought(MultiLanguageEvaluationSignature)
    
    def forward(self, field_name: str, expected_value: str, extracted_value: str,
                language: str, field_type: str) -> Dict[str, Any]:
        """
        Evaluate field in specific language using AI.
        
        Args:
            field_name: Name of the field being evaluated
            expected_value: Ground truth value
            extracted_value: Extracted value
            language: Language of the document
            field_type: Type of field
            
        Returns:
            Dict containing language-aware evaluation results
        """
        
        try:
            result = self.language_evaluator(
                field_name=field_name,
                expected_value=str(expected_value),
                extracted_value=str(extracted_value),
                language=language,
                field_type=field_type
            )
            
            return {
                "language_aware_score": float(result.language_aware_score),
                "language_specific_issues": result.language_specific_issues,
                "cultural_considerations": result.cultural_considerations,
                "translation_quality": result.translation_quality,
                "language_notes": result.language_notes,
                "method": "ai_powered"
            }
            
        except Exception as e:
            return {
                "language_aware_score": 0.0,
                "language_specific_issues": [],
                "cultural_considerations": [],
                "translation_quality": "unable_to_assess",
                "language_notes": f"Language evaluation failed: {str(e)}",
                "method": "ai_fallback"
            }


# Registry of all modules for easy access
MODULE_REGISTRY = {
    "field_evaluator": DSPyFieldEvaluator,
    "document_aggregator": DSPyDocumentAggregator,
    "failure_analyzer": DSPyFailureAnalyzer,
    "confidence_calibrator": DSPyConfidenceCalibrator,
    "prompt_optimizer": DSPyPromptOptimizer,
    "error_categorizer": DSPyErrorCategorizer,
    "performance_analyzer": DSPyPerformanceAnalyzer,
    "quality_assessor": DSPyQualityAssessor,
    "context_aware_evaluator": DSPyContextAwareEvaluator,
    "multi_language_evaluator": DSPyMultiLanguageEvaluator,
}


def get_module(module_name: str):
    """Get a module by name."""
    return MODULE_REGISTRY.get(module_name)


def list_available_modules():
    """List all available modules."""
    return list(MODULE_REGISTRY.keys())


def create_module_instance(module_name: str):
    """Create an instance of a module by name."""
    module_class = get_module(module_name)
    if module_class:
        return module_class()
    return None 