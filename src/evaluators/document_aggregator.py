"""
Document-level aggregation logic for evaluation results.

This module aggregates field-level evaluation results to produce document-level
metrics, performance analysis, and trend identification.
"""

import statistics
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..models.evaluation_models import (
    FieldEvaluationResult,
    DocumentEvaluationResult,
    ExtractionStatus
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DocumentAggregator:
    """
    Aggregates field-level evaluation results into document-level metrics.
    
    This class combines individual field evaluation results to produce
    overall document accuracy, performance trends, and quality assessments.
    """
    
    def __init__(self):
        """Initialize the document aggregator."""
        logger.info("DocumentAggregator initialized")
    
    def aggregate_evaluations(self,
                             field_evaluations: List[FieldEvaluationResult],
                             document_id: str,
                             document_type: str,
                             confidence_scores: Optional[Dict[str, float]] = None,
                             prompt_version: Optional[str] = None) -> DocumentEvaluationResult:
        """
        Aggregate field evaluations into document-level result.
        
        Args:
            field_evaluations: List of field evaluation results
            document_id: Document identifier
            document_type: Type of document
            confidence_scores: Confidence scores for all fields
            prompt_version: Version of prompt used
            
        Returns:
            DocumentEvaluationResult: Aggregated document evaluation result
        """
        
        if not field_evaluations:
            return self._create_empty_result(document_id, document_type, prompt_version)
        
        # Calculate overall accuracy
        overall_accuracy = self._calculate_overall_accuracy(field_evaluations)
        
        # Calculate confidence correlation
        confidence_correlation = self._calculate_confidence_correlation(
            field_evaluations, confidence_scores
        )
        
        # Create evaluation metadata
        evaluation_metadata = self._create_evaluation_metadata(field_evaluations)
        
        return DocumentEvaluationResult(
            document_id=document_id,
            document_type=document_type,
            field_evaluations=field_evaluations,
            overall_accuracy=overall_accuracy,
            confidence_correlation=confidence_correlation,
            prompt_version=prompt_version,
            evaluation_metadata=evaluation_metadata
        )
    
    def _calculate_overall_accuracy(self, field_evaluations: List[FieldEvaluationResult]) -> float:
        """
        Calculate overall accuracy from field evaluations.
        
        Args:
            field_evaluations: List of field evaluation results
            
        Returns:
            float: Overall accuracy score (0.0-1.0)
        """
        
        if not field_evaluations:
            return 0.0
        
        # Calculate weighted average based on field importance
        total_weight = 0.0
        weighted_sum = 0.0
        
        for field_eval in field_evaluations:
            # Determine field weight (could be configurable)
            weight = self._get_field_weight(field_eval.field_name, field_eval.field_type)
            
            weighted_sum += field_eval.evaluation_score * weight
            total_weight += weight
        
        if total_weight == 0.0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _get_field_weight(self, field_name: str, field_type: str) -> float:
        """
        Get weight for a field based on its importance.
        
        Args:
            field_name: Name of the field
            field_type: Type of the field
            
        Returns:
            float: Weight for the field
        """
        
        # Define field importance weights
        critical_fields = {
            "invoice_number", "total_amount", "vendor_name", "invoice_date",
            "merchant_name", "transaction_amount", "transaction_date"
        }
        
        important_fields = {
            "due_date", "customer_name", "tax_amount", "subtotal",
            "payment_method", "receipt_number"
        }
        
        if field_name.lower() in critical_fields:
            return 2.0  # Critical fields get double weight
        elif field_name.lower() in important_fields:
            return 1.5  # Important fields get 1.5x weight
        else:
            return 1.0  # Standard weight
    
    def _calculate_confidence_correlation(self,
                                        field_evaluations: List[FieldEvaluationResult],
                                        confidence_scores: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate correlation between confidence scores and actual accuracy.
        
        Args:
            field_evaluations: List of field evaluation results
            confidence_scores: Confidence scores for fields
            
        Returns:
            float: Correlation coefficient (0.0-1.0)
        """
        
        if not confidence_scores or len(field_evaluations) < 2:
            return 0.0
        
        # Extract confidence and accuracy pairs
        confidence_values = []
        accuracy_values = []
        
        for field_eval in field_evaluations:
            if field_eval.field_name in confidence_scores:
                confidence_values.append(confidence_scores[field_eval.field_name])
                accuracy_values.append(field_eval.evaluation_score)
        
        if len(confidence_values) < 2:
            return 0.0
        
        # Calculate Pearson correlation
        try:
            correlation = statistics.correlation(confidence_values, accuracy_values)
            return max(0.0, correlation)  # Ensure non-negative
        except (ValueError, statistics.StatisticsError):
            return 0.0
    
    def _create_evaluation_metadata(self, field_evaluations: List[FieldEvaluationResult]) -> Dict[str, Any]:
        """
        Create comprehensive evaluation metadata.
        
        Args:
            field_evaluations: List of field evaluation results
            
        Returns:
            Dict[str, Any]: Evaluation metadata
        """
        
        # Count statuses
        status_counts = {
            ExtractionStatus.SUCCESS: 0,
            ExtractionStatus.PARTIAL: 0,
            ExtractionStatus.FAILED: 0,
            ExtractionStatus.MISSING: 0
        }
        
        for field_eval in field_evaluations:
            status_counts[field_eval.status] += 1
        
        # Calculate statistics
        evaluation_scores = [f.evaluation_score for f in field_evaluations]
        confidence_scores = [f.confidence_score for f in field_evaluations]
        
        metadata = {
            "total_fields": len(field_evaluations),
            "successful_fields": status_counts[ExtractionStatus.SUCCESS],
            "partial_fields": status_counts[ExtractionStatus.PARTIAL],
            "failed_fields": status_counts[ExtractionStatus.FAILED],
            "missing_fields": status_counts[ExtractionStatus.MISSING],
            "success_rate": status_counts[ExtractionStatus.SUCCESS] / len(field_evaluations),
            "failure_rate": (status_counts[ExtractionStatus.FAILED] + 
                           status_counts[ExtractionStatus.MISSING]) / len(field_evaluations),
            "average_evaluation_score": statistics.mean(evaluation_scores) if evaluation_scores else 0.0,
            "average_confidence_score": statistics.mean(confidence_scores) if confidence_scores else 0.0,
            "evaluation_score_std": statistics.stdev(evaluation_scores) if len(evaluation_scores) > 1 else 0.0,
            "confidence_score_std": statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0.0,
            "field_types": list(set(f.field_type for f in field_evaluations)),
            "field_names": [f.field_name for f in field_evaluations],
            "processing_timestamp": datetime.now().isoformat()
        }
        
        return metadata
    
    def _create_empty_result(self,
                           document_id: str,
                           document_type: str,
                           prompt_version: Optional[str] = None) -> DocumentEvaluationResult:
        """Create an empty evaluation result for documents with no fields."""
        
        return DocumentEvaluationResult(
            document_id=document_id,
            document_type=document_type,
            field_evaluations=[],
            overall_accuracy=0.0,
            confidence_correlation=0.0,
            prompt_version=prompt_version,
            evaluation_metadata={
                "total_fields": 0,
                "successful_fields": 0,
                "partial_fields": 0,
                "failed_fields": 0,
                "missing_fields": 0,
                "success_rate": 0.0,
                "failure_rate": 0.0,
                "average_evaluation_score": 0.0,
                "average_confidence_score": 0.0,
                "evaluation_score_std": 0.0,
                "confidence_score_std": 0.0,
                "field_types": [],
                "field_names": [],
                "processing_timestamp": datetime.now().isoformat(),
                "note": "No fields to evaluate"
            }
        )
    
    def analyze_performance_trends(self,
                                 evaluation_results: List[DocumentEvaluationResult]) -> Dict[str, Any]:
        """
        Analyze performance trends across multiple documents.
        
        Args:
            evaluation_results: List of document evaluation results
            
        Returns:
            Dict[str, Any]: Performance trend analysis
        """
        
        if not evaluation_results:
            return {"error": "No evaluation results to analyze"}
        
        # Extract metrics over time
        accuracies = [r.overall_accuracy for r in evaluation_results]
        confidence_correlations = [r.confidence_correlation for r in evaluation_results]
        
        # Calculate trends
        trend_analysis = {
            "total_documents": len(evaluation_results),
            "average_accuracy": statistics.mean(accuracies),
            "accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            "average_confidence_correlation": statistics.mean(confidence_correlations),
            "confidence_correlation_std": statistics.stdev(confidence_correlations) if len(confidence_correlations) > 1 else 0.0,
            "accuracy_trend": self._calculate_trend(accuracies),
            "confidence_trend": self._calculate_trend(confidence_correlations),
            "document_types": list(set(r.document_type for r in evaluation_results)),
            "prompt_versions": list(set(r.prompt_version for r in evaluation_results if r.prompt_version))
        }
        
        return trend_analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend direction for a list of values.
        
        Args:
            values: List of numeric values
            
        Returns:
            str: Trend direction ("improving", "declining", "stable")
        """
        
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        if not first_half or not second_half:
            return "insufficient_data"
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if second_avg > first_avg + 0.05:  # 5% improvement threshold
            return "improving"
        elif second_avg < first_avg - 0.05:  # 5% decline threshold
            return "declining"
        else:
            return "stable"
    
    def generate_quality_report(self,
                              evaluation_result: DocumentEvaluationResult) -> Dict[str, Any]:
        """
        Generate a comprehensive quality report for a document.
        
        Args:
            evaluation_result: Document evaluation result
            
        Returns:
            Dict[str, Any]: Quality report
        """
        
        metadata = evaluation_result.evaluation_metadata
        
        # Quality assessment
        quality_score = self._calculate_quality_score(evaluation_result)
        quality_level = self._determine_quality_level(quality_score)
        
        # Field-level analysis
        field_analysis = self._analyze_field_performance(evaluation_result.field_evaluations)
        
        # Recommendations
        recommendations = self._generate_recommendations(evaluation_result)
        
        report = {
            "document_id": evaluation_result.document_id,
            "document_type": evaluation_result.document_type,
            "overall_accuracy": evaluation_result.overall_accuracy,
            "confidence_correlation": evaluation_result.confidence_correlation,
            "quality_score": quality_score,
            "quality_level": quality_level,
            "field_analysis": field_analysis,
            "recommendations": recommendations,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def _calculate_quality_score(self, evaluation_result: DocumentEvaluationResult) -> float:
        """
        Calculate overall quality score for a document.
        
        Args:
            evaluation_result: Document evaluation result
            
        Returns:
            float: Quality score (0.0-1.0)
        """
        
        # Weighted combination of accuracy and confidence correlation
        accuracy_weight = 0.7
        correlation_weight = 0.3
        
        quality_score = (
            evaluation_result.overall_accuracy * accuracy_weight +
            evaluation_result.confidence_correlation * correlation_weight
        )
        
        return min(1.0, max(0.0, quality_score))
    
    def _determine_quality_level(self, quality_score: float) -> str:
        """
        Determine quality level based on score.
        
        Args:
            quality_score: Quality score (0.0-1.0)
            
        Returns:
            str: Quality level
        """
        
        if quality_score >= 0.9:
            return "excellent"
        elif quality_score >= 0.8:
            return "good"
        elif quality_score >= 0.7:
            return "acceptable"
        elif quality_score >= 0.6:
            return "needs_improvement"
        else:
            return "poor"
    
    def _analyze_field_performance(self,
                                 field_evaluations: List[FieldEvaluationResult]) -> Dict[str, Any]:
        """
        Analyze performance of individual fields.
        
        Args:
            field_evaluations: List of field evaluation results
            
        Returns:
            Dict[str, Any]: Field performance analysis
        """
        
        field_performance = {}
        
        for field_eval in field_evaluations:
            field_name = field_eval.field_name
            
            if field_name not in field_performance:
                field_performance[field_name] = {
                    "evaluation_score": field_eval.evaluation_score,
                    "confidence_score": field_eval.confidence_score,
                    "status": field_eval.status.value,
                    "field_type": field_eval.field_type,
                    "error_message": field_eval.error_message,
                    "performance_level": self._get_performance_level(field_eval.evaluation_score)
                }
        
        return field_performance
    
    def _get_performance_level(self, evaluation_score: float) -> str:
        """
        Get performance level for a field.
        
        Args:
            evaluation_score: Evaluation score (0.0-1.0)
            
        Returns:
            str: Performance level
        """
        
        if evaluation_score >= 0.95:
            return "excellent"
        elif evaluation_score >= 0.85:
            return "good"
        elif evaluation_score >= 0.75:
            return "acceptable"
        elif evaluation_score >= 0.5:
            return "needs_improvement"
        else:
            return "poor"
    
    def _generate_recommendations(self,
                                evaluation_result: DocumentEvaluationResult) -> List[str]:
        """
        Generate recommendations for improvement.
        
        Args:
            evaluation_result: Document evaluation result
            
        Returns:
            List[str]: List of recommendations
        """
        
        recommendations = []
        
        # Overall accuracy recommendations
        if evaluation_result.overall_accuracy < 0.8:
            recommendations.append("Overall accuracy is below target. Consider prompt optimization.")
        
        # Confidence correlation recommendations
        if evaluation_result.confidence_correlation < 0.7:
            recommendations.append("Low confidence correlation suggests model uncertainty. Review confidence calibration.")
        
        # Field-specific recommendations
        failed_fields = [f for f in evaluation_result.field_evaluations if f.is_failed()]
        if failed_fields:
            field_names = [f.field_name for f in failed_fields]
            recommendations.append(f"Failed fields detected: {', '.join(field_names)}. Focus optimization on these fields.")
        
        # Missing fields recommendations
        missing_fields = [f for f in evaluation_result.field_evaluations if f.status == ExtractionStatus.MISSING]
        if missing_fields:
            field_names = [f.field_name for f in missing_fields]
            recommendations.append(f"Missing fields: {', '.join(field_names)}. Check field extraction logic.")
        
        return recommendations 