"""
Statistics collection and analysis engine for evaluation results.

This module provides comprehensive statistics collection, trend analysis,
and performance monitoring for document extraction evaluations.
"""

import statistics
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json

from ..models.evaluation_models import (
    DocumentEvaluationResult,
    FieldEvaluationResult,
    EvaluationStatistics,
    ExtractionStatus
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class StatisticsEngine:
    """
    Comprehensive statistics collection and analysis engine.
    
    This class tracks performance metrics, analyzes trends, and provides
    insights for optimization and monitoring.
    """
    
    def __init__(self):
        """Initialize the statistics engine."""
        self.statistics = EvaluationStatistics()
        self.historical_data = []
        logger.info("StatisticsEngine initialized")
    
    def update_statistics(self, evaluation_result: DocumentEvaluationResult) -> None:
        """
        Update statistics with new evaluation result.
        
        Args:
            evaluation_result: New evaluation result to incorporate
        """
        
        # Update basic counts
        self.statistics.total_documents += 1
        
        # Process field evaluations
        for field_eval in evaluation_result.field_evaluations:
            self.statistics.total_fields += 1
            
            # Update status counts
            if field_eval.is_successful():
                self.statistics.successful_extractions += 1
            elif field_eval.is_failed():
                self.statistics.failed_extractions += 1
            elif field_eval.is_partial():
                self.statistics.partial_extractions += 1
            
            # Update field success rates
            field_name = field_eval.field_name
            success_score = 1.0 if field_eval.is_successful() else 0.0
            self.statistics.field_success_rates.setdefault(field_name, []).append(success_score)
            
            # Track common errors
            if field_eval.error_message:
                self.statistics.common_errors[field_eval.error_message] = \
                    self.statistics.common_errors.get(field_eval.error_message, 0) + 1
            
            # Categorize confidence levels
            if field_eval.confidence_score > 0.8:
                self.statistics.confidence_distribution["high"] = \
                    self.statistics.confidence_distribution.get("high", 0) + 1
            elif field_eval.confidence_score > 0.5:
                self.statistics.confidence_distribution["medium"] = \
                    self.statistics.confidence_distribution.get("medium", 0) + 1
            else:
                self.statistics.confidence_distribution["low"] = \
                    self.statistics.confidence_distribution.get("low", 0) + 1
        
        # Update document-level statistics
        self._update_document_level_statistics(evaluation_result)
        
        # Store historical data
        self._store_historical_data(evaluation_result)
        
        logger.debug(f"Updated statistics for document {evaluation_result.document_id}")
    
    def _update_document_level_statistics(self, evaluation_result: DocumentEvaluationResult) -> None:
        """Update document-level statistics."""
        
        # Update prompt version performance
        if evaluation_result.prompt_version:
            current_performance = self.statistics.prompt_version_performance.get(
                evaluation_result.prompt_version, []
            )
            current_performance.append(evaluation_result.overall_accuracy)
            self.statistics.prompt_version_performance[evaluation_result.prompt_version] = current_performance
        
        # Update document type performance
        current_performance = self.statistics.document_type_performance.get(
            evaluation_result.document_type, []
        )
        current_performance.append(evaluation_result.overall_accuracy)
        self.statistics.document_type_performance[evaluation_result.document_type] = current_performance
        
        # Update average accuracy
        all_accuracies = []
        for accuracies in self.statistics.document_type_performance.values():
            all_accuracies.extend(accuracies)
        
        if all_accuracies:
            self.statistics.average_accuracy = statistics.mean(all_accuracies)
        
        # Update average confidence
        confidence_scores = []
        for field_eval in evaluation_result.field_evaluations:
            confidence_scores.append(field_eval.confidence_score)
        
        if confidence_scores:
            self.statistics.average_confidence = statistics.mean(confidence_scores)
    
    def _store_historical_data(self, evaluation_result: DocumentEvaluationResult) -> None:
        """Store evaluation result in historical data."""
        
        historical_entry = {
            "document_id": evaluation_result.document_id,
            "document_type": evaluation_result.document_type,
            "overall_accuracy": evaluation_result.overall_accuracy,
            "confidence_correlation": evaluation_result.confidence_correlation,
            "prompt_version": evaluation_result.prompt_version,
            "timestamp": evaluation_result.evaluation_timestamp.isoformat(),
            "field_count": len(evaluation_result.field_evaluations),
            "successful_fields": len([f for f in evaluation_result.field_evaluations if f.is_successful()]),
            "failed_fields": len([f for f in evaluation_result.field_evaluations if f.is_failed()])
        }
        
        self.historical_data.append(historical_entry)
        self.statistics.evaluation_history.append(historical_entry)
    
    def get_performance_metrics(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.
        
        Args:
            time_window: Optional time window for filtering data
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        
        # Filter data by time window if specified
        if time_window:
            cutoff_time = datetime.now() - time_window
            filtered_data = [
                entry for entry in self.historical_data
                if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
            ]
        else:
            filtered_data = self.historical_data
        
        if not filtered_data:
            return {"message": "No data available for the specified time window"}
        
        # Calculate metrics
        accuracies = [entry["overall_accuracy"] for entry in filtered_data]
        confidence_correlations = [entry["confidence_correlation"] for entry in filtered_data]
        
        metrics = {
            "total_documents": len(filtered_data),
            "average_accuracy": statistics.mean(accuracies),
            "accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "average_confidence_correlation": statistics.mean(confidence_correlations),
            "confidence_correlation_std": statistics.stdev(confidence_correlations) if len(confidence_correlations) > 1 else 0.0,
            "success_rate": self.statistics.successful_extractions / self.statistics.total_fields if self.statistics.total_fields > 0 else 0.0,
            "failure_rate": self.statistics.failed_extractions / self.statistics.total_fields if self.statistics.total_fields > 0 else 0.0,
            "partial_rate": self.statistics.partial_extractions / self.statistics.total_fields if self.statistics.total_fields > 0 else 0.0
        }
        
        return metrics
    
    def get_field_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics for individual fields.
        
        Returns:
            Dict[str, Dict[str, Any]]: Field performance metrics
        """
        
        field_performance = {}
        
        for field_name, success_scores in self.statistics.field_success_rates.items():
            if success_scores:
                field_performance[field_name] = {
                    "success_rate": statistics.mean(success_scores),
                    "consistency": 1.0 - statistics.stdev(success_scores) if len(success_scores) > 1 else 1.0,
                    "total_evaluations": len(success_scores),
                    "recent_trend": self._calculate_field_trend(field_name)
                }
        
        return field_performance
    
    def _calculate_field_trend(self, field_name: str) -> str:
        """Calculate trend for a specific field."""
        
        success_scores = self.statistics.field_success_rates.get(field_name, [])
        if len(success_scores) < 4:
            return "insufficient_data"
        
        # Split into two halves
        first_half = success_scores[:len(success_scores)//2]
        second_half = success_scores[len(success_scores)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if second_avg > first_avg + 0.1:
            return "improving"
        elif second_avg < first_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    def get_document_type_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics by document type.
        
        Returns:
            Dict[str, Dict[str, Any]]: Document type performance metrics
        """
        
        doc_type_performance = {}
        
        for doc_type, accuracies in self.statistics.document_type_performance.items():
            if accuracies:
                doc_type_performance[doc_type] = {
                    "average_accuracy": statistics.mean(accuracies),
                    "accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                    "total_documents": len(accuracies),
                    "min_accuracy": min(accuracies),
                    "max_accuracy": max(accuracies),
                    "trend": self._calculate_doc_type_trend(doc_type)
                }
        
        return doc_type_performance
    
    def _calculate_doc_type_trend(self, doc_type: str) -> str:
        """Calculate trend for a document type."""
        
        accuracies = self.statistics.document_type_performance.get(doc_type, [])
        if len(accuracies) < 4:
            return "insufficient_data"
        
        # Split into two halves
        first_half = accuracies[:len(accuracies)//2]
        second_half = accuracies[len(accuracies)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if second_avg > first_avg + 0.05:
            return "improving"
        elif second_avg < first_avg - 0.05:
            return "declining"
        else:
            return "stable"
    
    def get_prompt_version_performance(self) -> Dict[str, Dict[str, Any]]:
        """
        Get performance metrics by prompt version.
        
        Returns:
            Dict[str, Dict[str, Any]]: Prompt version performance metrics
        """
        
        prompt_performance = {}
        
        for prompt_version, accuracies in self.statistics.prompt_version_performance.items():
            if accuracies:
                prompt_performance[prompt_version] = {
                    "average_accuracy": statistics.mean(accuracies),
                    "accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                    "total_documents": len(accuracies),
                    "min_accuracy": min(accuracies),
                    "max_accuracy": max(accuracies),
                    "improvement": self._calculate_prompt_improvement(prompt_version)
                }
        
        return prompt_performance
    
    def _calculate_prompt_improvement(self, prompt_version: str) -> Optional[float]:
        """Calculate improvement compared to previous prompt version."""
        
        # This would need to be implemented based on prompt version ordering
        # For now, return None
        return None
    
    def get_error_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive error analysis.
        
        Returns:
            Dict[str, Any]: Error analysis
        """
        
        # Most common errors
        top_errors = sorted(
            self.statistics.common_errors.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Error categories
        error_categories = defaultdict(int)
        for error_msg, count in self.statistics.common_errors.items():
            category = self._categorize_error(error_msg)
            error_categories[category] += count
        
        # Confidence distribution analysis
        confidence_analysis = {
            "high_confidence_rate": self.statistics.confidence_distribution.get("high", 0) / self.statistics.total_fields if self.statistics.total_fields > 0 else 0.0,
            "medium_confidence_rate": self.statistics.confidence_distribution.get("medium", 0) / self.statistics.total_fields if self.statistics.total_fields > 0 else 0.0,
            "low_confidence_rate": self.statistics.confidence_distribution.get("low", 0) / self.statistics.total_fields if self.statistics.total_fields > 0 else 0.0
        }
        
        return {
            "top_errors": top_errors,
            "error_categories": dict(error_categories),
            "confidence_analysis": confidence_analysis,
            "total_errors": sum(self.statistics.common_errors.values())
        }
    
    def _categorize_error(self, error_message: str) -> str:
        """Categorize error message."""
        
        error_lower = error_message.lower()
        
        if any(word in error_lower for word in ["date", "format", "parse"]):
            return "date_format"
        elif any(word in error_lower for word in ["number", "currency", "amount"]):
            return "number_format"
        elif any(word in error_lower for word in ["missing", "empty", "not found"]):
            return "field_missing"
        elif any(word in error_lower for word in ["partial", "incomplete"]):
            return "partial_extraction"
        elif any(word in error_lower for word in ["confidence", "uncertain"]):
            return "confidence_issue"
        else:
            return "other"
    
    def get_trend_analysis(self, days: int = 30) -> Dict[str, Any]:
        """
        Get trend analysis for the specified number of days.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict[str, Any]: Trend analysis
        """
        
        cutoff_time = datetime.now() - timedelta(days=days)
        recent_data = [
            entry for entry in self.historical_data
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]
        
        if len(recent_data) < 2:
            return {"message": "Insufficient data for trend analysis"}
        
        # Group by day
        daily_data = defaultdict(list)
        for entry in recent_data:
            date = datetime.fromisoformat(entry["timestamp"]).date()
            daily_data[date].append(entry)
        
        # Calculate daily averages
        daily_averages = []
        for date, entries in sorted(daily_data.items()):
            avg_accuracy = statistics.mean(entry["overall_accuracy"] for entry in entries)
            daily_averages.append({
                "date": date.isoformat(),
                "average_accuracy": avg_accuracy,
                "document_count": len(entries)
            })
        
        # Calculate trend
        if len(daily_averages) >= 2:
            accuracies = [day["average_accuracy"] for day in daily_averages]
            trend = self._calculate_linear_trend(accuracies)
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "daily_averages": daily_averages,
            "total_days": len(daily_averages),
            "overall_trend_accuracy": statistics.mean(day["average_accuracy"] for day in daily_averages)
        }
    
    def _calculate_linear_trend(self, values: List[float]) -> str:
        """Calculate linear trend for a list of values."""
        
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if second_avg > first_avg + 0.05:
            return "improving"
        elif second_avg < first_avg - 0.05:
            return "declining"
        else:
            return "stable"
    
    def get_optimization_metrics(self) -> Dict[str, float]:
        """
        Get metrics specifically for DSPy optimization.
        
        Returns:
            Dict[str, float]: Optimization metrics
        """
        
        metrics = {}
        
        # Field-specific metrics
        for field_name, scores in self.statistics.field_success_rates.items():
            if scores:
                metrics[f"{field_name}_success_rate"] = statistics.mean(scores)
                metrics[f"{field_name}_consistency"] = 1.0 - statistics.stdev(scores) if len(scores) > 1 else 1.0
        
        # Overall metrics
        if self.statistics.total_documents > 0:
            metrics["overall_success_rate"] = self.statistics.successful_extractions / self.statistics.total_fields
            metrics["error_rate"] = self.statistics.failed_extractions / self.statistics.total_fields
            metrics["average_accuracy"] = self.statistics.average_accuracy
        
        return metrics
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        
        self.statistics = EvaluationStatistics()
        self.historical_data = []
        logger.info("Statistics reset")
    
    def export_statistics(self, filename: str) -> None:
        """
        Export statistics to a JSON file.
        
        Args:
            filename: Output filename
        """
        
        export_data = {
            "statistics": self.statistics.dict(),
            "historical_data": self.historical_data,
            "export_timestamp": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Statistics exported to {filename}")
    
    def import_statistics(self, filename: str) -> None:
        """
        Import statistics from a JSON file.
        
        Args:
            filename: Input filename
        """
        
        with open(filename, 'r') as f:
            import_data = json.load(f)
        
        self.statistics = EvaluationStatistics(**import_data["statistics"])
        self.historical_data = import_data.get("historical_data", [])
        
        logger.info(f"Statistics imported from {filename}") 