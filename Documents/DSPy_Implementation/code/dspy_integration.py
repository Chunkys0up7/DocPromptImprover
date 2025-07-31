"""
DSPy Integration with Existing Statistical Evaluation Framework

This module demonstrates how to integrate DSPy AI-powered evaluation
with the existing statistical evaluation framework to create a hybrid system.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import existing statistical evaluators
from src.evaluators.field_evaluator import FieldEvaluator
from src.evaluators.document_aggregator import DocumentAggregator
from src.evaluators.error_pattern_detector import ErrorPatternDetector
from src.statistics.statistics_engine import StatisticsEngine

# Import DSPy components
from .dspy_config import initialize_dspy, get_dspy_manager, is_dspy_available
from .dspy_modules import (
    DSPyFieldEvaluator,
    DSPyDocumentAggregator,
    DSPyFailureAnalyzer,
    DSPyConfidenceCalibrator,
    DSPyPromptOptimizer
)


class HybridEvaluator:
    """
    Hybrid evaluator that combines statistical and AI-powered evaluation.
    
    This class provides a seamless integration between the existing statistical
    evaluation framework and DSPy AI-powered capabilities.
    """
    
    def __init__(self, use_ai: bool = True, ai_confidence_threshold: float = 0.8):
        """
        Initialize the hybrid evaluator.
        
        Args:
            use_ai: Whether to use AI-powered evaluation
            ai_confidence_threshold: Confidence threshold for using AI evaluation
        """
        self.use_ai = use_ai and is_dspy_available()
        self.ai_confidence_threshold = ai_confidence_threshold
        
        # Initialize statistical evaluators
        self.statistical_field_evaluator = FieldEvaluator()
        self.statistical_document_aggregator = DocumentAggregator()
        self.statistical_error_detector = ErrorPatternDetector()
        self.statistics_engine = StatisticsEngine()
        
        # Initialize AI-powered evaluators if available
        if self.use_ai:
            try:
                initialize_dspy()
                self.ai_field_evaluator = DSPyFieldEvaluator()
                self.ai_document_aggregator = DSPyDocumentAggregator()
                self.ai_failure_analyzer = DSPyFailureAnalyzer()
                self.ai_confidence_calibrator = DSPyConfidenceCalibrator()
                self.ai_prompt_optimizer = DSPyPromptOptimizer()
                print("✅ AI-powered evaluation enabled")
            except Exception as e:
                print(f"⚠️ AI evaluation disabled: {e}")
                self.use_ai = False
        else:
            print("ℹ️ Using statistical evaluation only")
    
    def evaluate_field(self, field_name: str, expected_value: str, 
                      extracted_value: str, confidence_score: float, 
                      field_type: str) -> Dict[str, Any]:
        """
        Evaluate a field using hybrid approach.
        
        Args:
            field_name: Name of the field
            expected_value: Ground truth value
            extracted_value: Extracted value
            confidence_score: Confidence score from extraction
            field_type: Type of field
            
        Returns:
            Dict containing evaluation results
        """
        
        # Always perform statistical evaluation
        statistical_result = self.statistical_field_evaluator.evaluate_field(
            field_name, expected_value, extracted_value, confidence_score, field_type
        )
        
        # Convert to dict format
        statistical_dict = {
            "evaluation_score": statistical_result.evaluation_score,
            "status": statistical_result.status.value,
            "error_message": statistical_result.error_message,
            "evaluation_notes": statistical_result.evaluation_notes,
            "method": "statistical"
        }
        
        # Use AI evaluation if enabled and confidence is high enough
        if self.use_ai and confidence_score >= self.ai_confidence_threshold:
            try:
                ai_result = self.ai_field_evaluator(
                    field_name, expected_value, extracted_value, confidence_score, field_type
                )
                
                # Combine results (weighted average)
                combined_score = (statistical_result.evaluation_score * 0.6 + 
                                ai_result["evaluation_score"] * 0.4)
                
                return {
                    "evaluation_score": combined_score,
                    "status": ai_result["status"],
                    "error_message": ai_result["error_message"],
                    "evaluation_notes": f"Hybrid: {statistical_result.evaluation_notes}; AI: {ai_result['evaluation_notes']}",
                    "method": "hybrid",
                    "statistical_score": statistical_result.evaluation_score,
                    "ai_score": ai_result["evaluation_score"],
                    "confidence_assessment": ai_result.get("confidence_assessment", "unable_to_assess")
                }
                
            except Exception as e:
                # Fallback to statistical only
                return {
                    **statistical_dict,
                    "method": "statistical_fallback",
                    "ai_error": str(e)
                }
        
        return statistical_dict
    
    def aggregate_document(self, field_evaluations: List[Dict[str, Any]], 
                          document_id: str, document_type: str,
                          confidence_scores: Dict[str, float],
                          prompt_version: str = "v1.0") -> Dict[str, Any]:
        """
        Aggregate document results using hybrid approach.
        
        Args:
            field_evaluations: List of field evaluation results
            document_id: Document ID
            document_type: Type of document
            confidence_scores: Confidence scores for each field
            prompt_version: Version of the prompt used
            
        Returns:
            Dict containing aggregated results
        """
        
        # Always perform statistical aggregation
        statistical_result = self.statistical_document_aggregator.aggregate_evaluations(
            field_evaluations, document_id, document_type, confidence_scores, prompt_version
        )
        
        # Convert to dict format
        statistical_dict = {
            "overall_accuracy": statistical_result.overall_accuracy,
            "confidence_correlation": statistical_result.confidence_correlation,
            "total_fields": statistical_result.total_fields,
            "successful_fields": statistical_result.successful_fields,
            "failed_fields": statistical_result.failed_fields,
            "method": "statistical"
        }
        
        # Use AI aggregation if enabled
        if self.use_ai:
            try:
                ai_result = self.ai_document_aggregator(
                    field_evaluations, document_type, confidence_scores, prompt_version
                )
                
                # Combine results
                combined_accuracy = (statistical_result.overall_accuracy * 0.7 + 
                                   ai_result["overall_accuracy"] * 0.3)
                
                return {
                    "overall_accuracy": combined_accuracy,
                    "confidence_correlation": ai_result["confidence_correlation"],
                    "quality_assessment": ai_result["quality_assessment"],
                    "critical_errors": ai_result["critical_errors"],
                    "improvement_suggestions": ai_result["improvement_suggestions"],
                    "method": "hybrid",
                    "statistical_accuracy": statistical_result.overall_accuracy,
                    "ai_accuracy": ai_result["overall_accuracy"],
                    "total_fields": statistical_result.total_fields,
                    "successful_fields": statistical_result.successful_fields,
                    "failed_fields": statistical_result.failed_fields
                }
                
            except Exception as e:
                return {
                    **statistical_dict,
                    "method": "statistical_fallback",
                    "ai_error": str(e)
                }
        
        return statistical_dict
    
    def detect_error_patterns(self, evaluation_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect error patterns using hybrid approach.
        
        Args:
            evaluation_results: List of evaluation results
            
        Returns:
            List of detected error patterns
        """
        
        # Always perform statistical pattern detection
        statistical_patterns = self.statistical_error_detector.detect_patterns(evaluation_results)
        
        # Use AI pattern detection if enabled
        if self.use_ai:
            try:
                ai_patterns = self.ai_failure_analyzer(
                    evaluation_results,
                    list(set(r.get("document_type", "unknown") for r in evaluation_results)),
                    "recent"
                )
                
                # Combine patterns
                combined_patterns = []
                
                # Add statistical patterns
                for pattern in statistical_patterns:
                    combined_patterns.append({
                        **pattern.dict(),
                        "detection_method": "statistical"
                    })
                
                # Add AI patterns
                for pattern in ai_patterns.get("common_patterns", []):
                    combined_patterns.append({
                        "pattern_type": pattern,
                        "detection_method": "ai",
                        "frequency": ai_patterns.get("pattern_frequency", {}).get(pattern, 0),
                        "severity": ai_patterns.get("pattern_severity", {}).get(pattern, "medium"),
                        "suggested_fixes": ai_patterns.get("suggested_fixes", [])
                    })
                
                return combined_patterns
                
            except Exception as e:
                # Fallback to statistical only
                return [{
                    **pattern.dict(),
                    "detection_method": "statistical_fallback",
                    "ai_error": str(e)
                } for pattern in statistical_patterns]
        
        return [{
            **pattern.dict(),
            "detection_method": "statistical"
        } for pattern in statistical_patterns]
    
    def calibrate_confidence_scores(self, confidence_scores: List[float], 
                                   actual_accuracy: List[float],
                                   field_types: List[str]) -> Dict[str, Any]:
        """
        Calibrate confidence scores using hybrid approach.
        
        Args:
            confidence_scores: List of confidence scores
            actual_accuracy: List of actual accuracy scores
            field_types: List of field types
            
        Returns:
            Dict containing calibration results
        """
        
        # Use AI calibration if available
        if self.use_ai:
            try:
                return self.ai_confidence_calibrator(confidence_scores, actual_accuracy, field_types)
            except Exception as e:
                return {
                    "calibration_factor": 1.0,
                    "calibrated_scores": confidence_scores,
                    "calibration_quality": "unable_to_assess",
                    "calibration_notes": f"AI calibration failed: {str(e)}",
                    "method": "ai_fallback"
                }
        
        # Fallback to basic calibration
        return {
            "calibration_factor": 1.0,
            "calibrated_scores": confidence_scores,
            "calibration_quality": "basic",
            "calibration_notes": "Basic calibration (no AI available)",
            "method": "statistical"
        }
    
    def optimize_prompt(self, current_prompt: str, evaluation_statistics: Dict[str, Any],
                       failure_patterns: List[Dict[str, Any]], target_improvement: float = 0.1,
                       document_type: str = "general") -> Dict[str, Any]:
        """
        Optimize prompt using AI if available.
        
        Args:
            current_prompt: Current prompt being used
            evaluation_statistics: Evaluation statistics
            failure_patterns: Identified failure patterns
            target_improvement: Target improvement percentage
            document_type: Type of document
            
        Returns:
            Dict containing optimization results
        """
        
        if self.use_ai:
            try:
                return self.ai_prompt_optimizer(
                    current_prompt, evaluation_statistics, failure_patterns,
                    target_improvement, document_type
                )
            except Exception as e:
                return {
                    "optimized_prompt": current_prompt,
                    "improvement_rationale": f"AI optimization failed: {str(e)}",
                    "expected_improvement": 0.0,
                    "confidence_in_improvement": 0.0,
                    "implementation_notes": "No changes made due to optimization failure",
                    "method": "ai_fallback"
                }
        
        return {
            "optimized_prompt": current_prompt,
            "improvement_rationale": "No AI optimization available",
            "expected_improvement": 0.0,
            "confidence_in_improvement": 0.0,
            "implementation_notes": "Statistical evaluation only - no AI optimization",
            "method": "statistical"
        }
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get evaluation summary including both statistical and AI metrics."""
        
        summary = {
            "evaluation_method": "hybrid" if self.use_ai else "statistical",
            "ai_enabled": self.use_ai,
            "ai_confidence_threshold": self.ai_confidence_threshold,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add statistical summary
        stats_summary = self.statistics_engine.statistics
        summary["statistical_metrics"] = {
            "total_documents": stats_summary.total_documents,
            "total_fields": stats_summary.total_fields,
            "average_accuracy": stats_summary.average_accuracy,
            "success_rate": stats_summary.success_rate
        }
        
        # Add AI cost summary if available
        if self.use_ai:
            try:
                manager = get_dspy_manager()
                cost_summary = manager.get_cost_summary()
                summary["ai_metrics"] = {
                    "total_cost": cost_summary["total_cost"],
                    "evaluation_count": cost_summary["evaluation_count"],
                    "average_cost_per_evaluation": cost_summary["average_cost_per_evaluation"],
                    "budget_remaining": cost_summary["budget_remaining"]
                }
            except:
                summary["ai_metrics"] = {"error": "Unable to retrieve AI metrics"}
        
        return summary
    
    def reset_statistics(self):
        """Reset both statistical and AI statistics."""
        self.statistics_engine.reset_statistics()
        
        if self.use_ai:
            try:
                manager = get_dspy_manager()
                manager.cost_tracker.reset_costs()
            except:
                pass


class DSPyIntegrationManager:
    """
    Manager for DSPy integration with the existing system.
    
    This class provides high-level management of DSPy integration,
    including configuration, monitoring, and fallback strategies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the integration manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.hybrid_evaluator = None
        self.integration_status = "not_initialized"
    
    def initialize_integration(self, use_ai: bool = True, 
                             ai_confidence_threshold: float = 0.8) -> bool:
        """
        Initialize DSPy integration.
        
        Args:
            use_ai: Whether to enable AI-powered evaluation
            ai_confidence_threshold: Confidence threshold for AI evaluation
            
        Returns:
            True if initialization successful
        """
        
        try:
            self.hybrid_evaluator = HybridEvaluator(use_ai, ai_confidence_threshold)
            self.integration_status = "initialized"
            return True
            
        except Exception as e:
            print(f"Failed to initialize DSPy integration: {e}")
            self.integration_status = "failed"
            return False
    
    def get_evaluator(self) -> Optional[HybridEvaluator]:
        """Get the hybrid evaluator instance."""
        return self.hybrid_evaluator
    
    def is_ai_available(self) -> bool:
        """Check if AI evaluation is available."""
        return (self.hybrid_evaluator is not None and 
                self.hybrid_evaluator.use_ai)
    
    def get_integration_status(self) -> str:
        """Get the current integration status."""
        return self.integration_status
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        if self.hybrid_evaluator is None:
            return {"status": "not_initialized"}
        
        return {
            "status": self.integration_status,
            "ai_enabled": self.hybrid_evaluator.use_ai,
            "ai_confidence_threshold": self.hybrid_evaluator.ai_confidence_threshold,
            "dspy_available": is_dspy_available()
        }


# Global integration manager instance
_integration_manager: Optional[DSPyIntegrationManager] = None


def get_integration_manager() -> DSPyIntegrationManager:
    """Get the global integration manager instance."""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = DSPyIntegrationManager()
    return _integration_manager


def initialize_dspy_integration(use_ai: bool = True, 
                               ai_confidence_threshold: float = 0.8) -> bool:
    """Initialize DSPy integration globally."""
    manager = get_integration_manager()
    return manager.initialize_integration(use_ai, ai_confidence_threshold)


def get_hybrid_evaluator() -> Optional[HybridEvaluator]:
    """Get the global hybrid evaluator instance."""
    manager = get_integration_manager()
    return manager.get_evaluator()


def is_dspy_integration_available() -> bool:
    """Check if DSPy integration is available."""
    manager = get_integration_manager()
    return manager.get_integration_status() == "initialized" 