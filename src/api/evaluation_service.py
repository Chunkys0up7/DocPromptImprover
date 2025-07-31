"""
FastAPI evaluation service for document extraction evaluation.

This module provides the main API endpoints for evaluating the outputs of
existing OCR-plus-prompt pipelines using pure statistical analysis.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from ..models.evaluation_models import (
    DocumentEvaluationInput,
    DocumentEvaluationResult,
    FieldEvaluationResult,
    EvaluationStatistics,
    FailurePattern,
    OptimizationRecommendation,
    EvaluationConfig,
    ExtractionStatus
)
from ..evaluators.field_evaluator import FieldEvaluator
from ..evaluators.document_aggregator import DocumentAggregator
from ..evaluators.error_pattern_detector import ErrorPatternDetector
from ..statistics.statistics_engine import StatisticsEngine
from ..utils.config import get_config
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Extraction Evaluation Service",
    description="Evaluation-only microservice for assessing OCR-plus-prompt pipeline outputs using statistical analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
statistics_engine = StatisticsEngine()
evaluation_config = EvaluationConfig()


class DocumentExtractionEvaluator:
    """
    Main evaluation module for document extraction results.
    
    This module evaluates the outputs of existing OCR-plus-prompt pipelines
    using pure statistical analysis and provides detailed feedback for optimization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the evaluator with configuration."""
        
        self.config = config or get_config()
        
        # Initialize statistical evaluators
        self.field_evaluator = FieldEvaluator()
        self.document_aggregator = DocumentAggregator()
        self.error_detector = ErrorPatternDetector()
        
        logger.info("DocumentExtractionEvaluator initialized with statistical analysis")
    
    def evaluate_field(self, 
                      field_name: str,
                      expected_value: Optional[str],
                      extracted_value: Optional[str],
                      confidence_score: float = 0.0,
                      field_type: str = "text") -> FieldEvaluationResult:
        """
        Evaluate a single field extraction using statistical methods.
        
        Args:
            field_name: Name of the field
            expected_value: Ground truth value
            extracted_value: Value extracted by the model
            confidence_score: Confidence score from the model
            field_type: Expected data type of the field
            
        Returns:
            FieldEvaluationResult: Evaluation result
        """
        
        return self.field_evaluator.evaluate_field(
            field_name=field_name,
            expected_value=expected_value,
            extracted_value=extracted_value,
            confidence_score=confidence_score,
            field_type=field_type
        )
    
    def evaluate_document(self, 
                         evaluation_input: DocumentEvaluationInput) -> DocumentEvaluationResult:
        """
        Evaluate a complete document using statistical analysis.
        
        Args:
            evaluation_input: Input containing extracted fields and ground truth
            
        Returns:
            DocumentEvaluationResult: Complete evaluation result
        """
        
        start_time = time.time()
        
        try:
            # Step 1: Field-level evaluation
            field_evaluations = []
            
            for field_name in set(evaluation_input.extracted_fields.keys()) | set(evaluation_input.ground_truth.keys()):
                expected_value = evaluation_input.ground_truth.get(field_name)
                extracted_value = evaluation_input.extracted_fields.get(field_name)
                confidence_score = evaluation_input.confidence_scores.get(field_name, 0.0)
                
                # Determine field type
                field_type = self._determine_field_type(field_name, expected_value)
                
                # Evaluate field
                field_result = self.evaluate_field(
                    field_name=field_name,
                    expected_value=expected_value,
                    extracted_value=extracted_value,
                    confidence_score=confidence_score,
                    field_type=field_type
                )
                field_evaluations.append(field_result)
            
            # Step 2: Document-level aggregation
            document_result = self.document_aggregator.aggregate_evaluations(
                field_evaluations=field_evaluations,
                document_id=evaluation_input.document_id,
                document_type=evaluation_input.document_type,
                confidence_scores=evaluation_input.confidence_scores,
                prompt_version=evaluation_input.prompt_version
            )
            
            # Step 3: Update statistics
            statistics_engine.update_statistics(document_result)
            
            # Step 4: Calculate processing time
            processing_time = time.time() - start_time
            document_result.processing_time = processing_time
            
            logger.info(f"Document {evaluation_input.document_id} evaluated in {processing_time:.3f}s")
            
            return document_result
            
        except Exception as e:
            logger.error(f"Error evaluating document {evaluation_input.document_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
    
    def _determine_field_type(self, field_name: str, value: Any) -> str:
        """Determine field type based on field name and value."""
        
        if any(date_word in field_name.lower() for date_word in ["date", "birth"]):
            return "date"
        elif any(num_word in field_name.lower() for num_word in ["amount", "total", "tax", "number"]):
            return "number"
        elif "email" in field_name.lower():
            return "email"
        elif "phone" in field_name.lower():
            return "phone"
        else:
            return "text"
    
    def get_optimization_recommendations(self, 
                                       evaluation_results: List[DocumentEvaluationResult]) -> List[OptimizationRecommendation]:
        """
        Generate optimization recommendations using statistical analysis.
        
        Args:
            evaluation_results: List of evaluation results
            
        Returns:
            List[OptimizationRecommendation]: Optimization recommendations
        """
        
        recommendations = []
        
        # Detect error patterns
        error_patterns = self.error_detector.detect_patterns(evaluation_results)
        
        # Generate recommendations based on patterns
        for pattern in error_patterns:
            recommendation = OptimizationRecommendation(
                recommendation_id=f"rec_{pattern.pattern_id}",
                pattern_id=pattern.pattern_id,
                recommendation_type="pattern_based",
                description=f"Address {pattern.pattern_type} pattern affecting {len(pattern.affected_fields)} fields",
                suggested_actions=pattern.suggested_fixes,
                expected_impact=pattern.impact_score,
                priority="high" if pattern.impact_score > 0.7 else "medium",
                generated_at=datetime.now()
            )
            recommendations.append(recommendation)
        
        # Add general recommendations based on statistics
        performance_metrics = statistics_engine.get_performance_metrics()
        
        if performance_metrics.get("average_accuracy", 0) < 0.8:
            recommendations.append(OptimizationRecommendation(
                recommendation_id="rec_low_accuracy",
                pattern_id=None,
                recommendation_type="performance_based",
                description="Overall accuracy is below target threshold",
                suggested_actions=[
                    "Review field extraction instructions",
                    "Add more specific field definitions",
                    "Include validation examples"
                ],
                expected_impact=0.2,
                priority="high",
                generated_at=datetime.now()
            ))
        
        if performance_metrics.get("confidence_correlation", 0) < 0.7:
            recommendations.append(OptimizationRecommendation(
                recommendation_id="rec_confidence_calibration",
                pattern_id=None,
                recommendation_type="calibration_based",
                description="Low correlation between confidence scores and actual accuracy",
                suggested_actions=[
                    "Review confidence calibration",
                    "Improve validation logic",
                    "Add confidence threshold adjustments"
                ],
                expected_impact=0.15,
                priority="medium",
                generated_at=datetime.now()
            ))
        
        return recommendations


# Initialize global evaluator
evaluator = DocumentExtractionEvaluator()


@app.post("/evaluate", response_model=DocumentEvaluationResult)
async def evaluate_document(evaluation_input: DocumentEvaluationInput):
    """
    Evaluate document extraction results.
    
    This endpoint accepts extracted fields and ground truth, then evaluates
    the extraction quality using statistical analysis.
    """
    
    try:
        result = evaluator.evaluate_document(evaluation_input)
        return result
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/stats", response_model=EvaluationStatistics)
async def get_statistics():
    """Get current evaluation statistics."""
    
    return statistics_engine.statistics


@app.get("/performance-metrics")
async def get_performance_metrics():
    """Get comprehensive performance metrics."""
    
    return statistics_engine.get_performance_metrics()


@app.get("/field-performance")
async def get_field_performance():
    """Get field-level performance analysis."""
    
    return statistics_engine.get_field_performance()


@app.get("/document-type-performance")
async def get_document_type_performance():
    """Get document type performance analysis."""
    
    return statistics_engine.get_document_type_performance()


@app.post("/optimize")
async def generate_optimization_recommendations(
    background_tasks: BackgroundTasks = None
):
    """
    Generate optimization recommendations based on evaluation data.
    
    This endpoint analyzes current evaluation results and provides
    data-driven recommendations for prompt improvement.
    """
    
    try:
        # Get recent evaluation results (in a real implementation, this would come from a database)
        # For now, we'll use the statistics engine data
        recent_results = []  # This would be populated from persistent storage
        
        recommendations = evaluator.get_optimization_recommendations(recent_results)
        
        return {
            "recommendations": [rec.dict() for rec in recommendations],
            "total_recommendations": len(recommendations),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.get("/error-patterns")
async def get_error_patterns():
    """Get detected error patterns."""
    
    try:
        # In a real implementation, this would analyze recent evaluation results
        # For now, return empty patterns
        return {
            "patterns": [],
            "total_patterns": 0,
            "analyzed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error pattern analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pattern analysis failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "service": "Document Extraction Evaluation Service"
    }


@app.get("/config")
async def get_configuration():
    """Get current configuration."""
    
    return {
        "evaluation_config": evaluation_config.dict(),
        "service_config": get_config()
    }


@app.post("/config")
async def update_configuration(config_update: Dict[str, Any]):
    """Update configuration."""
    
    try:
        # Update evaluation config
        for key, value in config_update.items():
            if hasattr(evaluation_config, key):
                setattr(evaluation_config, key, value)
        
        logger.info("Configuration updated")
        
        return {
            "message": "Configuration updated successfully",
            "updated_config": evaluation_config.dict()
        }
        
    except Exception as e:
        logger.error(f"Configuration update failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Configuration update failed: {str(e)}")


@app.post("/reset")
async def reset_statistics():
    """Reset evaluation statistics."""
    
    try:
        statistics_engine.reset_statistics()
        logger.info("Statistics reset")
        
        return {
            "message": "Statistics reset successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Statistics reset failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Statistics reset failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with service information."""
    
    return {
        "service": "Document Extraction Evaluation Service",
        "version": "1.0.0",
        "description": "Statistical evaluation service for OCR-plus-prompt pipeline outputs",
        "endpoints": {
            "evaluate": "/evaluate",
            "stats": "/stats",
            "performance": "/performance-metrics",
            "field_performance": "/field-performance",
            "optimize": "/optimize",
            "health": "/health",
            "config": "/config"
        },
        "documentation": "/docs"
    } 