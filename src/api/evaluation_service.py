"""
FastAPI evaluation service for document extraction evaluation.

This module provides the main API endpoints for evaluating the outputs of
existing OCR-plus-prompt pipelines and generating optimization feedback.
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

import dspy
from dspy.teleprompt import MIPROv2

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
from ..evaluators.evaluation_signatures import (
    FieldEvaluationSignature,
    DocumentAggregationSignature,
    FailurePatternAnalysisSignature,
    PromptOptimizationSignature,
    get_evaluation_signature
)
from ..utils.config import get_config
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Document Extraction Evaluation Service",
    description="Evaluation-only microservice for assessing OCR-plus-prompt pipeline outputs",
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
evaluation_statistics = EvaluationStatistics()
evaluation_config = EvaluationConfig()


class DocumentExtractionEvaluator(dspy.Module):
    """
    Main evaluation module for document extraction results.
    
    This module evaluates the outputs of existing OCR-plus-prompt pipelines
    and provides detailed feedback for optimization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the evaluator with configuration."""
        super().__init__()
        
        self.config = config or get_config()
        
        # Initialize DSPy modules
        self.field_evaluator = dspy.ChainOfThought(FieldEvaluationSignature)
        self.document_aggregator = dspy.ChainOfThought(DocumentAggregationSignature)
        self.failure_analyzer = dspy.ChainOfThought(FailurePatternAnalysisSignature)
        self.optimizer = dspy.ChainOfThought(PromptOptimizationSignature)
        
        logger.info("DocumentExtractionEvaluator initialized")
    
    def evaluate_field(self, 
                      field_name: str,
                      expected_value: Optional[str],
                      extracted_value: Optional[str],
                      confidence_score: float = 0.0,
                      field_type: str = "text") -> FieldEvaluationResult:
        """
        Evaluate a single field extraction.
        
        Args:
            field_name: Name of the field
            expected_value: Ground truth value
            extracted_value: Value extracted by the model
            confidence_score: Confidence score from the model
            field_type: Expected data type of the field
            
        Returns:
            FieldEvaluationResult: Evaluation result for the field
        """
        
        try:
            # Use DSPy to evaluate the field
            evaluation_result = self.field_evaluator(
                field_name=field_name,
                expected_value=str(expected_value) if expected_value is not None else "",
                extracted_value=str(extracted_value) if extracted_value is not None else "",
                confidence_score=confidence_score,
                field_type=field_type
            )
            
            # Parse evaluation results
            try:
                evaluation_score = float(evaluation_result.evaluation_score)
                status_str = evaluation_result.status.lower()
                
                # Map status string to enum
                status_mapping = {
                    "success": ExtractionStatus.SUCCESS,
                    "partial": ExtractionStatus.PARTIAL,
                    "failed": ExtractionStatus.FAILED,
                    "missing": ExtractionStatus.MISSING
                }
                status = status_mapping.get(status_str, ExtractionStatus.FAILED)
                
            except (ValueError, AttributeError) as e:
                logger.warning(f"Failed to parse evaluation result for field {field_name}: {e}")
                evaluation_score = 0.0
                status = ExtractionStatus.FAILED
            
            # Create field evaluation result
            field_result = FieldEvaluationResult(
                field_name=field_name,
                expected_value=expected_value,
                extracted_value=extracted_value,
                confidence_score=confidence_score,
                status=status,
                evaluation_score=evaluation_score,
                error_message=evaluation_result.error_message if hasattr(evaluation_result, 'error_message') else None,
                evaluation_notes=evaluation_result.evaluation_notes if hasattr(evaluation_result, 'evaluation_notes') else None,
                field_type=field_type
            )
            
            return field_result
            
        except Exception as e:
            logger.error(f"Error evaluating field {field_name}: {e}")
            # Return failed result
            return FieldEvaluationResult(
                field_name=field_name,
                expected_value=expected_value,
                extracted_value=extracted_value,
                confidence_score=confidence_score,
                status=ExtractionStatus.FAILED,
                evaluation_score=0.0,
                error_message=f"Evaluation error: {str(e)}",
                field_type=field_type
            )
    
    def evaluate_document(self, 
                         evaluation_input: DocumentEvaluationInput) -> DocumentEvaluationResult:
        """
        Evaluate a complete document extraction.
        
        Args:
            evaluation_input: Input data for document evaluation
            
        Returns:
            DocumentEvaluationResult: Complete evaluation result
        """
        
        start_time = time.time()
        
        logger.info(f"Evaluating document {evaluation_input.document_id}")
        
        try:
            # Evaluate each field
            field_evaluations = []
            confidence_scores = evaluation_input.confidence_scores or {}
            
            # Get all unique field names from both extracted and ground truth
            all_fields = set(evaluation_input.extracted_fields.keys())
            all_fields.update(evaluation_input.ground_truth.keys())
            
            for field_name in all_fields:
                expected_value = evaluation_input.ground_truth.get(field_name)
                extracted_value = evaluation_input.extracted_fields.get(field_name)
                confidence_score = confidence_scores.get(field_name, 0.0)
                
                # Determine field type (basic heuristic)
                field_type = "text"
                if expected_value is not None:
                    if isinstance(expected_value, (int, float)):
                        field_type = "number"
                    elif isinstance(expected_value, str):
                        # Simple date detection
                        if any(char in expected_value for char in ["-", "/", "."]):
                            field_type = "date"
                
                field_result = self.evaluate_field(
                    field_name=field_name,
                    expected_value=expected_value,
                    extracted_value=extracted_value,
                    confidence_score=confidence_score,
                    field_type=field_type
                )
                field_evaluations.append(field_result)
            
            # Aggregate document-level metrics using DSPy
            try:
                aggregation_result = self.document_aggregator(
                    field_evaluations=json.dumps([f.dict() for f in field_evaluations]),
                    document_type=evaluation_input.document_type,
                    confidence_scores=json.dumps(confidence_scores)
                )
                
                overall_accuracy = float(aggregation_result.overall_accuracy)
                confidence_correlation = float(aggregation_result.confidence_correlation)
                
            except (ValueError, AttributeError) as e:
                logger.warning(f"Failed to parse aggregation result: {e}")
                # Calculate manually
                overall_accuracy = sum(f.evaluation_score for f in field_evaluations) / len(field_evaluations) if field_evaluations else 0.0
                confidence_correlation = 0.0
            
            # Create document evaluation result
            result = DocumentEvaluationResult(
                document_id=evaluation_input.document_id,
                document_type=evaluation_input.document_type,
                field_evaluations=field_evaluations,
                overall_accuracy=overall_accuracy,
                confidence_correlation=confidence_correlation,
                prompt_version=evaluation_input.prompt_version,
                evaluation_metadata={
                    "processing_time": time.time() - start_time,
                    "total_fields": len(field_evaluations),
                    "successful_fields": len([f for f in field_evaluations if f.is_successful()]),
                    "failed_fields": len([f for f in field_evaluations if f.is_failed()])
                }
            )
            
            logger.info(f"Document {evaluation_input.document_id} evaluation completed. "
                       f"Accuracy: {overall_accuracy:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating document {evaluation_input.document_id}: {e}")
            # Return error result
            return DocumentEvaluationResult(
                document_id=evaluation_input.document_id,
                document_type=evaluation_input.document_type,
                field_evaluations=[],
                overall_accuracy=0.0,
                confidence_correlation=0.0,
                evaluation_metadata={"error": str(e)}
            )


# Global evaluator instance
evaluator = DocumentExtractionEvaluator()


@app.post("/evaluate", response_model=DocumentEvaluationResult)
async def evaluate_document(evaluation_input: DocumentEvaluationInput):
    """
    Evaluate a document extraction result.
    
    This endpoint accepts ground truth and extracted values from an existing
    OCR-plus-prompt pipeline and returns detailed evaluation metrics.
    """
    
    try:
        # Validate input
        if not evaluation_input.extracted_fields and not evaluation_input.ground_truth:
            raise HTTPException(
                status_code=400,
                detail="At least one of extracted_fields or ground_truth must be provided"
            )
        
        # Perform evaluation
        result = evaluator.evaluate_document(evaluation_input)
        
        # Update global statistics
        evaluation_statistics.update_statistics(result)
        
        return result
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        logger.error(f"Error in evaluate endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/stats", response_model=EvaluationStatistics)
async def get_statistics():
    """
    Get current evaluation statistics.
    
    Returns aggregated statistics from all evaluations performed.
    """
    
    return evaluation_statistics


@app.post("/optimize")
async def optimize_prompt(
    current_prompt: str,
    target_improvement: float = 0.1,
    background_tasks: BackgroundTasks = None
):
    """
    Generate prompt optimization recommendations.
    
    This endpoint analyzes current evaluation statistics and failure patterns
    to generate recommendations for prompt improvement.
    """
    
    try:
        # Get optimization metrics
        metrics = evaluation_statistics.get_optimization_metrics()
        
        if not metrics:
            raise HTTPException(
                status_code=400,
                detail="No evaluation data available for optimization"
            )
        
        # Use DSPy to generate optimization recommendations
        optimization_result = evaluator.optimizer(
            evaluation_statistics=json.dumps(metrics),
            failure_patterns=json.dumps(evaluation_statistics.common_errors),
            current_prompt=current_prompt,
            target_improvement=target_improvement
        )
        
        # Create optimization recommendation
        recommendation = OptimizationRecommendation(
            recommendation_id=f"opt_{datetime.now().isoformat()}",
            priority="high" if target_improvement > 0.2 else "medium",
            target_fields=list(evaluation_statistics.field_success_rates.keys()),
            failure_patterns=list(evaluation_statistics.common_errors.keys()),
            suggested_prompt_changes=[optimization_result.optimized_prompt],
            expected_improvement=float(optimization_result.expected_improvement),
            confidence=float(optimization_result.confidence_in_improvement),
            reasoning=optimization_result.improvement_rationale
        )
        
        return {
            "recommendation": recommendation.dict(),
            "current_metrics": metrics,
            "optimization_result": {
                "optimized_prompt": optimization_result.optimized_prompt,
                "improvement_rationale": optimization_result.improvement_rationale,
                "expected_improvement": optimization_result.expected_improvement,
                "confidence_in_improvement": optimization_result.confidence_in_improvement
            }
        }
        
    except Exception as e:
        logger.error(f"Error in optimize endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns service health status and basic metrics.
    """
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "total_documents_evaluated": evaluation_statistics.total_documents,
        "total_fields_evaluated": evaluation_statistics.total_fields,
        "average_accuracy": evaluation_statistics.average_accuracy
    }


@app.get("/config")
async def get_configuration():
    """
    Get current evaluation configuration.
    
    Returns the current configuration parameters.
    """
    
    return evaluation_config.dict()


@app.post("/config")
async def update_configuration(config_update: Dict[str, Any]):
    """
    Update evaluation configuration.
    
    Updates the evaluation parameters with new values.
    """
    
    try:
        global evaluation_config
        
        # Update configuration
        for key, value in config_update.items():
            if hasattr(evaluation_config, key):
                setattr(evaluation_config, key, value)
        
        logger.info(f"Configuration updated: {config_update}")
        
        return {"message": "Configuration updated successfully", "config": evaluation_config.dict()}
        
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        raise HTTPException(status_code=400, detail=f"Configuration update failed: {str(e)}")


@app.post("/reset")
async def reset_statistics():
    """
    Reset evaluation statistics.
    
    Clears all accumulated statistics and starts fresh.
    """
    
    global evaluation_statistics
    evaluation_statistics = EvaluationStatistics()
    
    logger.info("Evaluation statistics reset")
    
    return {"message": "Statistics reset successfully"}


if __name__ == "__main__":
    import uvicorn
    
    # Setup DSPy
    try:
        lm = dspy.OpenAI(model="gpt-3.5-turbo", max_tokens=1000)
        dspy.settings.configure(lm=lm)
        logger.info("DSPy configured with OpenAI")
    except Exception as e:
        logger.warning(f"Failed to configure DSPy: {e}")
    
    # Run the service
    uvicorn.run(
        "evaluation_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 