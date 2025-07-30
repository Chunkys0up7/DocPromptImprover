"""
Pydantic models for evaluation-only document extraction framework.

This module defines the data structures for evaluating the outputs of existing
OCR-plus-prompt pipelines, focusing on metrics, statistics, and optimization feedback.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
import json


class ExtractionStatus(str, Enum):
    """Status of field extraction evaluation."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    MISSING = "missing"


class FieldEvaluationResult(BaseModel):
    """Result of evaluating a single field extraction."""
    
    field_name: str = Field(..., description="Name of the evaluated field")
    expected_value: Optional[str] = Field(None, description="Ground truth value")
    extracted_value: Optional[str] = Field(None, description="Value extracted by the model")
    confidence_score: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0, 
        description="Confidence score from the extraction model"
    )
    status: ExtractionStatus = Field(
        default=ExtractionStatus.FAILED,
        description="Evaluation status of the field"
    )
    evaluation_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Score from evaluation metric (0.0-1.0)"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if extraction failed"
    )
    evaluation_notes: Optional[str] = Field(
        None, description="Additional notes from evaluation"
    )
    field_type: Optional[str] = Field(
        None, description="Expected data type of the field"
    )

    @validator('evaluation_score')
    def validate_evaluation_score(cls, v):
        """Ensure evaluation score is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Evaluation score must be between 0.0 and 1.0')
        return v

    def is_successful(self) -> bool:
        """Check if field extraction was successful."""
        return self.status == ExtractionStatus.SUCCESS

    def is_partial(self) -> bool:
        """Check if field extraction was partially successful."""
        return self.status == ExtractionStatus.PARTIAL

    def is_failed(self) -> bool:
        """Check if field extraction failed."""
        return self.status in [ExtractionStatus.FAILED, ExtractionStatus.MISSING]


class DocumentEvaluationInput(BaseModel):
    """Input data for document evaluation."""
    
    document_id: str = Field(..., description="Unique identifier for the document")
    document_type: str = Field(..., description="Type of document (invoice, receipt, etc.)")
    extracted_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description="Fields extracted by the OCR/prompt pipeline"
    )
    ground_truth: Dict[str, Any] = Field(
        default_factory=dict,
        description="Ground truth values for comparison"
    )
    confidence_scores: Optional[Dict[str, float]] = Field(
        None, description="Confidence scores for extracted fields"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional document metadata"
    )
    prompt_version: Optional[str] = Field(
        None, description="Version of the prompt used for extraction"
    )
    extraction_timestamp: Optional[datetime] = Field(
        None, description="When the extraction was performed"
    )


class DocumentEvaluationResult(BaseModel):
    """Complete result of document evaluation."""
    
    document_id: str = Field(..., description="Document identifier")
    document_type: str = Field(..., description="Document type")
    field_evaluations: List[FieldEvaluationResult] = Field(
        default_factory=list,
        description="Evaluation results for each field"
    )
    overall_accuracy: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall accuracy score (0.0-1.0)"
    )
    confidence_correlation: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Correlation between confidence and accuracy"
    )
    evaluation_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the evaluation was performed"
    )
    prompt_version: Optional[str] = Field(
        None, description="Version of prompt used"
    )
    evaluation_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional evaluation metadata"
    )

    def get_successful_fields(self) -> List[FieldEvaluationResult]:
        """Get list of successfully extracted fields."""
        return [field for field in self.field_evaluations if field.is_successful()]

    def get_failed_fields(self) -> List[FieldEvaluationResult]:
        """Get list of failed field extractions."""
        return [field for field in self.field_evaluations if field.is_failed()]

    def get_field_by_name(self, field_name: str) -> Optional[FieldEvaluationResult]:
        """Get a specific field evaluation by name."""
        for field in self.field_evaluations:
            if field.field_name == field_name:
                return field
        return None

    def calculate_accuracy(self) -> float:
        """Calculate overall accuracy based on field evaluations."""
        if not self.field_evaluations:
            return 0.0
        
        total_score = sum(field.evaluation_score for field in self.field_evaluations)
        return total_score / len(self.field_evaluations)


class EvaluationStatistics(BaseModel):
    """Aggregated statistics for evaluation results."""
    
    total_documents: int = Field(default=0, description="Total documents evaluated")
    total_fields: int = Field(default=0, description="Total fields evaluated")
    successful_extractions: int = Field(default=0, description="Number of successful extractions")
    failed_extractions: int = Field(default=0, description="Number of failed extractions")
    partial_extractions: int = Field(default=0, description="Number of partial extractions")
    
    field_success_rates: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Success rates for each field type"
    )
    average_accuracy: float = Field(default=0.0, description="Average accuracy across all documents")
    average_confidence: float = Field(default=0.0, description="Average confidence score")
    
    common_errors: Dict[str, int] = Field(
        default_factory=dict,
        description="Frequency of common error types"
    )
    confidence_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of confidence scores"
    )
    
    prompt_version_performance: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance by prompt version"
    )
    document_type_performance: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance by document type"
    )
    
    evaluation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of evaluation runs"
    )

    def update_statistics(self, result: DocumentEvaluationResult) -> None:
        """Update statistics with new evaluation result."""
        self.total_documents += 1
        
        # Update field-level statistics
        for field in result.field_evaluations:
            self.total_fields += 1
            
            if field.is_successful():
                self.successful_extractions += 1
            elif field.is_failed():
                self.failed_extractions += 1
            elif field.is_partial():
                self.partial_extractions += 1
            
            # Update field success rates
            field_name = field.field_name
            success_score = 1.0 if field.is_successful() else 0.0
            self.field_success_rates.setdefault(field_name, []).append(success_score)
            
            # Track common errors
            if field.error_message:
                self.common_errors[field.error_message] = self.common_errors.get(field.error_message, 0) + 1
            
            # Categorize confidence levels
            if field.confidence_score > 0.8:
                self.confidence_distribution["high"] = self.confidence_distribution.get("high", 0) + 1
            elif field.confidence_score > 0.5:
                self.confidence_distribution["medium"] = self.confidence_distribution.get("medium", 0) + 1
            else:
                self.confidence_distribution["low"] = self.confidence_distribution.get("low", 0) + 1

    def get_optimization_metrics(self) -> Dict[str, float]:
        """Calculate metrics for DSPy optimization."""
        import statistics
        
        metrics = {}
        
        # Field-specific metrics
        for field_name, scores in self.field_success_rates.items():
            if scores:
                metrics[f"{field_name}_success_rate"] = statistics.mean(scores)
                metrics[f"{field_name}_consistency"] = (
                    1.0 - statistics.stdev(scores) if len(scores) > 1 else 1.0
                )
        
        # Overall metrics
        if self.total_documents > 0:
            metrics["overall_success_rate"] = self.successful_extractions / self.total_fields
            metrics["error_rate"] = self.failed_extractions / self.total_fields
            metrics["average_accuracy"] = self.average_accuracy
        
        return metrics


class FailurePattern(BaseModel):
    """Pattern of failures identified in evaluations."""
    
    pattern_id: str = Field(..., description="Unique identifier for the pattern")
    pattern_type: str = Field(..., description="Type of failure pattern")
    affected_fields: List[str] = Field(
        default_factory=list,
        description="Fields affected by this pattern"
    )
    error_messages: List[str] = Field(
        default_factory=list,
        description="Common error messages for this pattern"
    )
    frequency: int = Field(default=0, description="How often this pattern occurs")
    impact_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Impact score of this pattern (0.0-1.0)"
    )
    suggested_fixes: List[str] = Field(
        default_factory=list,
        description="Suggested fixes for this pattern"
    )
    first_seen: datetime = Field(
        default_factory=datetime.now,
        description="When this pattern was first identified"
    )
    last_seen: datetime = Field(
        default_factory=datetime.now,
        description="When this pattern was last seen"
    )


class OptimizationRecommendation(BaseModel):
    """Recommendation for prompt optimization."""
    
    recommendation_id: str = Field(..., description="Unique identifier for the recommendation")
    priority: str = Field(..., description="Priority level (high, medium, low)")
    target_fields: List[str] = Field(
        default_factory=list,
        description="Fields targeted by this optimization"
    )
    failure_patterns: List[str] = Field(
        default_factory=list,
        description="Failure patterns this optimization addresses"
    )
    suggested_prompt_changes: List[str] = Field(
        default_factory=list,
        description="Suggested changes to the prompt"
    )
    expected_improvement: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Expected improvement in accuracy (0.0-1.0)"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this recommendation (0.0-1.0)"
    )
    reasoning: str = Field(..., description="Explanation for this recommendation")
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this recommendation was created"
    )


class EvaluationConfig(BaseModel):
    """Configuration for evaluation parameters."""
    
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for success"
    )
    success_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum evaluation score for success"
    )
    partial_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum score for partial success"
    )
    enable_partial_credit: bool = Field(
        default=True,
        description="Whether to enable partial credit scoring"
    )
    strict_matching: bool = Field(
        default=False,
        description="Whether to use strict string matching"
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether string comparisons are case sensitive"
    )
    normalize_whitespace: bool = Field(
        default=True,
        description="Whether to normalize whitespace in comparisons"
    )
    
    class Config:
        validate_assignment = True 