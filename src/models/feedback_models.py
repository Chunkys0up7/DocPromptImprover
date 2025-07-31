"""
Feedback models for user-driven prompt evaluation system.

This module defines Pydantic models for collecting, validating, and processing
user feedback on document extraction results.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class FeedbackStatus(str, Enum):
    """Status of user feedback for a field."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"


class FeedbackReason(str, Enum):
    """Reason codes for user feedback."""
    BAD_OCR = "bad_ocr"
    PROMPT_AMBIGUOUS = "prompt_ambiguous"
    IRRELEVANT_TEXT = "irrelevant_text"
    WRONG_FORMAT = "wrong_format"
    MISSING_VALUE = "missing_value"
    EXTRANEOUS_VALUE = "extraneous_value"
    CONFUSING_CONTEXT = "confusing_context"
    OTHER = "other"


class FieldFeedback(BaseModel):
    """User feedback for a single field."""
    field_name: str = Field(..., description="Name of the field being evaluated")
    shown_value: Optional[str] = Field(None, description="Value that was shown to the user")
    feedback_status: FeedbackStatus = Field(..., description="User's assessment of the field")
    correction: Optional[str] = Field(None, description="Corrected value provided by user")
    comment: Optional[str] = Field(None, description="User's comment about the field")
    reason_code: Optional[FeedbackReason] = Field(None, description="Reason for the feedback")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Original confidence score")
    
    @validator('correction')
    def validate_correction(cls, v, values):
        """Validate that correction is provided when status is incorrect."""
        if values.get('feedback_status') == FeedbackStatus.INCORRECT and not v:
            raise ValueError("Correction must be provided when status is incorrect")
        return v


class UserFeedbackRecord(BaseModel):
    """Complete user feedback record for a document."""
    feedback_id: str = Field(..., description="Unique identifier for this feedback record")
    document_id: str = Field(..., description="ID of the document being evaluated")
    user_id: str = Field(..., description="ID of the user providing feedback")
    session_id: Optional[str] = Field(None, description="Session ID for anonymous users")
    prompt_version: str = Field(..., description="Version of the prompt used for extraction")
    document_type: str = Field(..., description="Type of document being evaluated")
    timestamp: datetime = Field(default_factory=datetime.now, description="When feedback was provided")
    field_feedback: List[FieldFeedback] = Field(..., description="Feedback for each field")
    overall_comment: Optional[str] = Field(None, description="Overall comment about the document")
    processing_time: Optional[float] = Field(None, description="Time taken to process feedback")
    
    @validator('field_feedback')
    def validate_field_feedback(cls, v):
        """Validate that at least one field feedback is provided."""
        if not v:
            raise ValueError("At least one field feedback must be provided")
        return v


class FeedbackAggregation(BaseModel):
    """Aggregated feedback statistics."""
    field_name: str = Field(..., description="Name of the field")
    total_feedback: int = Field(..., description="Total number of feedback records")
    correct_count: int = Field(..., description="Number of correct assessments")
    incorrect_count: int = Field(..., description="Number of incorrect assessments")
    partial_count: int = Field(..., description="Number of partial assessments")
    accuracy_rate: float = Field(..., ge=0.0, le=1.0, description="Accuracy rate based on feedback")
    common_reasons: List[Dict[str, Any]] = Field(..., description="Most common reason codes")
    sample_comments: List[str] = Field(..., description="Sample user comments")
    prompt_versions: List[str] = Field(..., description="Prompt versions involved")
    document_types: List[str] = Field(..., description="Document types involved")


class FeedbackTrend(BaseModel):
    """Feedback trend over time."""
    field_name: str = Field(..., description="Name of the field")
    time_period: str = Field(..., description="Time period (day, week, month)")
    date: str = Field(..., description="Date of the period")
    total_feedback: int = Field(..., description="Total feedback in period")
    accuracy_rate: float = Field(..., description="Accuracy rate in period")
    incorrect_count: int = Field(..., description="Number of incorrect assessments")
    trend_direction: str = Field(..., description="Trend direction (improving, declining, stable)")


class FeedbackAlert(BaseModel):
    """Alert for feedback-based issues."""
    alert_id: str = Field(..., description="Unique alert identifier")
    field_name: str = Field(..., description="Field causing the alert")
    alert_type: str = Field(..., description="Type of alert (accuracy_drop, high_error_rate, etc.)")
    severity: str = Field(..., description="Alert severity (low, medium, high, critical)")
    description: str = Field(..., description="Description of the alert")
    threshold_value: float = Field(..., description="Threshold that was exceeded")
    current_value: float = Field(..., description="Current value")
    prompt_version: str = Field(..., description="Prompt version affected")
    created_at: datetime = Field(default_factory=datetime.now, description="When alert was created")
    resolved_at: Optional[datetime] = Field(None, description="When alert was resolved")
    status: str = Field(default="active", description="Alert status")


class FeedbackOptimizationRecommendation(BaseModel):
    """Optimization recommendation based on user feedback."""
    recommendation_id: str = Field(..., description="Unique recommendation identifier")
    field_name: str = Field(..., description="Field the recommendation applies to")
    recommendation_type: str = Field(..., description="Type of recommendation")
    description: str = Field(..., description="Description of the recommendation")
    suggested_actions: List[str] = Field(..., description="Suggested actions to take")
    feedback_evidence: List[str] = Field(..., description="Evidence from user feedback")
    expected_impact: float = Field(..., ge=0.0, le=1.0, description="Expected impact on accuracy")
    priority: str = Field(..., description="Priority level (low, medium, high, critical)")
    affected_prompt_versions: List[str] = Field(..., description="Prompt versions affected")
    generated_at: datetime = Field(default_factory=datetime.now, description="When recommendation was generated")


class FeedbackStatistics(BaseModel):
    """Overall feedback statistics."""
    total_feedback_records: int = Field(..., description="Total number of feedback records")
    total_fields_evaluated: int = Field(..., description="Total number of field evaluations")
    overall_accuracy_rate: float = Field(..., ge=0.0, le=1.0, description="Overall accuracy rate")
    most_problematic_fields: List[Dict[str, Any]] = Field(..., description="Fields with highest error rates")
    feedback_trends: List[FeedbackTrend] = Field(..., description="Feedback trends over time")
    active_alerts: List[FeedbackAlert] = Field(..., description="Currently active alerts")
    recent_optimizations: List[FeedbackOptimizationRecommendation] = Field(..., description="Recent optimization recommendations")
    last_updated: datetime = Field(default_factory=datetime.now, description="When statistics were last updated") 