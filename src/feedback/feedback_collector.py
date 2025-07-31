"""
Feedback collection and management system.

This module handles the collection, validation, and storage of user feedback
for document extraction evaluation.
"""

import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter

from ..models.feedback_models import (
    UserFeedbackRecord,
    FieldFeedback,
    FeedbackStatus,
    FeedbackReason,
    FeedbackAggregation,
    FeedbackTrend,
    FeedbackAlert,
    FeedbackOptimizationRecommendation,
    FeedbackStatistics
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class FeedbackCollector:
    """
    Collects and manages user feedback for document extraction evaluation.
    
    This class provides methods for collecting, validating, and storing
    user feedback, as well as generating feedback-based insights.
    """
    
    def __init__(self):
        """Initialize the feedback collector."""
        self.feedback_records: List[UserFeedbackRecord] = []
        self.alerts: List[FeedbackAlert] = []
        self.optimization_recommendations: List[FeedbackOptimizationRecommendation] = []
        
        # Configuration
        self.alert_thresholds = {
            "accuracy_drop": 0.1,  # 10% drop in accuracy
            "high_error_rate": 0.3,  # 30% error rate
            "feedback_spike": 2.0,  # 2x increase in feedback volume
        }
        
        logger.info("FeedbackCollector initialized")
    
    def collect_feedback(self, feedback_data: Dict[str, Any]) -> UserFeedbackRecord:
        """
        Collect and validate user feedback.
        
        Args:
            feedback_data: Raw feedback data from the frontend
            
        Returns:
            UserFeedbackRecord: Validated feedback record
        """
        start_time = time.time()
        
        try:
            # Generate unique feedback ID
            feedback_id = str(uuid.uuid4())
            
            # Create feedback record
            feedback_record = UserFeedbackRecord(
                feedback_id=feedback_id,
                document_id=feedback_data["document_id"],
                user_id=feedback_data["user_id"],
                session_id=feedback_data.get("session_id"),
                prompt_version=feedback_data["prompt_version"],
                document_type=feedback_data["document_type"],
                field_feedback=[
                    FieldFeedback(**field_data) 
                    for field_data in feedback_data["field_feedback"]
                ],
                overall_comment=feedback_data.get("overall_comment")
            )
            
            # Store the feedback
            self.feedback_records.append(feedback_record)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            feedback_record.processing_time = processing_time
            
            # Check for alerts
            self._check_for_alerts(feedback_record)
            
            # Generate optimization recommendations if needed
            self._generate_optimization_recommendations()
            
            logger.info(f"Feedback collected for document {feedback_record.document_id}")
            
            return feedback_record
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {str(e)}")
            raise ValueError(f"Invalid feedback data: {str(e)}")
    
    def get_feedback_aggregation(self, 
                                field_name: Optional[str] = None,
                                prompt_version: Optional[str] = None,
                                document_type: Optional[str] = None,
                                time_period: Optional[str] = None) -> List[FeedbackAggregation]:
        """
        Get aggregated feedback statistics.
        
        Args:
            field_name: Filter by specific field
            prompt_version: Filter by prompt version
            document_type: Filter by document type
            time_period: Filter by time period (e.g., "7d", "30d")
            
        Returns:
            List[FeedbackAggregation]: Aggregated feedback statistics
        """
        # Filter feedback records
        filtered_records = self._filter_feedback_records(
            field_name, prompt_version, document_type, time_period
        )
        
        # Group by field
        field_groups = defaultdict(list)
        for record in filtered_records:
            for field_feedback in record.field_feedback:
                if field_name is None or field_feedback.field_name == field_name:
                    field_groups[field_feedback.field_name].append((record, field_feedback))
        
        aggregations = []
        
        for field_name, field_data in field_groups.items():
            # Calculate statistics
            total_feedback = len(field_data)
            correct_count = sum(1 for _, feedback in field_data 
                              if feedback.feedback_status == FeedbackStatus.CORRECT)
            incorrect_count = sum(1 for _, feedback in field_data 
                                if feedback.feedback_status == FeedbackStatus.INCORRECT)
            partial_count = sum(1 for _, feedback in field_data 
                              if feedback.feedback_status == FeedbackStatus.PARTIAL)
            
            accuracy_rate = correct_count / total_feedback if total_feedback > 0 else 0.0
            
            # Get common reasons
            reason_counter = Counter()
            for _, feedback in field_data:
                if feedback.reason_code:
                    reason_counter[feedback.reason_code] += 1
            
            common_reasons = [
                {"reason": reason, "count": count, "percentage": count / total_feedback}
                for reason, count in reason_counter.most_common(5)
            ]
            
            # Get sample comments
            sample_comments = []
            for _, feedback in field_data:
                if feedback.comment and len(sample_comments) < 5:
                    sample_comments.append(feedback.comment)
            
            # Get prompt versions and document types
            prompt_versions = list(set(record.prompt_version for record, _ in field_data))
            document_types = list(set(record.document_type for record, _ in field_data))
            
            aggregation = FeedbackAggregation(
                field_name=field_name,
                total_feedback=total_feedback,
                correct_count=correct_count,
                incorrect_count=incorrect_count,
                partial_count=partial_count,
                accuracy_rate=accuracy_rate,
                common_reasons=common_reasons,
                sample_comments=sample_comments,
                prompt_versions=prompt_versions,
                document_types=document_types
            )
            
            aggregations.append(aggregation)
        
        return aggregations
    
    def get_feedback_trends(self, 
                           field_name: Optional[str] = None,
                           time_period: str = "7d") -> List[FeedbackTrend]:
        """
        Get feedback trends over time.
        
        Args:
            field_name: Filter by specific field
            time_period: Time period for trends (e.g., "7d", "30d")
            
        Returns:
            List[FeedbackTrend]: Feedback trends over time
        """
        # Parse time period
        days = int(time_period[:-1]) if time_period.endswith('d') else 7
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Filter records by date
        recent_records = [
            record for record in self.feedback_records
            if start_date <= record.timestamp <= end_date
        ]
        
        # Group by date and field
        trends = []
        date_groups = defaultdict(lambda: defaultdict(list))
        
        for record in recent_records:
            date_key = record.timestamp.strftime("%Y-%m-%d")
            for field_feedback in record.field_feedback:
                if field_name is None or field_feedback.field_name == field_name:
                    date_groups[date_key][field_feedback.field_name].append(field_feedback)
        
        for date_key, field_groups in date_groups.items():
            for field_name, field_data in field_groups.items():
                total_feedback = len(field_data)
                correct_count = sum(1 for feedback in field_data 
                                  if feedback.feedback_status == FeedbackStatus.CORRECT)
                incorrect_count = sum(1 for feedback in field_data 
                                    if feedback.feedback_status == FeedbackStatus.INCORRECT)
                
                accuracy_rate = correct_count / total_feedback if total_feedback > 0 else 0.0
                
                # Determine trend direction (simplified)
                trend_direction = "stable"
                if total_feedback > 5:  # Need sufficient data
                    if accuracy_rate > 0.8:
                        trend_direction = "improving"
                    elif accuracy_rate < 0.6:
                        trend_direction = "declining"
                
                trend = FeedbackTrend(
                    field_name=field_name,
                    time_period="day",
                    date=date_key,
                    total_feedback=total_feedback,
                    accuracy_rate=accuracy_rate,
                    incorrect_count=incorrect_count,
                    trend_direction=trend_direction
                )
                
                trends.append(trend)
        
        return trends
    
    def get_active_alerts(self) -> List[FeedbackAlert]:
        """Get currently active alerts."""
        return [alert for alert in self.alerts if alert.status == "active"]
    
    def get_optimization_recommendations(self) -> List[FeedbackOptimizationRecommendation]:
        """Get recent optimization recommendations."""
        return self.optimization_recommendations[-10:]  # Last 10 recommendations
    
    def get_feedback_statistics(self) -> FeedbackStatistics:
        """Get overall feedback statistics."""
        total_records = len(self.feedback_records)
        total_fields = sum(len(record.field_feedback) for record in self.feedback_records)
        
        # Calculate overall accuracy
        correct_fields = 0
        total_evaluated = 0
        
        for record in self.feedback_records:
            for field_feedback in record.field_feedback:
                total_evaluated += 1
                if field_feedback.feedback_status == FeedbackStatus.CORRECT:
                    correct_fields += 1
        
        overall_accuracy = correct_fields / total_evaluated if total_evaluated > 0 else 0.0
        
        # Get most problematic fields
        field_aggregations = self.get_feedback_aggregation()
        problematic_fields = [
            {
                "field_name": agg.field_name,
                "error_rate": 1 - agg.accuracy_rate,
                "total_feedback": agg.total_feedback
            }
            for agg in field_aggregations
            if agg.total_feedback >= 5  # Minimum sample size
        ]
        
        # Sort by error rate
        problematic_fields.sort(key=lambda x: x["error_rate"], reverse=True)
        
        # Get trends and active alerts
        trends = self.get_feedback_trends()
        active_alerts = self.get_active_alerts()
        recent_optimizations = self.get_optimization_recommendations()
        
        return FeedbackStatistics(
            total_feedback_records=total_records,
            total_fields_evaluated=total_fields,
            overall_accuracy_rate=overall_accuracy,
            most_problematic_fields=problematic_fields[:10],  # Top 10
            feedback_trends=trends,
            active_alerts=active_alerts,
            recent_optimizations=recent_optimizations
        )
    
    def _filter_feedback_records(self,
                                field_name: Optional[str] = None,
                                prompt_version: Optional[str] = None,
                                document_type: Optional[str] = None,
                                time_period: Optional[str] = None) -> List[UserFeedbackRecord]:
        """Filter feedback records based on criteria."""
        filtered = self.feedback_records
        
        if prompt_version:
            filtered = [r for r in filtered if r.prompt_version == prompt_version]
        
        if document_type:
            filtered = [r for r in filtered if r.document_type == document_type]
        
        if time_period:
            days = int(time_period[:-1]) if time_period.endswith('d') else 7
            cutoff_date = datetime.now() - timedelta(days=days)
            filtered = [r for r in filtered if r.timestamp >= cutoff_date]
        
        return filtered
    
    def _check_for_alerts(self, feedback_record: UserFeedbackRecord):
        """Check for potential alerts based on new feedback."""
        # Check for high error rate in recent feedback
        recent_records = self._filter_feedback_records(time_period="1d")
        
        for field_feedback in feedback_record.field_feedback:
            field_name = field_feedback.field_name
            
            # Get recent feedback for this field
            field_feedback_recent = []
            for record in recent_records:
                for ff in record.field_feedback:
                    if ff.field_name == field_name:
                        field_feedback_recent.append(ff)
            
            if len(field_feedback_recent) >= 5:  # Minimum sample size
                error_rate = sum(1 for ff in field_feedback_recent 
                               if ff.feedback_status == FeedbackStatus.INCORRECT) / len(field_feedback_recent)
                
                if error_rate > self.alert_thresholds["high_error_rate"]:
                    # Check if alert already exists
                    existing_alert = next(
                        (alert for alert in self.alerts 
                         if alert.field_name == field_name and 
                         alert.alert_type == "high_error_rate" and 
                         alert.status == "active"),
                        None
                    )
                    
                    if not existing_alert:
                        alert = FeedbackAlert(
                            alert_id=str(uuid.uuid4()),
                            field_name=field_name,
                            alert_type="high_error_rate",
                            severity="high" if error_rate > 0.5 else "medium",
                            description=f"High error rate ({error_rate:.1%}) for field '{field_name}'",
                            threshold_value=self.alert_thresholds["high_error_rate"],
                            current_value=error_rate,
                            prompt_version=feedback_record.prompt_version
                        )
                        self.alerts.append(alert)
                        logger.warning(f"Alert created: {alert.description}")
    
    def _generate_optimization_recommendations(self):
        """Generate optimization recommendations based on feedback patterns."""
        # Get field aggregations
        field_aggregations = self.get_feedback_aggregation()
        
        for aggregation in field_aggregations:
            if (aggregation.total_feedback >= 10 and 
                aggregation.accuracy_rate < 0.8):  # Significant issues
                
                # Check if recommendation already exists
                existing_rec = next(
                    (rec for rec in self.optimization_recommendations 
                     if rec.field_name == aggregation.field_name and 
                     rec.status == "active"),
                    None
                )
                
                if not existing_rec:
                    # Generate recommendation based on common reasons
                    common_reasons = [r["reason"] for r in aggregation.common_reasons[:3]]
                    
                    if FeedbackReason.PROMPT_AMBIGUOUS in common_reasons:
                        recommendation = FeedbackOptimizationRecommendation(
                            recommendation_id=str(uuid.uuid4()),
                            field_name=aggregation.field_name,
                            recommendation_type="prompt_clarification",
                            description=f"Clarify prompt for field '{aggregation.field_name}' due to ambiguity",
                            suggested_actions=[
                                "Add more specific field definitions",
                                "Include validation examples",
                                "Specify expected format more clearly"
                            ],
                            feedback_evidence=aggregation.sample_comments,
                            expected_impact=0.2,
                            priority="high" if aggregation.accuracy_rate < 0.6 else "medium",
                            affected_prompt_versions=aggregation.prompt_versions
                        )
                        self.optimization_recommendations.append(recommendation)
                    
                    elif FeedbackReason.WRONG_FORMAT in common_reasons:
                        recommendation = FeedbackOptimizationRecommendation(
                            recommendation_id=str(uuid.uuid4()),
                            field_name=aggregation.field_name,
                            recommendation_type="format_standardization",
                            description=f"Standardize format for field '{aggregation.field_name}'",
                            suggested_actions=[
                                "Add format validation rules",
                                "Include format examples in prompt",
                                "Implement post-processing normalization"
                            ],
                            feedback_evidence=aggregation.sample_comments,
                            expected_impact=0.15,
                            priority="medium",
                            affected_prompt_versions=aggregation.prompt_versions
                        )
                        self.optimization_recommendations.append(recommendation)
    
    def reset_feedback(self):
        """Reset all feedback data (for testing purposes)."""
        self.feedback_records.clear()
        self.alerts.clear()
        self.optimization_recommendations.clear()
        logger.info("Feedback data reset") 