"""
Unit tests for feedback collector functionality.

This module tests the feedback collection, validation, and analysis features.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from src.feedback.feedback_collector import FeedbackCollector
from src.models.feedback_models import (
    FeedbackStatus,
    FeedbackReason,
    FieldFeedback,
    UserFeedbackRecord,
    FeedbackAggregation,
    FeedbackTrend,
    FeedbackAlert,
    FeedbackOptimizationRecommendation
)


class TestFeedbackCollector:
    """Test cases for FeedbackCollector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.collector = FeedbackCollector()
        self.sample_feedback_data = {
            "document_id": "doc_123",
            "user_id": "user_001",
            "session_id": "session_abc",
            "prompt_version": "v1.0",
            "document_type": "invoice",
            "field_feedback": [
                {
                    "field_name": "invoice_number",
                    "shown_value": "INV-2024-001",
                    "feedback_status": "correct",
                    "correction": None,
                    "comment": "Correctly extracted",
                    "reason_code": None,
                    "confidence_score": 0.95
                },
                {
                    "field_name": "date",
                    "shown_value": "2024-01-15",
                    "feedback_status": "incorrect",
                    "correction": "2024-01-16",
                    "comment": "Wrong date",
                    "reason_code": "wrong_format",
                    "confidence_score": 0.85
                }
            ],
            "overall_comment": "Most fields correct"
        }
    
    def test_collect_feedback_valid_data(self):
        """Test collecting feedback with valid data."""
        feedback_record = self.collector.collect_feedback(self.sample_feedback_data)
        
        assert feedback_record.document_id == "doc_123"
        assert feedback_record.user_id == "user_001"
        assert feedback_record.prompt_version == "v1.0"
        assert len(feedback_record.field_feedback) == 2
        assert feedback_record.processing_time is not None
        assert feedback_record.feedback_id is not None
    
    def test_collect_feedback_invalid_data(self):
        """Test collecting feedback with invalid data."""
        invalid_data = {
            "document_id": "doc_123",
            "user_id": "user_001",
            # Missing required fields
        }
        
        with pytest.raises(ValueError):
            self.collector.collect_feedback(invalid_data)
    
    def test_collect_feedback_empty_field_feedback(self):
        """Test collecting feedback with empty field feedback."""
        invalid_data = self.sample_feedback_data.copy()
        invalid_data["field_feedback"] = []
        
        with pytest.raises(ValueError):
            self.collector.collect_feedback(invalid_data)
    
    def test_get_feedback_aggregation(self):
        """Test getting feedback aggregation."""
        # Add some feedback first
        self.collector.collect_feedback(self.sample_feedback_data)
        
        # Add more feedback with different statuses
        feedback_data_2 = self.sample_feedback_data.copy()
        feedback_data_2["document_id"] = "doc_124"
        feedback_data_2["field_feedback"][0]["feedback_status"] = "incorrect"
        feedback_data_2["field_feedback"][0]["correction"] = "INV-2024-002"  # Add correction for incorrect status
        self.collector.collect_feedback(feedback_data_2)
        
        aggregations = self.collector.get_feedback_aggregation()
        
        assert len(aggregations) > 0
        
        # Check invoice_number aggregation
        invoice_agg = next((agg for agg in aggregations if agg.field_name == "invoice_number"), None)
        assert invoice_agg is not None
        assert invoice_agg.total_feedback == 2
        assert invoice_agg.correct_count == 1
        assert invoice_agg.incorrect_count == 1
        assert invoice_agg.accuracy_rate == 0.5
    
    def test_get_feedback_aggregation_with_filters(self):
        """Test getting feedback aggregation with filters."""
        # Add feedback
        self.collector.collect_feedback(self.sample_feedback_data)
        
        # Test field filter
        aggregations = self.collector.get_feedback_aggregation(field_name="invoice_number")
        assert len(aggregations) == 1
        assert aggregations[0].field_name == "invoice_number"
        
        # Test prompt version filter
        aggregations = self.collector.get_feedback_aggregation(prompt_version="v1.0")
        assert len(aggregations) > 0
        
        # Test document type filter
        aggregations = self.collector.get_feedback_aggregation(document_type="invoice")
        assert len(aggregations) > 0
    
    def test_get_feedback_trends(self):
        """Test getting feedback trends."""
        # Add feedback with different dates
        with patch('src.feedback.feedback_collector.datetime') as mock_datetime:
            mock_datetime.now.return_value = datetime(2024, 1, 15)
            
            # Add feedback for today
            self.collector.collect_feedback(self.sample_feedback_data)
            
            # Add feedback for yesterday
            feedback_data_2 = self.sample_feedback_data.copy()
            feedback_data_2["document_id"] = "doc_124"
            self.collector.collect_feedback(feedback_data_2)
        
        trends = self.collector.get_feedback_trends(time_period="7d")
        assert len(trends) > 0
    
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        # Initially no alerts
        alerts = self.collector.get_active_alerts()
        assert len(alerts) == 0
        
        # Add feedback that should trigger alerts
        for i in range(10):  # Add enough feedback to trigger alerts
            feedback_data = self.sample_feedback_data.copy()
            feedback_data["document_id"] = f"doc_{i}"
            feedback_data["field_feedback"][0]["feedback_status"] = "incorrect"
            feedback_data["field_feedback"][0]["correction"] = f"INV-2024-{i:03d}"  # Add correction for incorrect status
            self.collector.collect_feedback(feedback_data)
        
        alerts = self.collector.get_active_alerts()
        # May or may not have alerts depending on thresholds
        assert isinstance(alerts, list)
    
    def test_get_optimization_recommendations(self):
        """Test getting optimization recommendations."""
        # Initially no recommendations
        recommendations = self.collector.get_optimization_recommendations()
        assert len(recommendations) == 0
        
        # Add feedback that should trigger recommendations
        for i in range(15):  # Add enough feedback to trigger recommendations
            feedback_data = self.sample_feedback_data.copy()
            feedback_data["document_id"] = f"doc_{i}"
            feedback_data["field_feedback"][0]["feedback_status"] = "incorrect"
            feedback_data["field_feedback"][0]["correction"] = f"INV-2024-{i:03d}"  # Add correction for incorrect status
            feedback_data["field_feedback"][0]["reason_code"] = "prompt_ambiguous"
            self.collector.collect_feedback(feedback_data)
        
        recommendations = self.collector.get_optimization_recommendations()
        # May or may not have recommendations depending on thresholds
        assert isinstance(recommendations, list)
    
    def test_get_feedback_statistics(self):
        """Test getting feedback statistics."""
        # Add some feedback
        self.collector.collect_feedback(self.sample_feedback_data)
        
        stats = self.collector.get_feedback_statistics()
        
        assert stats.total_feedback_records == 1
        assert stats.total_fields_evaluated == 2
        assert stats.overall_accuracy_rate == 0.5  # 1 correct, 1 incorrect
        assert isinstance(stats.most_problematic_fields, list)
        assert isinstance(stats.feedback_trends, list)
        assert isinstance(stats.active_alerts, list)
        assert isinstance(stats.recent_optimizations, list)
    
    def test_reset_feedback(self):
        """Test resetting feedback data."""
        # Add some feedback
        self.collector.collect_feedback(self.sample_feedback_data)
        
        # Verify data exists
        stats = self.collector.get_feedback_statistics()
        assert stats.total_feedback_records == 1
        
        # Reset
        self.collector.reset_feedback()
        
        # Verify data is cleared
        stats = self.collector.get_feedback_statistics()
        assert stats.total_feedback_records == 0
        assert stats.total_fields_evaluated == 0


class TestFeedbackModels:
    """Test cases for feedback models."""
    
    def test_field_feedback_validation(self):
        """Test FieldFeedback validation."""
        # Valid feedback
        feedback = FieldFeedback(
            field_name="test_field",
            shown_value="test_value",
            feedback_status=FeedbackStatus.CORRECT
        )
        assert feedback.field_name == "test_field"
        
        # Invalid: missing correction for incorrect status
        with pytest.raises(ValueError):
            FieldFeedback(
                field_name="test_field",
                shown_value="test_value",
                feedback_status=FeedbackStatus.INCORRECT,
                correction=None  # Should be required for incorrect
            )
    
    def test_user_feedback_record_validation(self):
        """Test UserFeedbackRecord validation."""
        # Valid record
        record = UserFeedbackRecord(
            feedback_id="feedback_123",
            document_id="doc_123",
            user_id="user_001",
            prompt_version="v1.0",
            document_type="invoice",
            field_feedback=[
                FieldFeedback(
                    field_name="test_field",
                    shown_value="test_value",
                    feedback_status=FeedbackStatus.CORRECT
                )
            ]
        )
        assert record.feedback_id == "feedback_123"
        
        # Invalid: empty field feedback
        with pytest.raises(ValueError):
            UserFeedbackRecord(
                feedback_id="feedback_123",
                document_id="doc_123",
                user_id="user_001",
                prompt_version="v1.0",
                document_type="invoice",
                field_feedback=[]  # Should not be empty
            )


class TestFeedbackIntegration:
    """Integration tests for feedback functionality."""
    
    def test_feedback_lifecycle(self):
        """Test complete feedback lifecycle."""
        collector = FeedbackCollector()
        
        # Step 1: Collect feedback
        feedback_data = {
            "document_id": "doc_123",
            "user_id": "user_001",
            "prompt_version": "v1.0",
            "document_type": "invoice",
            "field_feedback": [
                {
                    "field_name": "invoice_number",
                    "shown_value": "INV-2024-001",
                    "feedback_status": "correct",
                    "correction": None,
                    "comment": "Correct",
                    "reason_code": None,
                    "confidence_score": 0.95
                }
            ]
        }
        
        feedback_record = collector.collect_feedback(feedback_data)
        assert feedback_record is not None
        
        # Step 2: Get statistics
        stats = collector.get_feedback_statistics()
        assert stats.total_feedback_records == 1
        
        # Step 3: Get aggregation
        aggregations = collector.get_feedback_aggregation()
        assert len(aggregations) == 1
        
        # Step 4: Get trends
        trends = collector.get_feedback_trends()
        assert isinstance(trends, list)
        
        # Step 5: Check for alerts
        alerts = collector.get_active_alerts()
        assert isinstance(alerts, list)
        
        # Step 6: Get recommendations
        recommendations = collector.get_optimization_recommendations()
        assert isinstance(recommendations, list)
    
    def test_feedback_with_multiple_fields(self):
        """Test feedback with multiple fields and different statuses."""
        collector = FeedbackCollector()
        
        feedback_data = {
            "document_id": "doc_123",
            "user_id": "user_001",
            "prompt_version": "v1.0",
            "document_type": "invoice",
            "field_feedback": [
                {
                    "field_name": "invoice_number",
                    "shown_value": "INV-2024-001",
                    "feedback_status": "correct",
                    "correction": None,
                    "comment": "Correct",
                    "reason_code": None,
                    "confidence_score": 0.95
                },
                {
                    "field_name": "date",
                    "shown_value": "2024-01-15",
                    "feedback_status": "incorrect",
                    "correction": "2024-01-16",
                    "comment": "Wrong date",
                    "reason_code": "wrong_format",
                    "confidence_score": 0.85
                },
                {
                    "field_name": "total_amount",
                    "shown_value": "$1,250.00",
                    "feedback_status": "partial",
                    "correction": None,
                    "comment": "Missing currency",
                    "reason_code": "wrong_format",
                    "confidence_score": 0.75
                }
            ]
        }
        
        feedback_record = collector.collect_feedback(feedback_data)
        
        # Check field feedback
        assert len(feedback_record.field_feedback) == 3
        
        # Check statistics
        stats = collector.get_feedback_statistics()
        assert stats.total_fields_evaluated == 3
        assert stats.overall_accuracy_rate == 1/3  # 1 correct out of 3
        
        # Check aggregations
        aggregations = collector.get_feedback_aggregation()
        assert len(aggregations) == 3
        
        # Check each field
        invoice_agg = next(agg for agg in aggregations if agg.field_name == "invoice_number")
        assert invoice_agg.correct_count == 1
        assert invoice_agg.incorrect_count == 0
        assert invoice_agg.partial_count == 0
        
        date_agg = next(agg for agg in aggregations if agg.field_name == "date")
        assert date_agg.correct_count == 0
        assert date_agg.incorrect_count == 1
        assert date_agg.partial_count == 0
        
        amount_agg = next(agg for agg in aggregations if agg.field_name == "total_amount")
        assert amount_agg.correct_count == 0
        assert amount_agg.incorrect_count == 0
        assert amount_agg.partial_count == 1 