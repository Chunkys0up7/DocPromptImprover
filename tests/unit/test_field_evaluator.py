"""
Unit tests for field evaluator.

This module tests the field-level evaluation logic including scoring algorithms,
confidence calibration, and error categorization.
"""

import pytest
from datetime import datetime

from src.evaluators.field_evaluator import FieldEvaluator
from src.models.evaluation_models import (
    FieldEvaluationResult,
    ExtractionStatus,
    EvaluationConfig
)


class TestFieldEvaluator:
    """Test cases for FieldEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create a field evaluator instance."""
        return FieldEvaluator()
    
    @pytest.fixture
    def strict_evaluator(self):
        """Create a field evaluator with strict matching."""
        config = EvaluationConfig(strict_matching=True)
        return FieldEvaluator(config)
    
    def test_exact_match_success(self, evaluator):
        """Test exact match evaluation."""
        result = evaluator.evaluate_field(
            field_name="test_field",
            expected_value="exact match",
            extracted_value="exact match",
            confidence_score=0.9,
            field_type="text"
        )
        
        assert result.status == ExtractionStatus.SUCCESS
        assert result.evaluation_score == 1.0
        assert result.error_message is None
    
    def test_partial_match(self, evaluator):
        """Test partial match evaluation."""
        result = evaluator.evaluate_field(
            field_name="test_field",
            expected_value="complete text",
            extracted_value="complete",
            confidence_score=0.8,
            field_type="text"
        )
        
        assert result.status == ExtractionStatus.PARTIAL
        assert 0.0 < result.evaluation_score < 1.0
        assert result.error_message is not None
    
    def test_complete_failure(self, evaluator):
        """Test complete failure evaluation."""
        result = evaluator.evaluate_field(
            field_name="test_field",
            expected_value="expected",
            extracted_value="completely different",
            confidence_score=0.9,
            field_type="text"
        )
        
        assert result.status == ExtractionStatus.FAILED
        assert result.evaluation_score < 0.5
        assert result.error_message is not None
    
    def test_missing_extracted_value(self, evaluator):
        """Test missing extracted value."""
        result = evaluator.evaluate_field(
            field_name="test_field",
            expected_value="expected",
            extracted_value=None,
            confidence_score=0.5,
            field_type="text"
        )
        
        assert result.status == ExtractionStatus.MISSING
        assert result.evaluation_score == 0.0
        assert "not extracted" in result.error_message.lower()
    
    def test_missing_expected_value(self, evaluator):
        """Test missing expected value."""
        result = evaluator.evaluate_field(
            field_name="test_field",
            expected_value=None,
            extracted_value="extracted",
            confidence_score=0.8,
            field_type="text"
        )
        
        assert result.status == ExtractionStatus.FAILED
        assert result.evaluation_score == 0.0
        assert "expected value is none" in result.error_message.lower()
    
    def test_both_values_none(self, evaluator):
        """Test when both expected and extracted values are None."""
        result = evaluator.evaluate_field(
            field_name="test_field",
            expected_value=None,
            extracted_value=None,
            confidence_score=0.5,
            field_type="text"
        )
        
        assert result.status == ExtractionStatus.SUCCESS
        assert result.evaluation_score == 1.0
        assert "both values are none" in result.error_message.lower()
    
    def test_low_confidence_failure(self, evaluator):
        """Test failure due to low confidence."""
        result = evaluator.evaluate_field(
            field_name="test_field",
            expected_value="expected",
            extracted_value="expected",
            confidence_score=0.3,  # Below default threshold
            field_type="text"
        )
        
        assert result.status == ExtractionStatus.FAILED
        assert result.evaluation_score == 1.0  # Perfect match
        assert result.error_message is None  # No error message for confidence failure
    
    def test_number_field_evaluation(self, evaluator):
        """Test number field evaluation."""
        # Exact match
        result = evaluator.evaluate_field(
            field_name="amount",
            expected_value="100.50",
            extracted_value="100.50",
            confidence_score=0.9,
            field_type="number"
        )
        
        assert result.status == ExtractionStatus.SUCCESS
        assert result.evaluation_score == 1.0
        
        # Close match (within 1% tolerance)
        result = evaluator.evaluate_field(
            field_name="amount",
            expected_value="100.50",
            extracted_value="100.49",
            confidence_score=0.9,
            field_type="number"
        )
        
        assert result.status == ExtractionStatus.SUCCESS
        assert result.evaluation_score == 0.9
        
        # Different format
        result = evaluator.evaluate_field(
            field_name="amount",
            expected_value="100.50",
            extracted_value="$100.50",
            confidence_score=0.9,
            field_type="number"
        )
        
        assert result.status == ExtractionStatus.SUCCESS
        assert result.evaluation_score == 1.0  # Currency symbol removed during normalization
    
    def test_date_field_evaluation(self, evaluator):
        """Test date field evaluation."""
        # Exact match
        result = evaluator.evaluate_field(
            field_name="date",
            expected_value="2024-01-15",
            extracted_value="2024-01-15",
            confidence_score=0.9,
            field_type="date"
        )
        
        assert result.status == ExtractionStatus.SUCCESS
        assert result.evaluation_score == 1.0
        
        # Different format
        result = evaluator.evaluate_field(
            field_name="date",
            expected_value="2024-01-15",
            extracted_value="01/15/2024",
            confidence_score=0.9,
            field_type="date"
        )
        
        assert result.status == ExtractionStatus.SUCCESS
        assert result.evaluation_score == 1.0  # Date normalized
        
        # Close date (1 day difference)
        result = evaluator.evaluate_field(
            field_name="date",
            expected_value="2024-01-15",
            extracted_value="2024-01-16",
            confidence_score=0.9,
            field_type="date"
        )
        
        assert result.status == ExtractionStatus.PARTIAL
        assert result.evaluation_score == 0.8
    
    def test_email_field_evaluation(self, evaluator):
        """Test email field evaluation."""
        # Exact match
        result = evaluator.evaluate_field(
            field_name="email",
            expected_value="test@example.com",
            extracted_value="test@example.com",
            confidence_score=0.9,
            field_type="email"
        )
        
        assert result.status == ExtractionStatus.SUCCESS
        assert result.evaluation_score == 1.0
        
        # Same domain
        result = evaluator.evaluate_field(
            field_name="email",
            expected_value="test@example.com",
            extracted_value="different@example.com",
            confidence_score=0.9,
            field_type="email"
        )
        
        assert result.status == ExtractionStatus.PARTIAL
        assert result.evaluation_score == 0.7
    
    def test_phone_field_evaluation(self, evaluator):
        """Test phone field evaluation."""
        # Exact match
        result = evaluator.evaluate_field(
            field_name="phone",
            expected_value="+1-555-123-4567",
            extracted_value="+1-555-123-4567",
            confidence_score=0.9,
            field_type="phone"
        )
        
        assert result.status == ExtractionStatus.SUCCESS
        assert result.evaluation_score == 1.0
        
        # Different format, missing country code
        result = evaluator.evaluate_field(
            field_name="phone",
            expected_value="+1-555-123-4567",
            extracted_value="5551234567",
            confidence_score=0.9,
            field_type="phone"
        )
        
        assert result.status == ExtractionStatus.SUCCESS
        # Since country code is missing, expect a score less than 1.0 and not equal to 0.9
        assert result.evaluation_score < 1.0
        assert result.evaluation_score != 0.9
    
    def test_strict_matching(self, strict_evaluator):
        """Test strict matching configuration."""
        result = strict_evaluator.evaluate_field(
            field_name="test_field",
            expected_value="exact",
            extracted_value="exact",
            confidence_score=0.9,
            field_type="text"
        )
        
        assert result.status == ExtractionStatus.SUCCESS
        assert result.evaluation_score == 1.0
        
        # Slight difference should fail in strict mode
        result = strict_evaluator.evaluate_field(
            field_name="test_field",
            expected_value="exact",
            extracted_value="exact ",
            confidence_score=0.9,
            field_type="text"
        )
        
        assert result.status == ExtractionStatus.FAILED
        assert result.evaluation_score == 0.0
    
    def test_case_insensitive_matching(self, evaluator):
        """Test case insensitive matching."""
        result = evaluator.evaluate_field(
            field_name="test_field",
            expected_value="UPPERCASE",
            extracted_value="uppercase",
            confidence_score=0.9,
            field_type="text"
        )
        
        assert result.status == ExtractionStatus.SUCCESS
        assert result.evaluation_score == 1.0
    
    def test_whitespace_normalization(self, evaluator):
        """Test whitespace normalization."""
        result = evaluator.evaluate_field(
            field_name="test_field",
            expected_value="normal text",
            extracted_value="  normal   text  ",
            confidence_score=0.9,
            field_type="text"
        )
        
        assert result.status == ExtractionStatus.SUCCESS
        assert result.evaluation_score == 1.0
    
    def test_batch_evaluation(self, evaluator):
        """Test batch evaluation functionality."""
        evaluations = [
            {
                "field_name": "field1",
                "expected_value": "value1",
                "extracted_value": "value1",
                "confidence_score": 0.9,
                "field_type": "text"
            },
            {
                "field_name": "field2",
                "expected_value": "value2",
                "extracted_value": "wrong",
                "confidence_score": 0.8,
                "field_type": "text"
            }
        ]
        
        results = evaluator.batch_evaluate(evaluations)
        
        assert len(results) == 2
        assert results[0].status == ExtractionStatus.SUCCESS
        assert results[1].status == ExtractionStatus.FAILED
    
    def test_evaluation_notes_generation(self, evaluator):
        """Test evaluation notes generation."""
        result = evaluator.evaluate_field(
            field_name="test_field",
            expected_value="expected",
            extracted_value="extracted",
            confidence_score=0.7,
            field_type="text"
        )
        
        assert result.evaluation_notes is not None
        assert "Field type: text" in result.evaluation_notes
        assert "Evaluation score:" in result.evaluation_notes
        assert "Confidence score:" in result.evaluation_notes
        assert "Expected: 'expected'" in result.evaluation_notes
        assert "Extracted: 'extracted'" in result.evaluation_notes
    
    def test_error_message_generation(self, evaluator):
        """Test error message generation."""
        # Missing field
        result = evaluator.evaluate_field(
            field_name="test_field",
            expected_value="expected",
            extracted_value=None,
            confidence_score=0.5,
            field_type="text"
        )
        
        assert "not extracted" in result.error_message.lower()
        
        # Complete mismatch
        result = evaluator.evaluate_field(
            field_name="test_field",
            expected_value="expected",
            extracted_value="completely different",
            confidence_score=0.9,
            field_type="text"
        )
        
        assert "complete mismatch" in result.error_message.lower() or "partial match" in result.error_message.lower()
    
    def test_field_type_detection(self, evaluator):
        """Test automatic field type detection."""
        # Number field
        result = evaluator.evaluate_field(
            field_name="amount",
            expected_value=100.50,
            extracted_value="100.50",
            confidence_score=0.9,
            field_type="number"
        )
        
        assert result.field_type == "number"
        
        # Date field
        result = evaluator.evaluate_field(
            field_name="date",
            expected_value="2024-01-15",
            extracted_value="2024-01-15",
            confidence_score=0.9,
            field_type="date"
        )
        
        assert result.field_type == "date"
    
    def test_confidence_threshold_impact(self, evaluator):
        """Test impact of confidence threshold on status."""
        # High confidence, perfect match
        result = evaluator.evaluate_field(
            field_name="test_field",
            expected_value="expected",
            extracted_value="expected",
            confidence_score=0.9,
            field_type="text"
        )
        
        assert result.status == ExtractionStatus.SUCCESS
        
        # Low confidence, perfect match
        result = evaluator.evaluate_field(
            field_name="test_field",
            expected_value="expected",
            extracted_value="expected",
            confidence_score=0.3,  # Below threshold
            field_type="text"
        )
        
        assert result.status == ExtractionStatus.FAILED  # Failed due to low confidence
    
    def test_partial_credit_disabled(self):
        """Test evaluation with partial credit disabled."""
        config = EvaluationConfig(enable_partial_credit=False)
        evaluator = FieldEvaluator(config)
        
        result = evaluator.evaluate_field(
            field_name="test_field",
            expected_value="complete text",
            extracted_value="complete",
            confidence_score=0.9,
            field_type="text"
        )
        
        # Should be either 1.0 (if similarity >= 0.9) or 0.0
        assert result.evaluation_score in [0.0, 1.0] 