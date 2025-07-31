"""
Field-level evaluation logic for document extraction evaluation.

This module implements the core evaluation algorithms for individual fields,
including exact match scoring, partial credit, confidence calibration, and error categorization.
"""

import re
import difflib
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
import statistics

from ..models.evaluation_models import (
    FieldEvaluationResult,
    ExtractionStatus,
    EvaluationConfig
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class FieldEvaluator:
    """
    Evaluates individual field extractions with various scoring algorithms.
    
    This class implements field-level evaluation logic including exact matching,
    partial credit scoring, confidence calibration, and error categorization.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize the field evaluator with configuration."""
        self.config = config or EvaluationConfig()
        logger.info("FieldEvaluator initialized with configuration")
    
    def evaluate_field(self,
                      field_name: str,
                      expected_value: Optional[str],
                      extracted_value: Optional[str],
                      confidence_score: float = 0.0,
                      field_type: str = "text") -> FieldEvaluationResult:
        """
        Evaluate a single field extraction.
        
        Args:
            field_name: Name of the field being evaluated
            expected_value: Ground truth value
            extracted_value: Value extracted by the model
            confidence_score: Confidence score from the extraction model
            field_type: Expected data type of the field
            
        Returns:
            FieldEvaluationResult: Complete evaluation result for the field
        """
        
        # Convert values to strings for consistency
        expected_str = str(expected_value) if expected_value is not None else None
        extracted_str = str(extracted_value) if extracted_value is not None else None
        
        # Handle missing values
        if expected_str is None and extracted_str is None:
            return self._create_result(
                field_name, expected_str, extracted_str, confidence_score,
                field_type, 1.0, ExtractionStatus.SUCCESS, "Both values are None"
            )
        
        if expected_str is None:
            return self._create_result(
                field_name, expected_str, extracted_str, confidence_score,
                field_type, 0.0, ExtractionStatus.FAILED, "Expected value is None"
            )
        
        if extracted_str is None:
            return self._create_result(
                field_name, expected_str, extracted_str, confidence_score,
                field_type, 0.0, ExtractionStatus.MISSING, "Field was not extracted"
            )
        
        # Normalize values for comparison
        expected_norm = self._normalize_value(expected_str, field_type)
        extracted_norm = self._normalize_value(extracted_str, field_type)
        
        # Calculate evaluation score
        evaluation_score = self._calculate_score(expected_norm, extracted_norm, field_type)
        
        # Determine status
        status = self._determine_status(evaluation_score, confidence_score)
        
        # Generate error message if needed
        error_message = self._generate_error_message(
            expected_norm, extracted_norm, evaluation_score, status, confidence_score, self.config
        )
        
        # Create evaluation notes
        evaluation_notes = self._generate_evaluation_notes(
            expected_norm, extracted_norm, evaluation_score, confidence_score, field_type
        )
        
        return self._create_result(
            field_name, expected_str, extracted_str, confidence_score,
            field_type, evaluation_score, status, error_message, evaluation_notes
        )
    
    def _normalize_value(self, value: str, field_type: str) -> str:
        """
        Normalize a value based on field type and configuration.
        
        Args:
            value: The value to normalize
            field_type: Type of field (text, number, date, etc.)
            
        Returns:
            str: Normalized value
        """
        
        if self.config.strict_matching:
            return value  # No normalization in strict mode
        
        if not value:
            return ""
        
        # Convert to string if needed
        value_str = str(value).strip()
        
        # Apply normalization based on configuration
        if not self.config.case_sensitive:
            value_str = value_str.lower()
        
        if self.config.normalize_whitespace:
            # Normalize whitespace
            value_str = re.sub(r'\s+', ' ', value_str).strip()
        
        # Field-type specific normalization
        if field_type == "number":
            # Remove currency symbols and normalize decimal separators
            value_str = re.sub(r'[^\d.,]', '', value_str)
            value_str = value_str.replace(',', '')
        
        elif field_type == "date":
            # Standardize date formats
            value_str = self._normalize_date(value_str)
        
        elif field_type == "email":
            # Normalize email addresses
            value_str = value_str.lower().strip()
        
        elif field_type == "phone":
            # Normalize phone numbers - remove all non-digits for consistent comparison
            value_str = re.sub(r'[^\d]', '', value_str)
        
        return value_str
    
    def _normalize_date(self, date_str: str) -> str:
        """
        Normalize date strings to a standard format.
        
        Args:
            date_str: Date string to normalize
            
        Returns:
            str: Normalized date string
        """
        
        # Common date patterns - assume MM/DD/YYYY for slash format
        patterns = [
            # MM/DD/YYYY
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\3-\1-\2'),
            # MM-DD-YYYY
            (r'(\d{1,2})-(\d{1,2})-(\d{4})', r'\3-\1-\2'),
            # YYYY-MM-DD (already standard)
            (r'(\d{4})-(\d{1,2})-(\d{1,2})', r'\1-\2-\3'),
        ]
        
        for pattern, replacement in patterns:
            if re.match(pattern, date_str):
                return re.sub(pattern, replacement, date_str)
        
        return date_str
    
    def _calculate_score(self, expected: str, extracted: str, field_type: str) -> float:
        """
        Calculate evaluation score between expected and extracted values.
        
        Args:
            expected: Normalized expected value
            extracted: Normalized extracted value
            field_type: Type of field
            
        Returns:
            float: Score between 0.0 and 1.0
        """
        
        # Exact match
        if expected == extracted:
            return 1.0
        
        # Empty values
        if not expected and not extracted:
            return 1.0
        if not expected or not extracted:
            return 0.0
        
        # Field-type specific scoring
        if field_type == "number":
            return self._score_numbers(expected, extracted)
        elif field_type == "date":
            return self._score_dates(expected, extracted)
        elif field_type == "email":
            return self._score_emails(expected, extracted)
        elif field_type == "phone":
            return self._score_phones(expected, extracted)
        else:
            return self._score_text(expected, extracted)
    
    def _score_text(self, expected: str, extracted: str) -> float:
        """Score text fields using string similarity."""
        
        if self.config.strict_matching:
            return 1.0 if expected == extracted else 0.0
        
        # Use difflib for string similarity
        similarity = difflib.SequenceMatcher(None, expected, extracted).ratio()
        
        # Apply partial credit if enabled
        if self.config.enable_partial_credit:
            return similarity
        else:
            return 1.0 if similarity >= 0.9 else 0.0
    
    def _score_numbers(self, expected: str, extracted: str) -> float:
        """Score numeric fields."""
        
        try:
            expected_num = float(expected)
            extracted_num = float(extracted)
            
            # Exact match
            if expected_num == extracted_num:
                return 1.0
            
            # Relative difference
            if expected_num != 0:
                relative_diff = abs(expected_num - extracted_num) / abs(expected_num)
                if relative_diff <= 0.01:  # 1% tolerance
                    return 0.9
                elif relative_diff <= 0.05:  # 5% tolerance
                    return 0.7
                elif relative_diff <= 0.1:  # 10% tolerance
                    return 0.5
            
            return 0.0
            
        except (ValueError, TypeError):
            # Fall back to text similarity
            return self._score_text(expected, extracted)
    
    def _score_dates(self, expected: str, extracted: str) -> float:
        """Score date fields."""
        
        try:
            # Try to parse dates
            expected_date = datetime.strptime(expected, "%Y-%m-%d")
            extracted_date = datetime.strptime(extracted, "%Y-%m-%d")
            
            # Exact match
            if expected_date == extracted_date:
                return 1.0
            
            # Day difference
            day_diff = abs((expected_date - extracted_date).days)
            if day_diff == 1:
                return 0.8
            elif day_diff <= 7:
                return 0.6
            elif day_diff <= 30:
                return 0.3
            
            return 0.0
            
        except ValueError:
            # Fall back to text similarity
            return self._score_text(expected, extracted)
    
    def _score_emails(self, expected: str, extracted: str) -> float:
        """Score email fields."""
        
        # Exact match
        if expected == extracted:
            return 1.0
        
        # Check if domains match
        expected_parts = expected.split('@')
        extracted_parts = extracted.split('@')
        
        if len(expected_parts) == 2 and len(extracted_parts) == 2:
            if expected_parts[1] == extracted_parts[1]:  # Same domain
                return 0.7
        
        return self._score_text(expected, extracted)
    
    def _score_phones(self, expected: str, extracted: str) -> float:
        """Score phone number fields."""
        
        # Exact match
        if expected == extracted:
            return 1.0
        
        # Since phone normalization already removes non-digits,
        # we can compare directly
        if expected == extracted:
            return 0.9
        
        # If they're not the same, fall back to text similarity
        return self._score_text(expected, extracted)
    
    def _determine_status(self, evaluation_score: float, confidence_score: float) -> ExtractionStatus:
        """
        Determine the extraction status based on evaluation score and confidence.
        
        Args:
            evaluation_score: Score from evaluation algorithm
            confidence_score: Confidence score from the model
            
        Returns:
            ExtractionStatus: Determined status
        """
        
        # Check confidence threshold
        if confidence_score < self.config.confidence_threshold:
            return ExtractionStatus.FAILED
        
        # If score is 0.0, always failed
        if evaluation_score == 0.0:
            return ExtractionStatus.FAILED
        
        # Determine status based on evaluation score
        if evaluation_score > self.config.success_threshold:
            return ExtractionStatus.SUCCESS
        elif evaluation_score >= self.config.partial_threshold:
            return ExtractionStatus.PARTIAL
        else:
            return ExtractionStatus.FAILED
    
    def _generate_error_message(self,
                               expected: str,
                               extracted: str,
                               evaluation_score: float,
                               status: ExtractionStatus,
                               confidence_score: float = None,
                               config = None) -> Optional[str]:
        """Generate error message for failed extractions."""
        
        if status == ExtractionStatus.SUCCESS:
            return None
        
        if status == ExtractionStatus.MISSING:
            return "Field was not extracted"
        
        if status == ExtractionStatus.FAILED:
            # If the score is perfect but confidence is low, do not generate an error message
            if evaluation_score == 1.0 and confidence_score is not None and config is not None:
                if confidence_score < config.confidence_threshold:
                    return None
            if evaluation_score == 0.0:
                return "Complete mismatch between expected and extracted values"
            else:
                return f"Partial match with score {evaluation_score:.2f}"
        
        if status == ExtractionStatus.PARTIAL:
            return f"Partial match with score {evaluation_score:.2f}"
        
        return None
    
    def _generate_evaluation_notes(self,
                                  expected: str,
                                  extracted: str,
                                  evaluation_score: float,
                                  confidence_score: float,
                                  field_type: str) -> str:
        """Generate detailed evaluation notes."""
        
        notes = []
        
        # Add field type information
        notes.append(f"Field type: {field_type}")
        
        # Add score information
        notes.append(f"Evaluation score: {evaluation_score:.3f}")
        notes.append(f"Confidence score: {confidence_score:.3f}")
        
        # Add comparison details
        if expected != extracted:
            notes.append(f"Expected: '{expected}'")
            notes.append(f"Extracted: '{extracted}'")
            
            # Add similarity information for text fields
            if field_type == "text":
                similarity = difflib.SequenceMatcher(None, expected, extracted).ratio()
                notes.append(f"String similarity: {similarity:.3f}")
        
        return "; ".join(notes)
    
    def _create_result(self,
                      field_name: str,
                      expected_value: Optional[str],
                      extracted_value: Optional[str],
                      confidence_score: float,
                      field_type: str,
                      evaluation_score: float,
                      status: ExtractionStatus,
                      error_message: Optional[str] = None,
                      evaluation_notes: Optional[str] = None) -> FieldEvaluationResult:
        """Create a FieldEvaluationResult object."""
        
        return FieldEvaluationResult(
            field_name=field_name,
            expected_value=expected_value,
            extracted_value=extracted_value,
            confidence_score=confidence_score,
            status=status,
            evaluation_score=evaluation_score,
            error_message=error_message,
            evaluation_notes=evaluation_notes,
            field_type=field_type
        )
    
    def batch_evaluate(self, evaluations: List[Dict[str, Any]]) -> List[FieldEvaluationResult]:
        """
        Evaluate multiple fields in batch.
        
        Args:
            evaluations: List of evaluation dictionaries
            
        Returns:
            List[FieldEvaluationResult]: List of evaluation results
        """
        
        results = []
        for eval_data in evaluations:
            result = self.evaluate_field(
                field_name=eval_data["field_name"],
                expected_value=eval_data.get("expected_value"),
                extracted_value=eval_data.get("extracted_value"),
                confidence_score=eval_data.get("confidence_score", 0.0),
                field_type=eval_data.get("field_type", "text")
            )
            results.append(result)
        
        return results 