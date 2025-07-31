"""
Pure statistical evaluation logic for document extraction assessment.

This module provides rule-based evaluation algorithms without any LLM dependencies.
It focuses on statistical analysis, pattern detection, and metrics calculation.
"""

import re
import statistics
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict, Counter

from ..models.evaluation_models import (
    FieldEvaluationResult,
    DocumentEvaluationResult,
    ExtractionStatus,
    EvaluationStatistics
)


class StatisticalEvaluator:
    """
    Pure statistical evaluator for document extraction results.
    
    This class provides rule-based evaluation algorithms without any LLM dependencies,
    focusing on statistical analysis and pattern detection.
    """
    
    def __init__(self):
        """Initialize the statistical evaluator."""
        self.evaluation_rules = self._initialize_evaluation_rules()
    
    def _initialize_evaluation_rules(self) -> Dict[str, Any]:
        """Initialize evaluation rules and patterns."""
        
        return {
            "field_patterns": {
                "date": {
                    "regex_patterns": [
                        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                        r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
                        r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
                    ],
                    "validation": self._validate_date,
                    "normalization": self._normalize_date
                },
                "number": {
                    "regex_patterns": [
                        r'^\d+(\.\d+)?$',  # Basic number
                        r'^\$\d+(\.\d+)?$',  # Currency
                        r'^\d+(,\d{3})*(\.\d+)?$',  # With commas
                    ],
                    "validation": self._validate_number,
                    "normalization": self._normalize_number
                },
                "email": {
                    "regex_patterns": [
                        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                    ],
                    "validation": self._validate_email,
                    "normalization": self._normalize_email
                },
                "phone": {
                    "regex_patterns": [
                        r'^\+?1?\s*\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$'
                    ],
                    "validation": self._validate_phone,
                    "normalization": self._normalize_phone
                }
            },
            "error_patterns": {
                "date_format": [
                    r"date.*format",
                    r"invalid.*date",
                    r"date.*parse"
                ],
                "number_format": [
                    r"currency.*format",
                    r"decimal.*separator",
                    r"invalid.*number"
                ],
                "field_missing": [
                    r"field.*missing",
                    r"not.*extracted",
                    r"empty.*field"
                ],
                "partial_extraction": [
                    r"partial.*match",
                    r"incomplete.*extraction",
                    r"truncated.*value"
                ]
            }
        }
    
    def evaluate_field_statistically(self,
                                   field_name: str,
                                   expected_value: Optional[str],
                                   extracted_value: Optional[str],
                                   confidence_score: float = 0.0,
                                   field_type: str = "text") -> FieldEvaluationResult:
        """
        Evaluate a field using pure statistical methods.
        
        Args:
            field_name: Name of the field
            expected_value: Ground truth value
            extracted_value: Extracted value
            confidence_score: Confidence score from extraction
            field_type: Type of field
            
        Returns:
            FieldEvaluationResult: Evaluation result
        """
        
        # Handle missing values
        if expected_value is None and extracted_value is None:
            return self._create_success_result(field_name, expected_value, extracted_value, confidence_score, field_type)
        
        if expected_value is None:
            return self._create_failure_result(field_name, expected_value, extracted_value, confidence_score, field_type, "Expected value is None")
        
        if extracted_value is None:
            return self._create_missing_result(field_name, expected_value, extracted_value, confidence_score, field_type)
        
        # Normalize values
        expected_norm = self._normalize_value(expected_value, field_type)
        extracted_norm = self._normalize_value(extracted_value, field_type)
        
        # Calculate evaluation score
        evaluation_score = self._calculate_statistical_score(expected_norm, extracted_norm, field_type)
        
        # Determine status
        status = self._determine_status(evaluation_score, confidence_score)
        
        # Generate error message if needed
        error_message = self._generate_error_message(expected_norm, extracted_norm, evaluation_score, status)
        
        # Create evaluation notes
        evaluation_notes = self._generate_evaluation_notes(expected_norm, extracted_norm, evaluation_score, confidence_score, field_type)
        
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
    
    def _normalize_value(self, value: str, field_type: str) -> str:
        """Normalize value based on field type."""
        
        if not value:
            return ""
        
        value_str = str(value).strip()
        
        # Apply field-type specific normalization
        if field_type in self.evaluation_rules["field_patterns"]:
            normalizer = self.evaluation_rules["field_patterns"][field_type].get("normalization")
            if normalizer:
                return normalizer(value_str)
        
        return value_str.lower()
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date strings."""
        
        # Try to standardize to YYYY-MM-DD format
        patterns = [
            (r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\3-\1-\2'),  # MM/DD/YYYY
            (r'(\d{1,2})-(\d{1,2})-(\d{4})', r'\3-\1-\2'),  # MM-DD-YYYY
        ]
        
        for pattern, replacement in patterns:
            if re.match(pattern, date_str):
                return re.sub(pattern, replacement, date_str)
        
        return date_str
    
    def _normalize_number(self, number_str: str) -> str:
        """Normalize number strings."""
        
        # Remove currency symbols and commas
        cleaned = re.sub(r'[^\d.,]', '', number_str)
        cleaned = cleaned.replace(',', '')
        
        return cleaned
    
    def _normalize_email(self, email_str: str) -> str:
        """Normalize email addresses."""
        
        return email_str.lower().strip()
    
    def _normalize_phone(self, phone_str: str) -> str:
        """Normalize phone numbers."""
        
        # Remove all non-digit characters except +
        cleaned = re.sub(r'[^\d+]', '', phone_str)
        
        return cleaned
    
    def _calculate_statistical_score(self, expected: str, extracted: str, field_type: str) -> float:
        """Calculate evaluation score using statistical methods."""
        
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
        
        # Use difflib for string similarity
        import difflib
        similarity = difflib.SequenceMatcher(None, expected, extracted).ratio()
        
        return similarity
    
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
        
        # Check if they're the same number with different formatting
        expected_digits = re.sub(r'[^\d]', '', expected)
        extracted_digits = re.sub(r'[^\d]', '', extracted)
        
        if expected_digits == extracted_digits:
            return 0.9
        
        return self._score_text(expected, extracted)
    
    def _determine_status(self, evaluation_score: float, confidence_score: float) -> ExtractionStatus:
        """Determine extraction status based on scores."""
        
        # Check confidence threshold
        if confidence_score < 0.5:  # Default threshold
            return ExtractionStatus.FAILED
        
        # Determine status based on evaluation score
        if evaluation_score >= 0.9:
            return ExtractionStatus.SUCCESS
        elif evaluation_score >= 0.7:
            return ExtractionStatus.PARTIAL
        else:
            return ExtractionStatus.FAILED
    
    def _generate_error_message(self,
                              expected: str,
                              extracted: str,
                              evaluation_score: float,
                              status: ExtractionStatus) -> Optional[str]:
        """Generate error message for failed extractions."""
        
        if status == ExtractionStatus.SUCCESS:
            return None
        
        if status == ExtractionStatus.MISSING:
            return "Field was not extracted"
        
        if status == ExtractionStatus.FAILED:
            if evaluation_score == 0.0:
                return "Complete mismatch between expected and extracted values"
            else:
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
                import difflib
                similarity = difflib.SequenceMatcher(None, expected, extracted).ratio()
                notes.append(f"String similarity: {similarity:.3f}")
        
        return "; ".join(notes)
    
    def _create_success_result(self, field_name: str, expected_value: Optional[str], 
                             extracted_value: Optional[str], confidence_score: float, 
                             field_type: str) -> FieldEvaluationResult:
        """Create a success result."""
        
        return FieldEvaluationResult(
            field_name=field_name,
            expected_value=expected_value,
            extracted_value=extracted_value,
            confidence_score=confidence_score,
            status=ExtractionStatus.SUCCESS,
            evaluation_score=1.0,
            error_message="Both values are None",
            evaluation_notes=f"Field type: {field_type}; Both values are None",
            field_type=field_type
        )
    
    def _create_failure_result(self, field_name: str, expected_value: Optional[str], 
                             extracted_value: Optional[str], confidence_score: float, 
                             field_type: str, error_message: str) -> FieldEvaluationResult:
        """Create a failure result."""
        
        return FieldEvaluationResult(
            field_name=field_name,
            expected_value=expected_value,
            extracted_value=extracted_value,
            confidence_score=confidence_score,
            status=ExtractionStatus.FAILED,
            evaluation_score=0.0,
            error_message=error_message,
            evaluation_notes=f"Field type: {field_type}; {error_message}",
            field_type=field_type
        )
    
    def _create_missing_result(self, field_name: str, expected_value: Optional[str], 
                             extracted_value: Optional[str], confidence_score: float, 
                             field_type: str) -> FieldEvaluationResult:
        """Create a missing result."""
        
        return FieldEvaluationResult(
            field_name=field_name,
            expected_value=expected_value,
            extracted_value=extracted_value,
            confidence_score=confidence_score,
            status=ExtractionStatus.MISSING,
            evaluation_score=0.0,
            error_message="Field was not extracted",
            evaluation_notes=f"Field type: {field_type}; Field was not extracted",
            field_type=field_type
        )
    
    def detect_error_patterns_statistically(self, evaluation_results: List[DocumentEvaluationResult]) -> List[Dict[str, Any]]:
        """
        Detect error patterns using statistical analysis.
        
        Args:
            evaluation_results: List of evaluation results
            
        Returns:
            List of detected error patterns
        """
        
        patterns = []
        
        # Collect all failed field evaluations
        failed_evaluations = []
        for result in evaluation_results:
            for field_eval in result.field_evaluations:
                if field_eval.is_failed():
                    failed_evaluations.append({
                        'field_eval': field_eval,
                        'document_id': result.document_id,
                        'document_type': result.document_type
                    })
        
        if not failed_evaluations:
            return patterns
        
        # Pattern 1: Field-specific failures
        field_patterns = self._detect_field_specific_patterns(failed_evaluations)
        patterns.extend(field_patterns)
        
        # Pattern 2: Error message patterns
        error_patterns = self._detect_error_message_patterns(failed_evaluations)
        patterns.extend(error_patterns)
        
        # Pattern 3: Document type patterns
        doc_type_patterns = self._detect_document_type_patterns(failed_evaluations)
        patterns.extend(doc_type_patterns)
        
        return patterns
    
    def _detect_field_specific_patterns(self, failed_evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns specific to certain fields."""
        
        # Group by field name
        field_groups = defaultdict(list)
        for eval_data in failed_evaluations:
            field_name = eval_data['field_eval'].field_name
            field_groups[field_name].append(eval_data)
        
        patterns = []
        for field_name, evaluations in field_groups.items():
            if len(evaluations) >= 3:  # Minimum threshold for pattern detection
                pattern = {
                    "pattern_type": "field_specific_failure",
                    "field_name": field_name,
                    "frequency": len(evaluations),
                    "error_messages": list(set(e['field_eval'].error_message for e in evaluations if e['field_eval'].error_message)),
                    "suggested_fixes": self._generate_field_fixes(field_name, evaluations)
                }
                patterns.append(pattern)
        
        return patterns
    
    def _detect_error_message_patterns(self, failed_evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns based on error messages."""
        
        # Group by error message similarity
        error_groups = defaultdict(list)
        
        for eval_data in failed_evaluations:
            error_msg = eval_data['field_eval'].error_message
            if error_msg:
                # Find matching pattern
                pattern_type = self._classify_error_message(error_msg)
                error_groups[pattern_type].append(eval_data)
        
        patterns = []
        for pattern_type, evaluations in error_groups.items():
            if len(evaluations) >= 2:  # Minimum threshold
                pattern = {
                    "pattern_type": pattern_type,
                    "frequency": len(evaluations),
                    "affected_fields": list(set(e['field_eval'].field_name for e in evaluations)),
                    "suggested_fixes": self._generate_error_fixes(pattern_type, evaluations)
                }
                patterns.append(pattern)
        
        return patterns
    
    def _detect_document_type_patterns(self, failed_evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect patterns specific to document types."""
        
        # Group by document type
        doc_type_groups = defaultdict(list)
        for eval_data in failed_evaluations:
            doc_type = eval_data['document_type']
            doc_type_groups[doc_type].append(eval_data)
        
        patterns = []
        for doc_type, evaluations in doc_type_groups.items():
            if len(evaluations) >= 3:  # Minimum threshold
                pattern = {
                    "pattern_type": "document_type_failure",
                    "document_type": doc_type,
                    "frequency": len(evaluations),
                    "affected_fields": list(set(e['field_eval'].field_name for e in evaluations)),
                    "suggested_fixes": self._generate_doc_type_fixes(doc_type, evaluations)
                }
                patterns.append(pattern)
        
        return patterns
    
    def _classify_error_message(self, error_message: str) -> str:
        """Classify error message into pattern types."""
        
        error_lower = error_message.lower()
        
        for pattern_type, patterns in self.evaluation_rules["error_patterns"].items():
            for pattern in patterns:
                if re.search(pattern, error_lower):
                    return pattern_type
        
        return "unknown_error"
    
    def _generate_field_fixes(self, field_name: str, evaluations: List[Dict[str, Any]]) -> List[str]:
        """Generate fixes for field-specific failures."""
        
        fixes = [
            f"Add specific instructions for {field_name} field",
            f"Provide examples for {field_name} extraction",
            f"Review field definition for {field_name}"
        ]
        
        return fixes
    
    def _generate_error_fixes(self, pattern_type: str, evaluations: List[Dict[str, Any]]) -> List[str]:
        """Generate fixes for error message patterns."""
        
        fixes = []
        
        if pattern_type == "date_format":
            fixes.extend([
                "Add explicit date format instructions",
                "Include date format examples",
                "Specify expected date patterns"
            ])
        elif pattern_type == "number_format":
            fixes.extend([
                "Add number format specifications",
                "Include currency handling instructions",
                "Specify decimal and thousand separators"
            ])
        elif pattern_type == "field_missing":
            fixes.extend([
                "Improve field detection instructions",
                "Add field location hints",
                "Include fallback extraction logic"
            ])
        elif pattern_type == "partial_extraction":
            fixes.extend([
                "Add completeness requirements",
                "Specify full field extraction",
                "Include validation for partial matches"
            ])
        
        return fixes
    
    def _generate_doc_type_fixes(self, doc_type: str, evaluations: List[Dict[str, Any]]) -> List[str]:
        """Generate fixes for document type patterns."""
        
        fixes = [
            f"Add {doc_type}-specific extraction instructions",
            f"Include {doc_type} field definitions",
            f"Provide {doc_type} examples in prompt"
        ]
        
        return fixes 