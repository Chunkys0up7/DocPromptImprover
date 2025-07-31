"""
Error pattern detection and analysis for document extraction evaluation.

This module identifies common failure patterns across multiple evaluations
and provides insights for prompt optimization and system improvement.
"""

import re
import statistics
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict, Counter
from datetime import datetime

from ..models.evaluation_models import (
    FieldEvaluationResult,
    DocumentEvaluationResult,
    FailurePattern,
    ExtractionStatus
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ErrorPatternDetector:
    """
    Detects and analyzes error patterns in document extraction evaluations.
    
    This class identifies common failure patterns, categorizes errors,
    and provides insights for prompt optimization.
    """
    
    def __init__(self):
        """Initialize the error pattern detector."""
        logger.info("ErrorPatternDetector initialized")
        
        # Common error patterns
        self.error_patterns = {
            "date_format": [
                r"date.*format",
                r"invalid.*date",
                r"date.*parse",
                r"mm/dd/yyyy",
                r"dd/mm/yyyy"
            ],
            "number_format": [
                r"currency.*format",
                r"decimal.*separator",
                r"thousand.*separator",
                r"invalid.*number",
                r"amount.*format"
            ],
            "field_missing": [
                r"field.*missing",
                r"not.*extracted",
                r"empty.*field",
                r"no.*value"
            ],
            "partial_extraction": [
                r"partial.*match",
                r"incomplete.*extraction",
                r"truncated.*value",
                r"missing.*part"
            ],
            "format_mismatch": [
                r"format.*mismatch",
                r"wrong.*format",
                r"unexpected.*format",
                r"format.*error"
            ],
            "confidence_low": [
                r"low.*confidence",
                r"uncertain.*extraction",
                r"confidence.*below.*threshold"
            ]
        }
    
    def detect_patterns(self,
                       evaluation_results: List[DocumentEvaluationResult]) -> List[FailurePattern]:
        """
        Detect failure patterns across multiple evaluation results.
        
        Args:
            evaluation_results: List of document evaluation results
            
        Returns:
            List[FailurePattern]: List of detected failure patterns
        """
        
        logger.info(f"Detecting patterns in {len(evaluation_results)} evaluation results")
        
        # Collect all failed field evaluations
        failed_evaluations = []
        for result in evaluation_results:
            for field_eval in result.field_evaluations:
                if field_eval.is_failed():
                    failed_evaluations.append({
                        'field_eval': field_eval,
                        'document_id': result.document_id,
                        'document_type': result.document_type,
                        'prompt_version': result.prompt_version,
                        'document_timestamp': result.evaluation_timestamp
                    })
        
        if not failed_evaluations:
            logger.info("No failed evaluations found")
            return []
        
        # Analyze patterns
        patterns = []
        
        # Pattern 1: Field-specific failures
        field_patterns = self._detect_field_specific_patterns(failed_evaluations)
        patterns.extend(field_patterns)
        
        # Pattern 2: Error message patterns
        error_patterns = self._detect_error_message_patterns(failed_evaluations)
        patterns.extend(error_patterns)
        
        # Pattern 3: Document type patterns
        doc_type_patterns = self._detect_document_type_patterns(failed_evaluations)
        patterns.extend(doc_type_patterns)
        
        # Pattern 4: Confidence-based patterns
        confidence_patterns = self._detect_confidence_patterns(failed_evaluations)
        patterns.extend(confidence_patterns)
        
        # Pattern 5: Value similarity patterns
        similarity_patterns = self._detect_similarity_patterns(failed_evaluations)
        patterns.extend(similarity_patterns)
        
        logger.info(f"Detected {len(patterns)} failure patterns")
        return patterns
    
    def _detect_field_specific_patterns(self, failed_evaluations: List[Dict[str, Any]]) -> List[FailurePattern]:
        """Detect patterns specific to certain fields."""
        
        # Group by field name
        field_groups = defaultdict(list)
        for eval_data in failed_evaluations:
            field_name = eval_data['field_eval'].field_name
            field_groups[field_name].append(eval_data)
        
        patterns = []
        for field_name, evaluations in field_groups.items():
            if len(evaluations) >= 3:  # Minimum threshold for pattern detection
                pattern = FailurePattern(
                    pattern_id=f"field_failure_{field_name}_{datetime.now().isoformat()}",
                    pattern_type="field_specific_failure",
                    affected_fields=[field_name],
                    error_messages=self._extract_error_messages(evaluations),
                    frequency=len(evaluations),
                    impact_score=self._calculate_impact_score(evaluations),
                    suggested_fixes=self._generate_field_fixes(field_name, evaluations),
                    first_seen=min(e['document_timestamp'] for e in evaluations),
                    last_seen=max(e['document_timestamp'] for e in evaluations)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_error_message_patterns(self, failed_evaluations: List[Dict[str, Any]]) -> List[FailurePattern]:
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
                pattern = FailurePattern(
                    pattern_id=f"error_message_{pattern_type}_{datetime.now().isoformat()}",
                    pattern_type=f"error_message_{pattern_type}",
                    affected_fields=list(set(e['field_eval'].field_name for e in evaluations)),
                    error_messages=self._extract_error_messages(evaluations),
                    frequency=len(evaluations),
                    impact_score=self._calculate_impact_score(evaluations),
                    suggested_fixes=self._generate_error_fixes(pattern_type, evaluations),
                    first_seen=min(e['document_timestamp'] for e in evaluations),
                    last_seen=max(e['document_timestamp'] for e in evaluations)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_document_type_patterns(self, failed_evaluations: List[Dict[str, Any]]) -> List[FailurePattern]:
        """Detect patterns specific to document types."""
        
        # Group by document type
        doc_type_groups = defaultdict(list)
        for eval_data in failed_evaluations:
            doc_type = eval_data['document_type']
            doc_type_groups[doc_type].append(eval_data)
        
        patterns = []
        for doc_type, evaluations in doc_type_groups.items():
            if len(evaluations) >= 3:  # Minimum threshold
                pattern = FailurePattern(
                    pattern_id=f"doc_type_failure_{doc_type}_{datetime.now().isoformat()}",
                    pattern_type="document_type_failure",
                    affected_fields=list(set(e['field_eval'].field_name for e in evaluations)),
                    error_messages=self._extract_error_messages(evaluations),
                    frequency=len(evaluations),
                    impact_score=self._calculate_impact_score(evaluations),
                    suggested_fixes=self._generate_doc_type_fixes(doc_type, evaluations),
                    first_seen=min(e['document_timestamp'] for e in evaluations),
                    last_seen=max(e['document_timestamp'] for e in evaluations)
                )
                patterns.append(pattern)
        
        return patterns
    
    def _detect_confidence_patterns(self, failed_evaluations: List[Dict[str, Any]]) -> List[FailurePattern]:
        """Detect patterns related to confidence scores."""
        
        # Analyze confidence distributions
        low_confidence = [e for e in failed_evaluations if e['field_eval'].confidence_score < 0.5]
        high_confidence_failures = [e for e in failed_evaluations if e['field_eval'].confidence_score > 0.8]
        
        patterns = []
        
        # Low confidence pattern
        if len(low_confidence) >= 3:
            pattern = FailurePattern(
                pattern_id=f"low_confidence_{datetime.now().isoformat()}",
                pattern_type="low_confidence_failures",
                affected_fields=list(set(e['field_eval'].field_name for e in low_confidence)),
                error_messages=["Low confidence in extraction"],
                frequency=len(low_confidence),
                impact_score=self._calculate_impact_score(low_confidence),
                suggested_fixes=[
                    "Improve field extraction instructions",
                    "Add more context to the prompt",
                    "Consider field-specific examples"
                ],
                first_seen=min(e['document_timestamp'] for e in low_confidence),
                last_seen=max(e['document_timestamp'] for e in low_confidence)
            )
            patterns.append(pattern)
        
        # High confidence but wrong pattern
        if len(high_confidence_failures) >= 2:
            pattern = FailurePattern(
                pattern_id=f"high_confidence_failures_{datetime.now().isoformat()}",
                pattern_type="high_confidence_failures",
                affected_fields=list(set(e['field_eval'].field_name for e in high_confidence_failures)),
                error_messages=["High confidence but incorrect extraction"],
                frequency=len(high_confidence_failures),
                impact_score=self._calculate_impact_score(high_confidence_failures),
                suggested_fixes=[
                    "Review confidence calibration",
                    "Improve validation logic",
                    "Add more specific field definitions"
                ],
                first_seen=min(e['document_timestamp'] for e in high_confidence_failures),
                last_seen=max(e['document_timestamp'] for e in high_confidence_failures)
            )
            patterns.append(pattern)
        
        return patterns
    
    def _detect_similarity_patterns(self, failed_evaluations: List[Dict[str, Any]]) -> List[FailurePattern]:
        """Detect patterns based on value similarities."""
        
        # Group by field type and analyze value patterns
        field_type_groups = defaultdict(list)
        for eval_data in failed_evaluations:
            field_type = eval_data['field_eval'].field_type
            field_type_groups[field_type].append(eval_data)
        
        patterns = []
        for field_type, evaluations in field_type_groups.items():
            if len(evaluations) >= 3:
                # Analyze common value patterns
                value_patterns = self._analyze_value_patterns(evaluations, field_type)
                if value_patterns:
                    pattern = FailurePattern(
                        pattern_id=f"value_pattern_{field_type}_{datetime.now().isoformat()}",
                        pattern_type="value_pattern_failure",
                        affected_fields=list(set(e['field_eval'].field_name for e in evaluations)),
                        error_messages=value_patterns,
                        frequency=len(evaluations),
                        impact_score=self._calculate_impact_score(evaluations),
                        suggested_fixes=self._generate_value_pattern_fixes(field_type, evaluations),
                        first_seen=min(e['document_timestamp'] for e in evaluations),
                        last_seen=max(e['document_timestamp'] for e in evaluations)
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _classify_error_message(self, error_message: str) -> str:
        """Classify error message into pattern types."""
        
        error_lower = error_message.lower()
        
        for pattern_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, error_lower):
                    return pattern_type
        
        return "unknown_error"
    
    def _extract_error_messages(self, evaluations: List[Dict[str, Any]]) -> List[str]:
        """Extract unique error messages from evaluations."""
        
        messages = []
        for eval_data in evaluations:
            error_msg = eval_data['field_eval'].error_message
            if error_msg and error_msg not in messages:
                messages.append(error_msg)
        
        return messages
    
    def _calculate_impact_score(self, evaluations: List[Dict[str, Any]]) -> float:
        """Calculate impact score for a pattern."""
        
        if not evaluations:
            return 0.0
        
        # Factors: frequency, field importance, evaluation scores
        frequency_factor = min(len(evaluations) / 10.0, 1.0)  # Normalize to 0-1
        
        # Field importance factor
        important_fields = {"invoice_number", "total_amount", "vendor_name", "invoice_date"}
        important_count = sum(1 for e in evaluations 
                            if e['field_eval'].field_name in important_fields)
        importance_factor = important_count / len(evaluations)
        
        # Evaluation score factor (lower scores = higher impact)
        avg_evaluation_score = statistics.mean(e['field_eval'].evaluation_score 
                                             for e in evaluations)
        score_factor = 1.0 - avg_evaluation_score
        
        # Weighted combination
        impact_score = (frequency_factor * 0.4 + 
                       importance_factor * 0.4 + 
                       score_factor * 0.2)
        
        return min(1.0, max(0.0, impact_score))
    
    def _generate_field_fixes(self, field_name: str, evaluations: List[Dict[str, Any]]) -> List[str]:
        """Generate fixes for field-specific failures."""
        
        fixes = []
        
        # Analyze field type
        field_types = set(e['field_eval'].field_type for e in evaluations)
        if len(field_types) == 1:
            field_type = list(field_types)[0]
            
            if field_type == "date":
                fixes.extend([
                    "Specify expected date format in prompt",
                    "Add date format examples",
                    "Include date parsing instructions"
                ])
            elif field_type == "number":
                fixes.extend([
                    "Specify number format requirements",
                    "Add currency symbol handling",
                    "Include decimal separator instructions"
                ])
            elif field_type == "email":
                fixes.extend([
                    "Add email validation rules",
                    "Specify email format requirements"
                ])
        
        # General fixes
        fixes.extend([
            f"Add specific instructions for {field_name} field",
            f"Provide examples for {field_name} extraction",
            f"Review field definition for {field_name}"
        ])
        
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
        elif pattern_type == "format_mismatch":
            fixes.extend([
                "Clarify expected formats",
                "Add format validation rules",
                "Include format conversion instructions"
            ])
        elif pattern_type == "confidence_low":
            fixes.extend([
                "Improve field extraction clarity",
                "Add more context to prompts",
                "Include confidence boosting instructions"
            ])
        
        return fixes
    
    def _generate_doc_type_fixes(self, doc_type: str, evaluations: List[Dict[str, Any]]) -> List[str]:
        """Generate fixes for document type patterns."""
        
        fixes = [
            f"Add {doc_type}-specific extraction instructions",
            f"Include {doc_type} field definitions",
            f"Provide {doc_type} examples in prompt",
            f"Add {doc_type} validation rules"
        ]
        
        return fixes
    
    def _analyze_value_patterns(self, evaluations: List[Dict[str, Any]], field_type: str) -> List[str]:
        """Analyze patterns in extracted values."""
        
        patterns = []
        
        # Extract values
        extracted_values = [e['field_eval'].extracted_value for e in evaluations 
                          if e['field_eval'].extracted_value]
        expected_values = [e['field_eval'].expected_value for e in evaluations 
                          if e['field_eval'].expected_value]
        
        if not extracted_values or not expected_values:
            return patterns
        
        # Analyze common patterns
        if field_type == "date":
            # Check for common date format issues
            if any("/" in val for val in extracted_values):
                patterns.append("Inconsistent date separators")
            if any(len(val) != 10 for val in extracted_values if val):
                patterns.append("Inconsistent date length")
        
        elif field_type == "number":
            # Check for currency symbol issues
            if any("$" in val for val in extracted_values):
                patterns.append("Currency symbols in numbers")
            if any("," in val for val in extracted_values):
                patterns.append("Thousand separators in numbers")
        
        elif field_type == "text":
            # Check for truncation
            avg_extracted_len = statistics.mean(len(val) for val in extracted_values)
            avg_expected_len = statistics.mean(len(val) for val in expected_values)
            if avg_extracted_len < avg_expected_len * 0.8:
                patterns.append("Text truncation detected")
        
        return patterns
    
    def _generate_value_pattern_fixes(self, field_type: str, evaluations: List[Dict[str, Any]]) -> List[str]:
        """Generate fixes for value pattern issues."""
        
        fixes = []
        
        if field_type == "date":
            fixes.extend([
                "Standardize date format instructions",
                "Add date validation rules",
                "Include date format examples"
            ])
        elif field_type == "number":
            fixes.extend([
                "Add number cleaning instructions",
                "Specify currency symbol handling",
                "Include number format examples"
            ])
        elif field_type == "text":
            fixes.extend([
                "Add text completeness requirements",
                "Specify text length expectations",
                "Include text validation rules"
            ])
        
        return fixes
    
    def get_pattern_summary(self, patterns: List[FailurePattern]) -> Dict[str, Any]:
        """
        Generate a summary of detected patterns.
        
        Args:
            patterns: List of failure patterns
            
        Returns:
            Dict[str, Any]: Pattern summary
        """
        
        if not patterns:
            return {"message": "No patterns detected"}
        
        # Group by pattern type
        type_groups = defaultdict(list)
        for pattern in patterns:
            type_groups[pattern.pattern_type].append(pattern)
        
        # Calculate statistics
        total_failures = sum(p.frequency for p in patterns)
        avg_impact = statistics.mean(p.impact_score for p in patterns)
        
        # Most common patterns
        sorted_patterns = sorted(patterns, key=lambda p: p.frequency, reverse=True)
        top_patterns = sorted_patterns[:5]
        
        summary = {
            "total_patterns": len(patterns),
            "total_failures": total_failures,
            "average_impact_score": avg_impact,
            "pattern_types": list(type_groups.keys()),
            "top_patterns": [
                {
                    "pattern_id": p.pattern_id,
                    "pattern_type": p.pattern_type,
                    "frequency": p.frequency,
                    "impact_score": p.impact_score,
                    "affected_fields": p.affected_fields
                }
                for p in top_patterns
            ],
            "detection_timestamp": datetime.now().isoformat()
        }
        
        return summary 