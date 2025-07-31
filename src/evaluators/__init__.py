"""
Evaluators package for evaluation-only document extraction framework.
"""

from .field_evaluator import FieldEvaluator
from .document_aggregator import DocumentAggregator
from .error_pattern_detector import ErrorPatternDetector
from .evaluation_signatures import StatisticalEvaluator

__all__ = [
    "FieldEvaluator",
    "DocumentAggregator",
    "ErrorPatternDetector",
    "StatisticalEvaluator"
] 