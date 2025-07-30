"""
API package for evaluation-only document extraction framework.
"""

from .evaluation_service import (
    app,
    DocumentExtractionEvaluator,
    evaluator
)

__all__ = [
    "app",
    "DocumentExtractionEvaluator", 
    "evaluator"
] 