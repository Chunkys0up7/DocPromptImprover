#!/usr/bin/env python3
"""
Simple demonstration of the evaluation framework.

This script demonstrates the core functionality of the evaluation framework
with a simple console output.
"""

import json
from pathlib import Path
from typing import List

from src.evaluators.field_evaluator import FieldEvaluator
from src.evaluators.document_aggregator import DocumentAggregator
from src.evaluators.error_pattern_detector import ErrorPatternDetector
from src.statistics.statistics_engine import StatisticsEngine
from src.models.evaluation_models import (
    DocumentEvaluationInput,
    DocumentEvaluationResult,
    ExtractionStatus
)
from data.dummy_data_generator import DummyDataGenerator


def main():
    """Run a simple demonstration of the evaluation framework."""
    
    print("=" * 60)
    print("Document Extraction Evaluation Framework - Simple Demo")
    print("=" * 60)
    
    # Initialize components
    print("\n🔧 Initializing components...")
    field_evaluator = FieldEvaluator()
    document_aggregator = DocumentAggregator()
    error_detector = ErrorPatternDetector()
    statistics_engine = StatisticsEngine()
    data_generator = DummyDataGenerator()
    
    # Generate test data
    print("\n📊 Generating test data...")
    evaluation_inputs = data_generator.generate_evaluation_inputs(3)
    print(f"Generated {len(evaluation_inputs)} test documents")
    
    # Process each document
    document_results = []
    
    for i, input_data in enumerate(evaluation_inputs, 1):
        print(f"\n📄 Processing document {i}: {input_data.document_id}")
        print(f"   Type: {input_data.document_type}")
        print(f"   Fields: {len(input_data.extracted_fields)}")
        
        # Evaluate fields
        field_evaluations = []
        for field_name in set(input_data.extracted_fields.keys()) | set(input_data.ground_truth.keys()):
            expected_value = input_data.ground_truth.get(field_name)
            extracted_value = input_data.extracted_fields.get(field_name)
            confidence_score = input_data.confidence_scores.get(field_name, 0.0)
            
            # Determine field type
            field_type = "text"  # Default type
            if field_name in ["total_amount", "tax_amount", "subtotal"]:
                field_type = "number"
            elif "date" in field_name:
                field_type = "date"
            elif "email" in field_name:
                field_type = "email"
            elif "phone" in field_name:
                field_type = "phone"
            
            result = field_evaluator.evaluate_field(
                field_name=field_name,
                expected_value=expected_value,
                extracted_value=extracted_value,
                confidence_score=confidence_score,
                field_type=field_type
            )
            field_evaluations.append(result)
            
            status_emoji = "✅" if result.status == ExtractionStatus.SUCCESS else "❌"
            print(f"   {status_emoji} {field_name}: {result.status.value} (score: {result.evaluation_score:.3f})")
        
        # Aggregate document results
        doc_result = document_aggregator.aggregate_evaluations(
            field_evaluations=field_evaluations,
            document_id=input_data.document_id,
            document_type=input_data.document_type,
            confidence_scores=input_data.confidence_scores,
            prompt_version=input_data.prompt_version
        )
        document_results.append(doc_result)
        
        print(f"   📈 Overall accuracy: {doc_result.overall_accuracy:.3f}")
        print(f"   🎯 Confidence correlation: {doc_result.confidence_correlation:.3f}")
        
        # Update statistics
        statistics_engine.update_statistics(doc_result)
    
    # Generate error patterns
    print("\n🔍 Analyzing error patterns...")
    error_patterns = error_detector.detect_patterns(document_results)
    
    if error_patterns:
        print(f"Found {len(error_patterns)} error patterns:")
        for pattern in error_patterns[:3]:  # Show first 3 patterns
            print(f"   - {pattern.pattern_type}: {pattern.error_messages[0] if pattern.error_messages else 'No specific error message'} (frequency: {pattern.frequency})")
    else:
        print("No significant error patterns detected.")
    
    # Display overall statistics
    print("\n📊 Overall Statistics:")
    stats = statistics_engine.statistics
    print(f"   Total documents: {stats.total_documents}")
    print(f"   Total fields: {stats.total_fields}")
    print(f"   Successful extractions: {stats.successful_extractions}")
    print(f"   Failed extractions: {stats.failed_extractions}")
    print(f"   Average accuracy: {stats.average_accuracy:.3f}")
    print(f"   Average confidence: {stats.average_confidence:.3f}")
    
    # Show field performance
    print("\n🎯 Field Performance:")
    for field_name, success_rates in stats.field_success_rates.items():
        avg_rate = sum(success_rates) / len(success_rates)
        print(f"   {field_name}: {avg_rate:.3f}")
    
    # Show document type performance
    print("\n📋 Document Type Performance:")
    for doc_type, accuracies in stats.document_type_performance.items():
        avg_accuracy = sum(accuracies) / len(accuracies)
        print(f"   {doc_type}: {avg_accuracy:.3f}")
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main() 