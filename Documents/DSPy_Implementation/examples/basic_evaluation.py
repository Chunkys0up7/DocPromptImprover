"""
Basic DSPy Evaluation Example

This example demonstrates how to use DSPy for basic field evaluation
and document aggregation in the document extraction evaluation system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import json
from typing import Dict, Any

# Import DSPy components
from code.dspy_config import initialize_dspy, get_dspy_manager
from code.dspy_modules import DSPyFieldEvaluator, DSPyDocumentAggregator


def basic_field_evaluation_example():
    """Demonstrate basic field evaluation using DSPy."""
    
    print("=== Basic Field Evaluation Example ===\n")
    
    # Initialize DSPy
    try:
        manager = initialize_dspy()
        print("✅ DSPy initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize DSPy: {e}")
        print("Make sure you have set up your API keys in the .env file")
        return
    
    # Create field evaluator
    field_evaluator = DSPyFieldEvaluator()
    
    # Test cases
    test_cases = [
        {
            "name": "Perfect Match",
            "field_name": "vendor_name",
            "expected_value": "Acme Corporation",
            "extracted_value": "Acme Corporation",
            "confidence_score": 0.95,
            "field_type": "text"
        },
        {
            "name": "Partial Match",
            "field_name": "total_amount",
            "expected_value": "1250.00",
            "extracted_value": "1250",
            "confidence_score": 0.8,
            "field_type": "number"
        },
        {
            "name": "Date Format Variation",
            "field_name": "invoice_date",
            "expected_value": "2024-01-15",
            "extracted_value": "01/15/2024",
            "confidence_score": 0.7,
            "field_type": "date"
        },
        {
            "name": "Complete Failure",
            "field_name": "email",
            "expected_value": "billing@acme.com",
            "extracted_value": "invalid@email.com",
            "confidence_score": 0.3,
            "field_type": "email"
        },
        {
            "name": "Missing Field",
            "field_name": "phone",
            "expected_value": "+1-555-123-4567",
            "extracted_value": None,
            "confidence_score": 0.1,
            "field_type": "phone"
        }
    ]
    
    # Evaluate each test case
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['name']} ---")
        
        try:
            result = field_evaluator(
                field_name=test_case["field_name"],
                expected_value=test_case["expected_value"],
                extracted_value=test_case["extracted_value"],
                confidence_score=test_case["confidence_score"],
                field_type=test_case["field_type"]
            )
            
            results.append(result)
            
            print(f"Field: {test_case['field_name']}")
            print(f"Expected: {test_case['expected_value']}")
            print(f"Extracted: {test_case['extracted_value']}")
            print(f"Confidence: {test_case['confidence_score']:.2f}")
            print(f"Evaluation Score: {result['evaluation_score']:.3f}")
            print(f"Status: {result['status']}")
            print(f"Method: {result['method']}")
            
            if result['error_message']:
                print(f"Error: {result['error_message']}")
            
            print(f"Notes: {result['evaluation_notes'][:100]}...")
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            results.append({
                "evaluation_score": 0.0,
                "status": "failed",
                "error_message": str(e),
                "method": "error"
            })
    
    return results


def document_aggregation_example():
    """Demonstrate document-level aggregation using DSPy."""
    
    print("\n\n=== Document Aggregation Example ===\n")
    
    # Create document aggregator
    document_aggregator = DSPyDocumentAggregator()
    
    # Sample field evaluations (from previous example)
    field_evaluations = [
        {
            "field_name": "vendor_name",
            "evaluation_score": 1.0,
            "status": "success",
            "error_message": None
        },
        {
            "field_name": "total_amount",
            "evaluation_score": 0.8,
            "status": "partial",
            "error_message": "Missing decimal places"
        },
        {
            "field_name": "invoice_date",
            "evaluation_score": 0.9,
            "status": "success",
            "error_message": None
        },
        {
            "field_name": "email",
            "evaluation_score": 0.0,
            "status": "failed",
            "error_message": "Completely wrong email"
        },
        {
            "field_name": "phone",
            "evaluation_score": 0.0,
            "status": "missing",
            "error_message": "Field not extracted"
        }
    ]
    
    confidence_scores = {
        "vendor_name": 0.95,
        "total_amount": 0.8,
        "invoice_date": 0.7,
        "email": 0.3,
        "phone": 0.1
    }
    
    try:
        result = document_aggregator(
            field_evaluations=field_evaluations,
            document_type="invoice",
            confidence_scores=confidence_scores,
            prompt_version="v1.0"
        )
        
        print("Document Aggregation Results:")
        print(f"Overall Accuracy: {result['overall_accuracy']:.3f}")
        print(f"Confidence Correlation: {result['confidence_correlation']:.3f}")
        print(f"Quality Assessment: {result['quality_assessment']}")
        print(f"Method: {result['method']}")
        
        if result['critical_errors']:
            print(f"Critical Errors: {result['critical_errors']}")
        
        if result['improvement_suggestions']:
            print(f"Improvement Suggestions: {result['improvement_suggestions']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Document aggregation failed: {e}")
        return None


def cost_tracking_example():
    """Demonstrate cost tracking functionality."""
    
    print("\n\n=== Cost Tracking Example ===\n")
    
    manager = get_dspy_manager()
    cost_summary = manager.get_cost_summary()
    
    print("Cost Summary:")
    print(f"Total Cost: ${cost_summary['total_cost']:.4f}")
    print(f"Evaluation Count: {cost_summary['evaluation_count']}")
    print(f"Average Cost per Evaluation: ${cost_summary['average_cost_per_evaluation']:.4f}")
    print(f"Budget Remaining: ${cost_summary['budget_remaining']:.2f}")
    
    return cost_summary


def configuration_summary():
    """Show DSPy configuration summary."""
    
    print("\n\n=== Configuration Summary ===\n")
    
    manager = get_dspy_manager()
    config_summary = manager.get_configuration_summary()
    
    print("DSPy Configuration:")
    for key, value in config_summary.items():
        print(f"  {key}: {value}")
    
    return config_summary


def main():
    """Run the basic evaluation example."""
    
    print("🚀 DSPy Basic Evaluation Example")
    print("=" * 50)
    
    try:
        # Run field evaluation example
        field_results = basic_field_evaluation_example()
        
        # Run document aggregation example
        doc_result = document_aggregation_example()
        
        # Show cost tracking
        cost_summary = cost_tracking_example()
        
        # Show configuration
        config_summary = configuration_summary()
        
        print("\n\n✅ Basic evaluation example completed successfully!")
        print("\nKey Takeaways:")
        print("- DSPy provides AI-powered field evaluation")
        print("- Intelligent document-level aggregation")
        print("- Cost tracking and budget management")
        print("- Flexible configuration options")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your API keys in the .env file")
        print("2. Ensure DSPy is properly installed: pip install dspy-ai")
        print("3. Verify your internet connection")
        print("4. Check the DSPy documentation for more details")


if __name__ == "__main__":
    main() 