"""
Dummy data generator for evaluation testing and demonstration.

This module generates realistic evaluation data for testing the evaluation
framework and demonstrating its capabilities.
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path

from src.models.evaluation_models import (
    DocumentEvaluationInput,
    FieldEvaluationResult,
    DocumentEvaluationResult,
    ExtractionStatus
)
from src.evaluators.field_evaluator import FieldEvaluator
from src.evaluators.document_aggregator import DocumentAggregator


class DummyDataGenerator:
    """
    Generates realistic dummy data for evaluation testing.
    
    This class creates various types of evaluation data including
    successful extractions, failures, and edge cases for comprehensive testing.
    """
    
    def __init__(self):
        """Initialize the dummy data generator."""
        self.field_evaluator = FieldEvaluator()
        self.document_aggregator = DocumentAggregator()
        
        # Sample data templates
        self.invoice_templates = [
            {
                "vendor_name": "Acme Corporation",
                "invoice_number": "INV-2024-001",
                "invoice_date": "2024-01-15",
                "due_date": "2024-02-15",
                "total_amount": "1250.00",
                "tax_amount": "125.00",
                "subtotal": "1125.00"
            },
            {
                "vendor_name": "Tech Solutions Inc",
                "invoice_number": "TSI-2024-005",
                "invoice_date": "2024-01-20",
                "due_date": "2024-02-20",
                "total_amount": "875.50",
                "tax_amount": "87.55",
                "subtotal": "787.95"
            },
            {
                "vendor_name": "Global Services Ltd",
                "invoice_number": "GS-2024-012",
                "invoice_date": "2024-01-30",
                "due_date": "2024-03-01",
                "total_amount": "2100.00",
                "tax_amount": "210.00",
                "subtotal": "1890.00"
            }
        ]
        
        self.receipt_templates = [
            {
                "merchant_name": "Coffee Shop",
                "transaction_date": "2024-01-25",
                "total_amount": "12.75",
                "payment_method": "Credit Card",
                "receipt_number": "RCP-001"
            },
            {
                "merchant_name": "Grocery Store",
                "transaction_date": "2024-01-26",
                "total_amount": "45.30",
                "payment_method": "Debit Card",
                "receipt_number": "RCP-002"
            },
            {
                "merchant_name": "Gas Station",
                "transaction_date": "2024-01-27",
                "total_amount": "35.00",
                "payment_method": "Cash",
                "receipt_number": "RCP-003"
            }
        ]
        
        self.form_templates = [
            {
                "applicant_name": "John Doe",
                "email": "john.doe@email.com",
                "phone": "+1-555-123-4567",
                "date_of_birth": "1985-03-15",
                "ssn": "123-45-6789"
            },
            {
                "applicant_name": "Jane Smith",
                "email": "jane.smith@email.com",
                "phone": "+1-555-987-6543",
                "date_of_birth": "1990-07-22",
                "ssn": "987-65-4321"
            }
        ]
    
    def generate_evaluation_inputs(self, num_documents: int = 10) -> List[DocumentEvaluationInput]:
        """
        Generate evaluation input data.
        
        Args:
            num_documents: Number of documents to generate
            
        Returns:
            List[DocumentEvaluationInput]: List of evaluation inputs
        """
        
        evaluation_inputs = []
        
        for i in range(num_documents):
            # Randomly select document type
            doc_type = random.choice(["invoice", "receipt", "form"])
            
            if doc_type == "invoice":
                template = random.choice(self.invoice_templates)
                extracted_fields = self._generate_extracted_fields(template, success_rate=0.8)
            elif doc_type == "receipt":
                template = random.choice(self.receipt_templates)
                extracted_fields = self._generate_extracted_fields(template, success_rate=0.85)
            else:  # form
                template = random.choice(self.form_templates)
                extracted_fields = self._generate_extracted_fields(template, success_rate=0.75)
            
            # Generate confidence scores
            confidence_scores = self._generate_confidence_scores(extracted_fields)
            
            # Create evaluation input
            evaluation_input = DocumentEvaluationInput(
                document_id=f"{doc_type}_{i+1:03d}",
                document_type=doc_type,
                extracted_fields=extracted_fields,
                ground_truth=template,
                confidence_scores=confidence_scores,
                prompt_version=f"v1.{random.randint(1, 5)}",
                extraction_timestamp=datetime.now() - timedelta(hours=random.randint(1, 24))
            )
            
            evaluation_inputs.append(evaluation_input)
        
        return evaluation_inputs
    
    def _generate_extracted_fields(self, template: Dict[str, Any], success_rate: float) -> Dict[str, Any]:
        """
        Generate extracted fields with realistic errors.
        
        Args:
            template: Ground truth template
            success_rate: Probability of successful extraction
            
        Returns:
            Dict[str, Any]: Extracted fields with errors
        """
        
        extracted_fields = {}
        
        for field_name, expected_value in template.items():
            if random.random() < success_rate:
                # Successful extraction (with minor variations)
                extracted_fields[field_name] = self._add_minor_variations(expected_value, field_name)
            else:
                # Failed extraction
                extracted_fields[field_name] = self._generate_error_value(expected_value, field_name)
        
        return extracted_fields
    
    def _add_minor_variations(self, value: Any, field_name: str) -> Any:
        """
        Add minor variations to successful extractions.
        
        Args:
            value: Original value
            field_name: Name of the field
            
        Returns:
            Any: Value with minor variations
        """
        
        if isinstance(value, str):
            # Add minor formatting variations
            if field_name in ["invoice_date", "due_date", "transaction_date", "date_of_birth"]:
                # Date format variations
                if random.random() < 0.3:
                    return value.replace("-", "/")
                elif random.random() < 0.2:
                    return value.replace("-", ".")
            
            elif field_name in ["total_amount", "tax_amount", "subtotal"]:
                # Number format variations
                if random.random() < 0.2:
                    return f"${value}"
                elif random.random() < 0.1:
                    return value.replace(".", ",")
            
            elif field_name == "phone":
                # Phone format variations
                if random.random() < 0.3:
                    return value.replace("-", " ")
                elif random.random() < 0.2:
                    return value.replace("+1-", "")
        
        return value
    
    def _generate_error_value(self, expected_value: Any, field_name: str) -> Any:
        """
        Generate error values for failed extractions.
        
        Args:
            expected_value: Expected value
            field_name: Name of the field
            
        Returns:
            Any: Error value
        """
        
        error_types = ["missing", "wrong_format", "partial", "completely_wrong"]
        error_type = random.choice(error_types)
        
        if error_type == "missing":
            return None
        
        elif error_type == "wrong_format":
            if field_name in ["invoice_date", "due_date", "transaction_date", "date_of_birth"]:
                # Wrong date format
                return "01/15/2024" if expected_value == "2024-01-15" else "2024-01-15"
            
            elif field_name in ["total_amount", "tax_amount", "subtotal"]:
                # Wrong number format
                return f"${expected_value}" if not str(expected_value).startswith("$") else str(expected_value).replace("$", "")
            
            elif field_name == "phone":
                # Wrong phone format
                return expected_value.replace("-", " ") if "-" in str(expected_value) else str(expected_value).replace(" ", "-")
        
        elif error_type == "partial":
            if isinstance(expected_value, str):
                # Return partial value
                if len(expected_value) > 3:
                    return expected_value[:len(expected_value)//2]
                else:
                    return expected_value[:1] if expected_value else ""
        
        else:  # completely_wrong
            if field_name in ["invoice_date", "due_date", "transaction_date", "date_of_birth"]:
                return "2023-12-31"
            elif field_name in ["total_amount", "tax_amount", "subtotal"]:
                return "999.99"
            elif field_name == "phone":
                return "+1-555-000-0000"
            elif field_name == "email":
                return "wrong@email.com"
            else:
                return "WRONG_VALUE"
        
        return expected_value
    
    def _generate_confidence_scores(self, extracted_fields: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate realistic confidence scores.
        
        Args:
            extracted_fields: Extracted fields
            
        Returns:
            Dict[str, float]: Confidence scores
        """
        
        confidence_scores = {}
        
        for field_name, value in extracted_fields.items():
            if value is None:
                # Missing field - low confidence
                confidence_scores[field_name] = random.uniform(0.1, 0.3)
            elif value == "WRONG_VALUE":
                # Completely wrong - very low confidence
                confidence_scores[field_name] = random.uniform(0.05, 0.2)
            elif len(str(value)) < len(str(field_name)) * 2:
                # Partial value - medium confidence
                confidence_scores[field_name] = random.uniform(0.4, 0.7)
            else:
                # Good value - high confidence
                confidence_scores[field_name] = random.uniform(0.7, 0.95)
        
        return confidence_scores
    
    def generate_evaluation_results(self, num_documents: int = 10) -> List[DocumentEvaluationResult]:
        """
        Generate complete evaluation results.
        
        Args:
            num_documents: Number of documents to generate
            
        Returns:
            List[DocumentEvaluationResult]: List of evaluation results
        """
        
        evaluation_inputs = self.generate_evaluation_inputs(num_documents)
        evaluation_results = []
        
        for evaluation_input in evaluation_inputs:
            # Generate field evaluations
            field_evaluations = []
            
            for field_name in set(evaluation_input.extracted_fields.keys()) | set(evaluation_input.ground_truth.keys()):
                expected_value = evaluation_input.ground_truth.get(field_name)
                extracted_value = evaluation_input.extracted_fields.get(field_name)
                confidence_score = evaluation_input.confidence_scores.get(field_name, 0.0)
                
                # Determine field type
                field_type = self._determine_field_type(field_name, expected_value)
                
                # Evaluate field
                field_result = self.field_evaluator.evaluate_field(
                    field_name=field_name,
                    expected_value=expected_value,
                    extracted_value=extracted_value,
                    confidence_score=confidence_score,
                    field_type=field_type
                )
                field_evaluations.append(field_result)
            
            # Aggregate document result
            document_result = self.document_aggregator.aggregate_evaluations(
                field_evaluations=field_evaluations,
                document_id=evaluation_input.document_id,
                document_type=evaluation_input.document_type,
                confidence_scores=evaluation_input.confidence_scores,
                prompt_version=evaluation_input.prompt_version
            )
            
            evaluation_results.append(document_result)
        
        return evaluation_results
    
    def _determine_field_type(self, field_name: str, value: Any) -> str:
        """
        Determine field type based on field name and value.
        
        Args:
            field_name: Name of the field
            value: Field value
            
        Returns:
            str: Field type
        """
        
        # Date fields
        if any(date_word in field_name.lower() for date_word in ["date", "birth"]):
            return "date"
        
        # Number fields
        if any(num_word in field_name.lower() for num_word in ["amount", "total", "tax", "subtotal", "number"]):
            return "number"
        
        # Email fields
        if "email" in field_name.lower():
            return "email"
        
        # Phone fields
        if "phone" in field_name.lower():
            return "phone"
        
        # Default to text
        return "text"
    
    def generate_edge_cases(self) -> List[DocumentEvaluationInput]:
        """
        Generate edge cases for testing.
        
        Returns:
            List[DocumentEvaluationInput]: Edge case evaluation inputs
        """
        
        edge_cases = []
        
        # Case 1: Empty document
        edge_cases.append(DocumentEvaluationInput(
            document_id="edge_empty",
            document_type="invoice",
            extracted_fields={},
            ground_truth={},
            confidence_scores={},
            prompt_version="v1.0"
        ))
        
        # Case 2: All fields missing
        edge_cases.append(DocumentEvaluationInput(
            document_id="edge_all_missing",
            document_type="invoice",
            extracted_fields={},
            ground_truth=self.invoice_templates[0],
            confidence_scores={},
            prompt_version="v1.0"
        ))
        
        # Case 3: Very low confidence
        edge_cases.append(DocumentEvaluationInput(
            document_id="edge_low_confidence",
            document_type="receipt",
            extracted_fields=self.receipt_templates[0],
            ground_truth=self.receipt_templates[0],
            confidence_scores={field: 0.1 for field in self.receipt_templates[0].keys()},
            prompt_version="v1.0"
        ))
        
        # Case 4: Mixed success/failure
        mixed_extracted = {}
        mixed_confidence = {}
        for field, value in self.form_templates[0].items():
            if random.random() < 0.5:
                mixed_extracted[field] = value
                mixed_confidence[field] = random.uniform(0.8, 0.95)
            else:
                mixed_extracted[field] = None
                mixed_confidence[field] = random.uniform(0.1, 0.3)
        
        edge_cases.append(DocumentEvaluationInput(
            document_id="edge_mixed",
            document_type="form",
            extracted_fields=mixed_extracted,
            ground_truth=self.form_templates[0],
            confidence_scores=mixed_confidence,
            prompt_version="v1.0"
        ))
        
        return edge_cases
    
    def save_dummy_data(self, filename: str, num_documents: int = 50) -> None:
        """
        Save dummy data to a JSON file.
        
        Args:
            filename: Output filename
            num_documents: Number of documents to generate
        """
        
        # Generate evaluation inputs
        evaluation_inputs = self.generate_evaluation_inputs(num_documents)
        
        # Add edge cases
        edge_cases = self.generate_edge_cases()
        evaluation_inputs.extend(edge_cases)
        
        # Convert to dictionaries
        data = {
            "evaluation_inputs": [input_data.dict() for input_data in evaluation_inputs],
            "generated_at": datetime.now().isoformat(),
            "total_documents": len(evaluation_inputs),
            "document_types": list(set(input_data.document_type for input_data in evaluation_inputs))
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Generated {len(evaluation_inputs)} dummy evaluation inputs")
        print(f"Saved to {filename}")
    
    def load_dummy_data(self, filename: str) -> List[DocumentEvaluationInput]:
        """
        Load dummy data from a JSON file.
        
        Args:
            filename: Input filename
            
        Returns:
            List[DocumentEvaluationInput]: List of evaluation inputs
        """
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        evaluation_inputs = []
        for input_dict in data["evaluation_inputs"]:
            evaluation_input = DocumentEvaluationInput(**input_dict)
            evaluation_inputs.append(evaluation_input)
        
        return evaluation_inputs


def main():
    """Generate and save dummy data."""
    
    generator = DummyDataGenerator()
    
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    
    # Generate dummy data
    generator.save_dummy_data("data/dummy_evaluation_data.json", num_documents=100)
    
    # Generate a smaller test set
    generator.save_dummy_data("data/test_evaluation_data.json", num_documents=20)


if __name__ == "__main__":
    main() 