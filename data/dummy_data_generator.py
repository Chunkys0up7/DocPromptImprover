"""
Enhanced dummy data generator for evaluation testing and demonstration.

This module generates realistic evaluation data for testing the evaluation
framework and demonstrating its capabilities with comprehensive scenarios.
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
        
        # Enhanced sample data templates
        self.invoice_templates = [
            {
                "vendor_name": "Acme Corporation",
                "invoice_number": "INV-2024-001",
                "invoice_date": "2024-01-15",
                "due_date": "2024-02-15",
                "total_amount": "1250.00",
                "tax_amount": "125.00",
                "subtotal": "1125.00",
                "vendor_address": "123 Business St, City, State 12345",
                "vendor_email": "billing@acme.com",
                "vendor_phone": "+1-555-123-4567"
            },
            {
                "vendor_name": "Tech Solutions Inc",
                "invoice_number": "TSI-2024-005",
                "invoice_date": "2024-01-20",
                "due_date": "2024-02-20",
                "total_amount": "875.50",
                "tax_amount": "87.55",
                "subtotal": "787.95",
                "vendor_address": "456 Tech Ave, Silicon Valley, CA 94025",
                "vendor_email": "accounts@techsolutions.com",
                "vendor_phone": "+1-555-987-6543"
            },
            {
                "vendor_name": "Global Services Ltd",
                "invoice_number": "GS-2024-012",
                "invoice_date": "2024-01-30",
                "due_date": "2024-03-01",
                "total_amount": "2100.00",
                "tax_amount": "210.00",
                "subtotal": "1890.00",
                "vendor_address": "789 Global Blvd, International City, IC 12345",
                "vendor_email": "finance@globalservices.com",
                "vendor_phone": "+1-555-456-7890"
            }
        ]
        
        self.receipt_templates = [
            {
                "merchant_name": "Coffee Shop",
                "transaction_date": "2024-01-25",
                "total_amount": "12.75",
                "payment_method": "Credit Card",
                "receipt_number": "RCP-001",
                "merchant_address": "123 Coffee St, Downtown, City",
                "tax_amount": "1.02",
                "subtotal": "11.73"
            },
            {
                "merchant_name": "Grocery Store",
                "transaction_date": "2024-01-26",
                "total_amount": "45.30",
                "payment_method": "Debit Card",
                "receipt_number": "RCP-002",
                "merchant_address": "456 Food Ave, Shopping Center",
                "tax_amount": "3.62",
                "subtotal": "41.68"
            },
            {
                "merchant_name": "Gas Station",
                "transaction_date": "2024-01-27",
                "total_amount": "35.00",
                "payment_method": "Cash",
                "receipt_number": "RCP-003",
                "merchant_address": "789 Fuel Rd, Highway Exit",
                "tax_amount": "2.80",
                "subtotal": "32.20"
            }
        ]
        
        self.form_templates = [
            {
                "applicant_name": "John Doe",
                "email": "john.doe@email.com",
                "phone": "+1-555-123-4567",
                "date_of_birth": "1985-03-15",
                "ssn": "123-45-6789",
                "address": "123 Main St, Anytown, USA 12345",
                "employment_status": "Full-time",
                "annual_income": "75000"
            },
            {
                "applicant_name": "Jane Smith",
                "email": "jane.smith@company.com",
                "phone": "+1-555-987-6543",
                "date_of_birth": "1990-07-22",
                "ssn": "987-65-4321",
                "address": "456 Oak Ave, Somewhere, USA 54321",
                "employment_status": "Part-time",
                "annual_income": "45000"
            },
            {
                "applicant_name": "Bob Johnson",
                "email": "bob.johnson@business.net",
                "phone": "+1-555-456-7890",
                "date_of_birth": "1978-11-08",
                "ssn": "456-78-9012",
                "address": "789 Pine St, Elsewhere, USA 67890",
                "employment_status": "Self-employed",
                "annual_income": "95000"
            }
        ]
        
        # New document types for comprehensive demo
        self.contract_templates = [
            {
                "contract_number": "CTR-2024-001",
                "contract_date": "2024-01-10",
                "expiration_date": "2025-01-10",
                "client_name": "ABC Corporation",
                "client_address": "123 Corporate Blvd, Business City, BC 12345",
                "contract_value": "50000.00",
                "payment_terms": "Net 30",
                "service_description": "Software Development Services"
            },
            {
                "contract_number": "CTR-2024-002",
                "contract_date": "2024-01-15",
                "expiration_date": "2024-12-31",
                "client_name": "XYZ Industries",
                "client_address": "456 Industrial Park, Factory Town, FT 54321",
                "contract_value": "75000.00",
                "payment_terms": "Net 45",
                "service_description": "Consulting Services"
            }
        ]
        
        self.medical_record_templates = [
            {
                "patient_name": "Alice Wilson",
                "patient_id": "P-2024-001",
                "date_of_birth": "1982-05-12",
                "visit_date": "2024-01-28",
                "diagnosis": "Hypertension",
                "prescription": "Lisinopril 10mg daily",
                "doctor_name": "Dr. Sarah Brown",
                "insurance_number": "INS-123-456-789"
            },
            {
                "patient_name": "Charlie Davis",
                "patient_id": "P-2024-002",
                "date_of_birth": "1975-09-30",
                "visit_date": "2024-01-29",
                "diagnosis": "Type 2 Diabetes",
                "prescription": "Metformin 500mg twice daily",
                "doctor_name": "Dr. Michael Johnson",
                "insurance_number": "INS-987-654-321"
            }
        ]
        
        self.bank_statement_templates = [
            {
                "account_number": "1234-5678-9012-3456",
                "statement_date": "2024-01-31",
                "opening_balance": "2500.00",
                "closing_balance": "2875.50",
                "total_deposits": "1500.00",
                "total_withdrawals": "1124.50",
                "bank_name": "First National Bank",
                "account_holder": "John Doe"
            },
            {
                "account_number": "9876-5432-1098-7654",
                "statement_date": "2024-01-31",
                "opening_balance": "5000.00",
                "closing_balance": "4875.25",
                "total_deposits": "2000.00",
                "total_withdrawals": "2124.75",
                "bank_name": "City Bank",
                "account_holder": "Jane Smith"
            }
        ]
        
        # Document type mapping
        self.document_types = {
            "invoice": self.invoice_templates,
            "receipt": self.receipt_templates,
            "form": self.form_templates,
            "contract": self.contract_templates,
            "medical_record": self.medical_record_templates,
            "bank_statement": self.bank_statement_templates
        }
    
    def generate_evaluation_inputs(self, num_documents: int = 50) -> List[DocumentEvaluationInput]:
        """
        Generate comprehensive evaluation inputs for testing.
        
        Args:
            num_documents: Number of documents to generate
            
        Returns:
            List[DocumentEvaluationInput]: Generated evaluation inputs
        """
        
        evaluation_inputs = []
        
        # Generate documents with different success rates to simulate real scenarios
        success_scenarios = [
            (0.9, 0.1),   # 90% success, 10% failure
            (0.7, 0.3),   # 70% success, 30% failure
            (0.5, 0.5),   # 50% success, 50% failure
            (0.3, 0.7),   # 30% success, 70% failure
            (0.1, 0.9),   # 10% success, 90% failure
        ]
        
        for i in range(num_documents):
            # Select document type
            doc_type = random.choice(list(self.document_types.keys()))
            template = random.choice(self.document_types[doc_type])
            
            # Select success scenario
            success_rate, failure_rate = random.choice(success_scenarios)
            
            # Generate ground truth
            ground_truth = template.copy()
            
            # Generate extracted fields with controlled success rate
            extracted_fields = self._generate_extracted_fields(template, success_rate)
            
            # Generate confidence scores
            confidence_scores = self._generate_confidence_scores(extracted_fields)
            
            # Create evaluation input
            evaluation_input = DocumentEvaluationInput(
                document_id=f"{doc_type.upper()}-{i+1:03d}",
                document_type=doc_type,
                extracted_fields=extracted_fields,
                ground_truth=ground_truth,
                confidence_scores=confidence_scores,
                prompt_version=f"v1.{random.randint(1, 5)}",
                evaluation_timestamp=datetime.now() - timedelta(days=random.randint(0, 30))
            )
            
            evaluation_inputs.append(evaluation_input)
        
        return evaluation_inputs
    
    def _generate_extracted_fields(self, template: Dict[str, Any], success_rate: float) -> Dict[str, Any]:
        """
        Generate extracted fields with controlled success rate.
        
        Args:
            template: Ground truth template
            success_rate: Probability of successful extraction
            
        Returns:
            Dict[str, Any]: Generated extracted fields
        """
        
        extracted_fields = {}
        
        for field_name, expected_value in template.items():
            if random.random() < success_rate:
                # Successful extraction with minor variations
                extracted_fields[field_name] = self._add_minor_variations(expected_value, field_name)
            else:
                # Failed extraction
                if random.random() < 0.3:  # 30% chance of missing field
                    continue  # Field not extracted
                else:
                    # 70% chance of wrong value
                    extracted_fields[field_name] = self._generate_error_value(expected_value, field_name)
        
        return extracted_fields
    
    def _add_minor_variations(self, value: Any, field_name: str) -> Any:
        """
        Add minor variations to simulate realistic extraction.
        
        Args:
            value: Original value
            field_name: Name of the field
            
        Returns:
            Any: Value with minor variations
        """
        
        if not value:
            return value
        
        value_str = str(value)
        
        # Field-specific variations
        if "amount" in field_name.lower() or "total" in field_name.lower():
            # Currency variations
            variations = [
                value_str,
                value_str.replace(".", ","),
                f"${value_str}",
                f"USD {value_str}",
                value_str + "0" if not value_str.endswith("0") else value_str[:-1]
            ]
            return random.choice(variations)
        
        elif "date" in field_name.lower():
            # Date format variations
            try:
                date_obj = datetime.strptime(value_str, "%Y-%m-%d")
                variations = [
                    value_str,
                    date_obj.strftime("%m/%d/%Y"),
                    date_obj.strftime("%d-%m-%Y"),
                    date_obj.strftime("%Y/%m/%d"),
                    date_obj.strftime("%B %d, %Y")
                ]
                return random.choice(variations)
            except:
                return value_str
        
        elif "phone" in field_name.lower():
            # Phone number variations
            clean_phone = ''.join(filter(str.isdigit, value_str))
            if len(clean_phone) == 10:
                variations = [
                    value_str,
                    f"({clean_phone[:3]}) {clean_phone[3:6]}-{clean_phone[6:]}",
                    f"{clean_phone[:3]}-{clean_phone[3:6]}-{clean_phone[6:]}",
                    f"+1-{clean_phone[:3]}-{clean_phone[3:6]}-{clean_phone[6:]}"
                ]
                return random.choice(variations)
            return value_str
        
        elif "email" in field_name.lower():
            # Email variations (usually exact)
            return value_str
        
        else:
            # Text variations
            if random.random() < 0.1:  # 10% chance of minor text variation
                variations = [
                    value_str,
                    value_str.title(),
                    value_str.upper(),
                    value_str.lower()
                ]
                return random.choice(variations)
            return value_str
    
    def _generate_error_value(self, expected_value: Any, field_name: str) -> Any:
        """
        Generate error values for failed extractions.
        
        Args:
            expected_value: Expected value
            field_name: Name of the field
            
        Returns:
            Any: Error value
        """
        
        if not expected_value:
            return "N/A"
        
        value_str = str(expected_value)
        
        # Field-specific error patterns
        if "amount" in field_name.lower() or "total" in field_name.lower():
            # Currency errors
            error_patterns = [
                "0.00",
                "999.99",
                str(random.randint(100, 9999)) + ".00",
                "Invalid amount",
                "N/A"
            ]
            return random.choice(error_patterns)
        
        elif "date" in field_name.lower():
            # Date errors
            error_patterns = [
                "01/01/1900",
                "12/31/2099",
                "Invalid date",
                "N/A",
                "00/00/0000"
            ]
            return random.choice(error_patterns)
        
        elif "phone" in field_name.lower():
            # Phone errors
            error_patterns = [
                "000-000-0000",
                "555-555-5555",
                "Invalid phone",
                "N/A",
                "123-456-7890"
            ]
            return random.choice(error_patterns)
        
        elif "email" in field_name.lower():
            # Email errors
            error_patterns = [
                "invalid@email.com",
                "test@test.com",
                "N/A",
                "Invalid email",
                "user@domain"
            ]
            return random.choice(error_patterns)
        
        else:
            # Text errors
            error_patterns = [
                "Invalid value",
                "N/A",
                "Unknown",
                "Error",
                "Not found"
            ]
            return random.choice(error_patterns)
    
    def _generate_confidence_scores(self, extracted_fields: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate realistic confidence scores.
        
        Args:
            extracted_fields: Extracted field values
            
        Returns:
            Dict[str, float]: Confidence scores
        """
        
        confidence_scores = {}
        
        for field_name, value in extracted_fields.items():
            if not value or value in ["N/A", "Invalid", "Error", "Unknown", "Not found"]:
                # Low confidence for error values
                confidence_scores[field_name] = random.uniform(0.1, 0.3)
            elif "amount" in field_name.lower() or "total" in field_name.lower():
                # Medium-high confidence for amounts
                confidence_scores[field_name] = random.uniform(0.7, 0.95)
            elif "date" in field_name.lower():
                # Medium confidence for dates
                confidence_scores[field_name] = random.uniform(0.6, 0.9)
            elif "email" in field_name.lower():
                # High confidence for emails (usually exact)
                confidence_scores[field_name] = random.uniform(0.8, 0.98)
            else:
                # Medium confidence for text
                confidence_scores[field_name] = random.uniform(0.5, 0.85)
        
        return confidence_scores
    
    def generate_evaluation_results(self, num_documents: int = 50) -> List[DocumentEvaluationResult]:
        """
        Generate complete evaluation results for testing.
        
        Args:
            num_documents: Number of documents to generate
            
        Returns:
            List[DocumentEvaluationResult]: Generated evaluation results
        """
        
        evaluation_inputs = self.generate_evaluation_inputs(num_documents)
        evaluation_results = []
        
        for input_data in evaluation_inputs:
            # Generate field evaluations
            field_evaluations = []
            
            for field_name in set(input_data.extracted_fields.keys()) | set(input_data.ground_truth.keys()):
                expected_value = input_data.ground_truth.get(field_name)
                extracted_value = input_data.extracted_fields.get(field_name)
                confidence_score = input_data.confidence_scores.get(field_name, 0.0)
                
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
            
            # Aggregate document results
            document_result = self.document_aggregator.aggregate_evaluations(
                field_evaluations=field_evaluations,
                document_id=input_data.document_id,
                document_type=input_data.document_type,
                confidence_scores=input_data.confidence_scores,
                prompt_version=input_data.prompt_version
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
        
        field_name_lower = field_name.lower()
        
        if any(word in field_name_lower for word in ["date", "birth", "expiration"]):
            return "date"
        elif any(word in field_name_lower for word in ["amount", "total", "tax", "subtotal", "balance", "income", "value"]):
            return "number"
        elif "email" in field_name_lower:
            return "email"
        elif any(word in field_name_lower for word in ["phone", "telephone"]):
            return "phone"
        else:
            return "text"
    
    def generate_edge_cases(self) -> List[DocumentEvaluationInput]:
        """
        Generate edge cases for comprehensive testing.
        
        Returns:
            List[DocumentEvaluationInput]: Edge case evaluation inputs
        """
        
        edge_cases = []
        
        # Case 1: Empty document
        edge_cases.append(DocumentEvaluationInput(
            document_id="EDGE-EMPTY-001",
            document_type="invoice",
            extracted_fields={},
            ground_truth={"vendor_name": "Test Vendor", "total_amount": "100.00"},
            confidence_scores={},
            prompt_version="v1.0"
        ))
        
        # Case 2: All fields missing
        edge_cases.append(DocumentEvaluationInput(
            document_id="EDGE-MISSING-001",
            document_type="receipt",
            extracted_fields={"merchant_name": "Test Merchant"},
            ground_truth={
                "merchant_name": "Test Merchant",
                "total_amount": "50.00",
                "transaction_date": "2024-01-01"
            },
            confidence_scores={"merchant_name": 0.9},
            prompt_version="v1.0"
        ))
        
        # Case 3: Very low confidence scores
        edge_cases.append(DocumentEvaluationInput(
            document_id="EDGE-LOW-CONF-001",
            document_type="form",
            extracted_fields={
                "applicant_name": "John Doe",
                "email": "john@email.com",
                "phone": "555-123-4567"
            },
            ground_truth={
                "applicant_name": "John Doe",
                "email": "john@email.com",
                "phone": "555-123-4567"
            },
            confidence_scores={
                "applicant_name": 0.1,
                "email": 0.05,
                "phone": 0.15
            },
            prompt_version="v1.0"
        ))
        
        # Case 4: Mixed success/failure with realistic variations
        edge_cases.append(DocumentEvaluationInput(
            document_id="EDGE-MIXED-001",
            document_type="contract",
            extracted_fields={
                "contract_number": "CTR-2024-001",
                "contract_date": "01/15/2024",  # Different format
                "client_name": "ABC Corp",  # Truncated
                "contract_value": "50000",  # Missing decimals
                "payment_terms": "Net 30",
                "service_description": "Software Development"  # Missing "Services"
            },
            ground_truth={
                "contract_number": "CTR-2024-001",
                "contract_date": "2024-01-15",
                "client_name": "ABC Corporation",
                "contract_value": "50000.00",
                "payment_terms": "Net 30",
                "service_description": "Software Development Services"
            },
            confidence_scores={
                "contract_number": 0.95,
                "contract_date": 0.8,
                "client_name": 0.7,
                "contract_value": 0.85,
                "payment_terms": 0.9,
                "service_description": 0.75
            },
            prompt_version="v1.0"
        ))
        
        return edge_cases
    
    def generate_comprehensive_demo_data(self, num_documents: int = 100) -> Dict[str, Any]:
        """
        Generate comprehensive demo data for meaningful demonstration.
        
        Args:
            num_documents: Number of documents to generate
            
        Returns:
            Dict[str, Any]: Comprehensive demo data
        """
        
        # Generate regular evaluation inputs
        regular_inputs = self.generate_evaluation_inputs(num_documents)
        
        # Generate edge cases
        edge_cases = self.generate_edge_cases()
        
        # Generate evaluation results
        evaluation_results = self.generate_evaluation_results(num_documents)
        
        # Create comprehensive demo data
        demo_data = {
            "regular_evaluation_inputs": regular_inputs,
            "edge_cases": edge_cases,
            "evaluation_results": evaluation_results,
            "document_type_distribution": self._get_document_type_distribution(regular_inputs),
            "success_rate_distribution": self._get_success_rate_distribution(evaluation_results),
            "field_performance_summary": self._get_field_performance_summary(evaluation_results),
            "generated_at": datetime.now().isoformat(),
            "total_documents": len(regular_inputs) + len(edge_cases),
            "document_types": list(self.document_types.keys())
        }
        
        return demo_data
    
    def _get_document_type_distribution(self, evaluation_inputs: List[DocumentEvaluationInput]) -> Dict[str, int]:
        """Get distribution of document types."""
        distribution = {}
        for input_data in evaluation_inputs:
            doc_type = input_data.document_type
            distribution[doc_type] = distribution.get(doc_type, 0) + 1
        return distribution
    
    def _get_success_rate_distribution(self, evaluation_results: List[DocumentEvaluationResult]) -> Dict[str, float]:
        """Get success rate distribution."""
        total_docs = len(evaluation_results)
        successful_docs = len([r for r in evaluation_results if r.overall_accuracy > 0.8])
        partial_docs = len([r for r in evaluation_results if 0.5 <= r.overall_accuracy <= 0.8])
        failed_docs = len([r for r in evaluation_results if r.overall_accuracy < 0.5])
        
        return {
            "high_accuracy": successful_docs / total_docs if total_docs > 0 else 0,
            "partial_accuracy": partial_docs / total_docs if total_docs > 0 else 0,
            "low_accuracy": failed_docs / total_docs if total_docs > 0 else 0
        }
    
    def _get_field_performance_summary(self, evaluation_results: List[DocumentEvaluationResult]) -> Dict[str, Any]:
        """Get field performance summary."""
        field_stats = {}
        
        for result in evaluation_results:
            for field_eval in result.field_evaluations:
                field_name = field_eval.field_name
                if field_name not in field_stats:
                    field_stats[field_name] = {
                        "total_evaluations": 0,
                        "successful_evaluations": 0,
                        "failed_evaluations": 0,
                        "average_score": 0.0
                    }
                
                field_stats[field_name]["total_evaluations"] += 1
                if field_eval.is_successful():
                    field_stats[field_name]["successful_evaluations"] += 1
                else:
                    field_stats[field_name]["failed_evaluations"] += 1
                field_stats[field_name]["average_score"] += field_eval.evaluation_score
        
        # Calculate averages
        for field_name, stats in field_stats.items():
            if stats["total_evaluations"] > 0:
                stats["average_score"] /= stats["total_evaluations"]
                stats["success_rate"] = stats["successful_evaluations"] / stats["total_evaluations"]
        
        return field_stats
    
    def save_dummy_data(self, filename: str, num_documents: int = 100) -> None:
        """
        Save dummy data to file.
        
        Args:
            filename: Output filename
            num_documents: Number of documents to generate
        """
        
        demo_data = self.generate_comprehensive_demo_data(num_documents)
        
        # Convert to serializable format
        serializable_data = {
            "regular_evaluation_inputs": [input_data.model_dump() for input_data in demo_data["regular_evaluation_inputs"]],
            "edge_cases": [input_data.model_dump() for input_data in demo_data["edge_cases"]],
            "evaluation_results": [result.model_dump() for result in demo_data["evaluation_results"]],
            "document_type_distribution": demo_data["document_type_distribution"],
            "success_rate_distribution": demo_data["success_rate_distribution"],
            "field_performance_summary": demo_data["field_performance_summary"],
            "generated_at": demo_data["generated_at"],
            "total_documents": demo_data["total_documents"],
            "document_types": demo_data["document_types"]
        }
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=str)
        
        print(f"Demo data saved to {filename}")
        print(f"Generated {demo_data['total_documents']} documents across {len(demo_data['document_types'])} document types")
    
    def load_dummy_data(self, filename: str) -> Dict[str, Any]:
        """
        Load dummy data from file.
        
        Args:
            filename: Input filename
            
        Returns:
            Dict[str, Any]: Loaded demo data
        """
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Convert back to model objects
        regular_inputs = [DocumentEvaluationInput(**input_data) for input_data in data["regular_evaluation_inputs"]]
        edge_cases = [DocumentEvaluationInput(**input_data) for input_data in data["edge_cases"]]
        evaluation_results = [DocumentEvaluationResult(**result_data) for result_data in data["evaluation_results"]]
        
        return {
            "regular_evaluation_inputs": regular_inputs,
            "edge_cases": edge_cases,
            "evaluation_results": evaluation_results,
            "document_type_distribution": data["document_type_distribution"],
            "success_rate_distribution": data["success_rate_distribution"],
            "field_performance_summary": data["field_performance_summary"],
            "generated_at": data["generated_at"],
            "total_documents": data["total_documents"],
            "document_types": data["document_types"]
        }


def main():
    """Generate and save comprehensive demo data."""
    
    generator = DummyDataGenerator()
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Generate comprehensive demo data
    print("Generating comprehensive demo data...")
    generator.save_dummy_data("data/comprehensive_demo_data.json", num_documents=100)
    
    # Generate smaller dataset for quick testing
    print("Generating quick test data...")
    generator.save_dummy_data("data/quick_test_data.json", num_documents=20)
    
    print("Demo data generation complete!")


if __name__ == "__main__":
    main() 